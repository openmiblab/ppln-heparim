import os
import pydicom
import pandas as pd
import numpy as np
import SimpleITK as sitk
import zarr
import zarr.storage
import mdreg
import ants

def get_precise_seconds_from_float(acq_time_str):
    """Parses a DICOM AcquisitionTime string (hhmmss.ffffff) into seconds from midnight[cite: 3]."""
    try:
        time_float = float(acq_time_str)
        hours = int(time_float // 10000)
        minutes = int((time_float % 10000) // 100)
        seconds = time_float % 100
        return (hours * 3600) + (minutes * 60) + seconds
    except (ValueError, TypeError):
        return 0.0

def group_dicom_paths_by_time(series_folder):
    """Scans a folder for DICOM files and groups them by unique AcquisitionTime[cite: 3]."""
    dicom_files = []
    for root_dir, _, files in os.walk(series_folder):
        for f in files:
            full_p = os.path.join(root_dir, f)
            if os.path.isfile(full_p):
                dicom_files.append(full_p)
                
    metadata_cache = []
    for path in dicom_files:
        try:
            hdr = pydicom.dcmread(path, stop_before_pixels=True)
            acq_t = getattr(hdr, 'AcquisitionTime', "000000.00")
            loc = float(getattr(hdr, 'SliceLocation', 0.0))
            metadata_cache.append({'path': path, 'time': acq_t, 'loc': loc})
        except Exception:
            continue

    if not metadata_cache:
        return []

    df = pd.DataFrame(metadata_cache)
    unique_times = sorted(df['time'].unique())
    
    grouped_phases = []
    for t in unique_times:
        phase_df = df[df['time'] == t].sort_values(by='loc')
        grouped_phases.append(phase_df['path'].tolist())
        
    return grouped_phases

def load_and_scale_4d_volume(phases, global_t0, zarr_dir_raw):
    """Reads grouped DICOM slices into a 4D array ordered as (Time, Z, Y, X) and saves to Zarr[cite: 3]."""
    num_timepoints = len(phases)
    first_valid_phase = next(p for p in phases if len(p) > 0)
    num_slices = len(first_valid_phase)
    
    sample_dcm = pydicom.dcmread(first_valid_phase[0])
    rows, cols = sample_dcm.Rows, sample_dcm.Columns
    
    store = zarr.storage.LocalStore(zarr_dir_raw)
    z_raw = zarr.empty(
        shape=(num_timepoints, num_slices, rows, cols), 
        chunks=(1, num_slices, rows, cols), 
        dtype=np.float32, 
        store=store,
        overwrite=True
    )
    
    tacq_list = []
    for t_idx, phase_slices in enumerate(phases):
        if not phase_slices:
            continue
        hdr = pydicom.dcmread(phase_slices[0], stop_before_pixels=True)
        acq_t = getattr(hdr, 'AcquisitionTime', "000000.00")
        current_t = get_precise_seconds_from_float(acq_t)
        tacq_list.append(current_t - global_t0)
        
        for z_idx, slice_path in enumerate(phase_slices):
            dcm = pydicom.dcmread(slice_path)
            arr = dcm.pixel_array.astype(np.float32)
            slope = float(getattr(dcm, 'RescaleSlope', 1.0))
            intercept = float(getattr(dcm, 'RescaleIntercept', 0.0))
            z_raw[t_idx, z_idx, :, :] = (arr * slope) + intercept
            
    return z_raw, tacq_list

def execute_model_driven_motion_correction(z_data, tacq, mask_aorta_t, baseline_frames, zarr_out_path, logger):
    """Performs model-driven motion correction leveraging mdreg framework[cite: 3]."""
    data_array = np.asarray(z_data, dtype=np.float32)
    time_array = np.array(tacq, dtype=np.float32).flatten()
    
    if data_array.shape[0] != len(time_array) and data_array.shape[-1] == len(time_array):
        data_array = np.transpose(data_array, (3, 0, 1, 2))
        
    num_frames = data_array.shape[0]
    num_times = time_array.shape[0]
    min_len = min(num_frames, num_times)
    
    data_array = data_array[:min_len, ...]
    time_array = time_array[:min_len]

    store_coreg = zarr.storage.LocalStore(os.path.join(zarr_out_path, "coreg"))
    store_fit = zarr.storage.LocalStore(os.path.join(zarr_out_path, "fit"))
    
    t_chunks = (1, *data_array.shape[1:])
    coreg_zarr = zarr.empty(shape=data_array.shape, chunks=t_chunks, dtype=data_array.dtype, store=store_coreg, overwrite=True)
    fit_zarr = zarr.empty(shape=data_array.shape, chunks=t_chunks, dtype=data_array.dtype, store=store_fit, overwrite=True)

    aif_curve = []
    if mask_aorta_t is not None:
        if mask_aorta_t.shape != data_array.shape[1:]:
            if mask_aorta_t.shape == data_array.shape[1:][::-1]:
                mask_aorta_t = np.transpose(mask_aorta_t, (2, 1, 0))

        if np.any(mask_aorta_t > 0):
            for t in range(min_len):
                frame = data_array[t, ...]
                aif_curve.append(np.mean(frame[mask_aorta_t > 0]))
        else:
            mask_aorta_t = None

    if mask_aorta_t is None:
        for t in range(min_len):
            frame = data_array[t, ...]
            aif_curve.append(np.mean(frame))
            
    aif_curve = np.array(aif_curve, dtype=np.float32).flatten()

    fit_coreg = {
        'package': 'ants',
        'type_of_transform': 'SyN',
        'metric': 'MI',
        'verbose': False
    }
    
    safe_baseline = int(baseline_frames)
    if safe_baseline >= min_len: safe_baseline = min_len - 1
    if safe_baseline <= 0: safe_baseline = 1
    
    fitting_function = mdreg.fit_2cm_lin if mask_aorta_t is not None else mdreg.fit_constant
    fit_image_params = {'func': fitting_function, 'baseline': safe_baseline}
    
    if fitting_function == mdreg.fit_2cm_lin:
        fit_image_params['time'] = time_array
        fit_image_params['aif'] = aif_curve
        fit_image_params['fixed'] = True

    coreg_res, fit_res, _, _ = mdreg.fit(
        data_array, fit_image=fit_image_params, fit_coreg=fit_coreg, maxit=3, verbose=0
    )
    
    coreg_zarr[:] = coreg_res
    fit_zarr[:] = fit_res
    return coreg_zarr

def register_inter_series_ants(z_moving, static_ref_arr, zarr_out_path):
    """Registers a 4D moving array to a static reference image volume-by-volume using ANTs[cite: 3]."""
    moving_array = np.asarray(z_moving)
    store = zarr.storage.LocalStore(zarr_out_path)
    t_chunks = (1, *moving_array.shape[1:])
    
    registered_zarr = zarr.empty(shape=moving_array.shape, chunks=t_chunks, dtype=moving_array.dtype, store=store, overwrite=True)
    fixed_ants = ants.from_numpy(static_ref_arr.astype(np.float32))
    
    for t in range(moving_array.shape[0]):
        moving_frame = moving_array[t, ...]
        moving_ants = ants.from_numpy(moving_frame.astype(np.float32))
        tx = ants.registration(fixed=fixed_ants, moving=moving_ants, type_of_transform='SyN')
        aligned_frame = tx['warpedmovout'].numpy()
        
        if aligned_frame.shape != moving_frame.shape:
            if aligned_frame.shape == moving_frame.shape[::-1]:
                aligned_frame = np.transpose(aligned_frame, (2, 1, 0))
            else:
                aligned_frame = np.reshape(aligned_frame, moving_frame.shape)
        
        registered_zarr[t, ...] = aligned_frame
        
    return registered_zarr

def save_zarr_as_4d_nifti(z_arr, ref_sitk, out_path):
    """Transforms a 4D array from Zarr into a valid 4D SimpleITK NIfTI image[cite: 3]."""
    np_data = np.asarray(z_arr)
    img_4d = sitk.GetImageFromArray(np_data.astype(np.float32))
    img_4d.SetOrigin(ref_sitk.GetOrigin())
    img_4d.SetDirection(ref_sitk.GetDirection())
    img_4d.SetSpacing((*ref_sitk.GetSpacing(), 1.0))
    sitk.WriteImage(img_4d, out_path)

def extract_registered_time_course(z_final, phases, ts_mask_liver, ts_mask_aorta, ts_mask_portal, global_t0):
    """Extracts regional signal-time dynamics across segmented ROIs[cite: 3]."""
    data_array = np.asarray(z_final)
    mask_liver_t = sitk.GetArrayFromImage(sitk.ReadImage(ts_mask_liver)) if os.path.exists(ts_mask_liver) else None
    mask_aorta_t = sitk.GetArrayFromImage(sitk.ReadImage(ts_mask_aorta)) if os.path.exists(ts_mask_aorta) else None
    mask_portal_t = sitk.GetArrayFromImage(sitk.ReadImage(ts_mask_portal)) if os.path.exists(ts_mask_portal) else None
    
    time_stamps = []
    for phase in phases:
        if phase:
            hdr = pydicom.dcmread(phase[0], stop_before_pixels=True)
            acq_t = getattr(hdr, 'AcquisitionTime', "000000.00")
            time_stamps.append(get_precise_seconds_from_float(acq_t) - global_t0)
            
    records = []
    for t in range(data_array.shape[0]):
        frame = data_array[t, ...]
        records.append({
            'Time_Seconds': time_stamps[t] if t < len(time_stamps) else t,
            'Aorta_Signal': np.mean(frame[mask_aorta_t > 0]) if (mask_aorta_t is not None and np.any(mask_aorta_t > 0)) else 0.0,
            'Portal_Vein_Signal': np.mean(frame[mask_portal_t > 0]) if (mask_portal_t is not None and np.any(mask_portal_t > 0)) else 0.0,
            'Liver_Signal': np.mean(frame[mask_liver_t > 0]) if (mask_liver_t is not None and np.any(mask_liver_t > 0)) else 0.0
        })
    return pd.DataFrame(records)