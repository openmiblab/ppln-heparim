import pydicom
import sys
from collections import defaultdict
import gc
import os
import dicom2nifti
import dicom2nifti.settings as settings
from totalsegmentator.python_api import totalsegmentator
from scipy.ndimage import map_coordinates
import SimpleITK as sitk
import zipfile
import tempfile
import shutil
import numpy as np
import pandas as pd
import re
import glob
import napari
import matplotlib.pyplot as plt
import tkinter as tk
import mdreg.ants as mds
from tkinter import filedialog, messagebox, ttk, simpledialog
from tqdm import tqdm
from scipy.ndimage import map_coordinates
import dcmri as dc

class Tee:
    def __init__(self, file):
        # Add encoding='utf-8' here
        self.file = open(file, 'w', encoding='utf-8')
        self.terminal = sys.stdout
  
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.file.flush()

class RadiologyControlPanel:
    def __init__(self, root, start_callback):
        self.root = root
        self.root.title("HEPARIM - AI Liver Segmentation Control Panel")
        self.root.geometry("600x400")
        self.start_callback = start_callback

        # Variables to store paths
        self.base_drive = tk.StringVar(value=r"G:\Shared drives\HEPARIM\Patients 001-029")
        self.output_base = tk.StringVar(value=r"X:\abdominal_imaging\Shared\HEPARIMTS\batch_processing")
        self.patient_list = tk.StringVar(value="001, 004, 005, 008, 010, 013, 019, 020, 022, 027")

        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill="both", expand=True)

        # --- Input Section ---
        ttk.Label(main_frame, text="Source Data (Base Drive):", font=('Helvetica', 10, 'bold')).pack(anchor="w")
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill="x", pady=(0, 15))
        ttk.Entry(input_frame, textvariable=self.base_drive).pack(side="left", fill="x", expand=True)
        ttk.Button(input_frame, text="Browse", command=self.browse_input).pack(side="right", padx=5)

        # --- Output Section ---
        ttk.Label(main_frame, text="Save Results To:", font=('Helvetica', 10, 'bold')).pack(anchor="w")
        output_frame = ttk.Frame(main_frame)
        output_frame.pack(fill="x", pady=(0, 15))
        ttk.Entry(output_frame, textvariable=self.output_base).pack(side="left", fill="x", expand=True)
        ttk.Button(output_frame, text="Browse", command=self.browse_output).pack(side="right", padx=5)

        # --- Patient IDs ---
        ttk.Label(main_frame, text="Patient IDs (comma separated):", font=('Helvetica', 10, 'bold')).pack(anchor="w")
        ttk.Entry(main_frame, textvariable=self.patient_list).pack(fill="x", pady=(0, 20))

        # --- Start Button ---
        self.start_btn = ttk.Button(main_frame, text="🚀 Start Processing Pipeline", command=self.run_pipeline)
        self.start_btn.pack(pady=20, ipadx=10, ipady=10)

        # --- Status Footer ---
        self.status = ttk.Label(main_frame, text="Status: Ready", foreground="gray")
        self.status.pack(side="bottom")

    def browse_input(self):
        path = filedialog.askdirectory()
        if path: self.base_drive.set(path)

    def browse_output(self):
        path = filedialog.askdirectory()
        if path: self.output_base.set(path)

    def run_pipeline(self):
        # Extract data from GUI
        ids = [i.strip() for i in self.patient_list.get().split(",")]
        base = self.base_drive.get()
        out = self.output_base.get()

        if not os.path.exists(base):
            messagebox.showerror("Error", "Source Drive path does not exist!")
            return

        self.start_btn.config(state="disabled")
        self.status.config(text="Status: Processing... check console for logs", foreground="blue")
        
        # Run the actual pipeline
        self.root.update()
        self.start_callback(ids, base, out)
        
        self.start_btn.config(state="normal")
        self.status.config(text="Status: Batch Complete", foreground="green")
        messagebox.showinfo("Done", f"Processed {len(ids)} patients successfully.")


def apply_3d_field(vol, field):
    """Memory efficient coordinate mapping."""
    # Using float32 for coords reduces memory vs default float64
    z, y, x = vol.shape
    coords = np.indices((z, y, x), dtype=np.float32)
    
    # Mapping logic (DZ, DY, DX)
    new_coords = [
        coords[0] + field[..., 0],
        coords[1] + field[..., 1],
        coords[2] + field[..., 2]
    ]
    
    # Perform warp and explicitly cast to float32
    warped = map_coordinates(vol.astype(np.float32), new_coords, order=1, mode='constant', cval=0)
    
    # Cleanup coords immediately
    del coords, new_coords
    return warped

def register_series(post_ref_arr, moving_series_paths, output_path, post_ref_img):
    """
    Registers any timecourse (PRE or POST) to a single reference volume.
    Streaming approach ensures minimal RAM usage.
    """
    reader = sitk.ImageSeriesReader()
    post_ref_arr = post_ref_arr.astype(np.float32)
    
    # 1. Calculate the deformation field
    # We use the FIRST volume of the series to calculate the shift to the reference
    reader.SetFileNames(moving_series_paths[0])
    series_start_arr = sitk.GetArrayFromImage(reader.Execute()).astype(np.float32)
    
    print("🔄 Calculating field: Anchor Reference vs Series Start...")
    _, defo = mds.coreg(
    post_ref_arr, 
    series_start_arr,
    return_transfo=True, 
    attachment=15, 
    type='rigid'       
    )
    
    del series_start_arr
    gc.collect()

    # 2. Initialize 4D Container (Float32 for intensity preservation)
    num_phases = len(moving_series_paths)
    z, y, x = post_ref_arr.shape
    series_4d = sitk.Image([x, y, z, num_phases], sitk.sitkFloat32)
    
    # Apply spatial metadata
    series_4d.SetSpacing((*post_ref_img.GetSpacing(), 1.0))
    series_4d.SetOrigin((*post_ref_img.GetOrigin(), 0.0))
    
    # Build 4x4 Direction matrix
    dir_3d = np.array(post_ref_img.GetDirection()).reshape(3, 3)
    dir_4d = np.eye(4)
    dir_4d[:3, :3] = dir_3d
    series_4d.SetDirection(dir_4d.flatten())

    # 3. Registration & Streaming Loop
    print(f"🔄 Streaming {num_phases} phases to {os.path.basename(output_path)}...")
    for i in tqdm(range(num_phases), desc="Registering Phases", unit="vol"):
        reader.SetFileNames(moving_series_paths[i])
        vol = sitk.GetArrayFromImage(reader.Execute()).astype(np.float32)
        
        # Apply the pre-calculated field
        #reg_vol = apply_3d_field(vol, defo)
        reg_vol = mds.transform(vol, defo, interpolator='linear')

        
        # Paste into 4D stack
        reg_vol_sitk = sitk.Cast(sitk.GetImageFromArray(reg_vol), sitk.sitkFloat32)
        series_4d = sitk.Paste(series_4d, reg_vol_sitk, reg_vol_sitk.GetSize(), [0, 0, 0, 0], [0, 0, 0, i])
        
        del vol, reg_vol, reg_vol_sitk
        if i % 10 == 0:
            gc.collect()

    # 4. Final Save
    sitk.WriteImage(series_4d, output_path)
    del series_4d
    gc.collect()
    
    return True

def visualize_heparim_results(patient_dir, pid):
    # 1. Define paths
    pre_path = os.path.join(patient_dir, f"patient_{pid}_mri_PRE_Registered_4D.nii.gz")
    post_path = os.path.join(patient_dir, f"patient_{pid}_mri_POST_Registered_4D.nii.gz")
    
    liver_nifti = os.path.join(patient_dir, "ts_liver_segments.nii.gz")
    aorta_nifti = os.path.join(patient_dir, "ts_aorta_segment.nii.gz")
    portal_nifti = os.path.join(patient_dir, "ts_portal_segment.nii.gz")

    # 2. Initialize Napari Viewer
    viewer = napari.Viewer(title=f"HEPARIM Review - Patient {pid}")

    # 3. Add MRI Images (Check existence first)
    if os.path.exists(pre_path):
        img_pre = sitk.GetArrayFromImage(sitk.ReadImage(pre_path))
        viewer.add_image(img_pre, name='MRI_PRE', colormap='gray', visible=True)
    
    if os.path.exists(post_path):
        img_post = sitk.GetArrayFromImage(sitk.ReadImage(post_path))
        viewer.add_image(img_post, name='MRI_POST', colormap='gray', visible=False)

    # 4. Process and Add Mirrored Masks
    # Axis 2 flip corrects the LPS (DICOM) to RAS (Napari) mirroring
    if os.path.exists(liver_nifti):
        m_liver = sitk.GetArrayFromImage(sitk.ReadImage(liver_nifti))
        viewer.add_labels(np.flip(m_liver, axis=1).astype(np.uint8), name='Liver_Segments', opacity=0.5)

    if os.path.exists(aorta_nifti):
        m_aorta = sitk.GetArrayFromImage(sitk.ReadImage(aorta_nifti))
        viewer.add_labels(np.flip(m_aorta, axis=1).astype(np.uint8), name='Aorta', colormap={1: 'red'}, opacity=0.8)

    if os.path.exists(portal_nifti):
        m_pv = sitk.GetArrayFromImage(sitk.ReadImage(portal_nifti))
        viewer.add_labels(np.flip(m_pv, axis=1).astype(np.uint8), name='Portal_Vein', colormap={1: 'blue'}, opacity=0.8)

    print(f"✅ Viewer launched for Patient {pid}")
    napari.run()
       
        
def get_best_time(ds):
    """
    Finds the most accurate relative time in seconds for DCE fitting.
    """
    # 1. Try TriggerTime (0018, 1060) - often used for relative sequence timing
    # DICOM stores this in milliseconds, so we divide by 1000
    trigger_time = ds.get('TriggerTime')
    if trigger_time is not None:
        return float(trigger_time) / 1000.0

    # 2. Fallback to AcquisitionTime (0008, 0032) - absolute clock
    # Must convert HHMMSS.ffffff to total seconds
    acq_time = ds.get('AcquisitionTime')
    if acq_time:
        t_str = str(acq_time)
        try:
            hms = t_str.split('.')[0]
            ms = t_str.split('.')[1] if '.' in t_str else "0"
            hh, mm, ss = int(hms[:2]), int(hms[2:4]), int(hms[4:6])
            return (hh * 3600) + (mm * 60) + ss + float("0." + ms)
        except:
            return 0.0
            
    return 0.0




def group_dicom_paths_by_time(dicom_dir):
    """
    Groups DICOM file paths by AcquisitionTime and sorts them spatially.
    Requires ZERO write permissions.
    """
    print(f"--- Scanning DICOM Headers: {os.path.basename(dicom_dir)} ---")
    
    dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if os.path.isfile(os.path.join(dicom_dir, f))]
    time_groups = defaultdict(list)
    
    for f in dicom_files:
        try:
            # Read only the header (extremely fast)
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            
            # Group by time
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            precise_time = get_best_time(ds)
    
            # We use a rounded time as a key to group slices into the same phase
            # (Accounting for tiny millisecond jitters between slices)
            t_key = round(precise_time, 2)

            
            # Sort spatially by Z-axis position (fallback to InstanceNumber)
            try:
                z_pos = float(ds.ImagePositionPatient[2])
            except:
                z_pos = float(ds.get('InstanceNumber', 0))
                
            time_groups[t_key].append((z_pos, f))
        except:
            continue
            
    # Chronological sort
    sorted_times = sorted(time_groups.keys())
    
    grouped_filepaths = []
    for t in sorted_times:
        # Spatial sort (ensures slices are stacked in the correct order)
        # Note: If your Z-axis is inverted, you may need reverse=True
        sorted_by_z = sorted(time_groups[t], key=lambda x: x[0])
        just_paths = [x[1] for x in sorted_by_z]
        grouped_filepaths.append(just_paths)
        
    print(f"✅ Found {len(grouped_filepaths)} time points.")
    return grouped_filepaths



def get_precise_seconds_from_float(ds_or_val):
    """Converts DICOM AcquisitionTime (HHMMSS.ffffff) to absolute seconds from midnight."""
    # Safety: check if we were passed a DICOM object or just the value
    val = ds_or_val.AcquisitionTime if hasattr(ds_or_val, 'AcquisitionTime') else ds_or_val
    
    t_str = f"{float(val):010.3f}"
    hms, ms = t_str.split('.')
    hh, mm, ss = int(hms[:2]), int(hms[2:4]), int(hms[4:6])
    return (hh * 3600) + (mm * 60) + ss + float("0." + ms)


def extract_registered_time_course(all_registered_arrays, grouped_filepaths, liver_nifti, aorta_nifti, pv_nifti, t0=None):
    results = []

    # Load Masks
    m_liver = sitk.GetArrayFromImage(sitk.ReadImage(liver_nifti))
    m_aorta = sitk.GetArrayFromImage(sitk.ReadImage(aorta_nifti))
    m_pv = sitk.GetArrayFromImage(sitk.ReadImage(pv_nifti))
    
    # Mirroring (Standard LPS -> RAS fix)
    mask_liver = np.flip(m_liver, axis=1)
    mask_aorta = np.flip(m_aorta, axis=1)
    mask_pv = np.flip(m_pv, axis=1)

    # 1. Pre-calculate Segment Volumes (Voxel Counts) for weighting
    valid_segments = [1, 2, 3, 4, 5, 6, 8]
    seg_volumes = {i: np.sum(mask_liver == i) for i in valid_segments}
    total_liver_volume = sum(seg_volumes.values())

    # 2. Establish t0 if not provided
    if t0 is None:
        first_path = grouped_filepaths[0][0]
        ds_first = pydicom.dcmread(first_path, stop_before_pixels=True)
        t0 = get_precise_seconds_from_float(ds_first.AcquisitionTime)

    # 3. Iterate through phases
    for phase_idx, phase_arr in enumerate(all_registered_arrays):
        current_path = grouped_filepaths[phase_idx][0]
        ds = pydicom.dcmread(current_path, stop_before_pixels=True)
        
        current_abs_time = get_precise_seconds_from_float(ds.AcquisitionTime)
        relative_time = current_abs_time - t0
        
        slope = float(ds.get("RescaleSlope", 1))
        intercept = float(ds.get("RescaleIntercept", 0))
        scaled_arr = phase_arr.astype(np.float64) * slope + intercept
        
        # Base data for this timepoint
        phase_data = {
            "Global_Phase": phase_idx,
            "Time": relative_time,
            "Aorta": np.mean(scaled_arr[mask_aorta > 0]) if np.any(mask_aorta > 0) else 1e-6,
            "Portal_Vein": np.mean(scaled_arr[mask_pv > 0]) if np.any(mask_pv > 0) else 1e-6,
        }

        # Calculate individual segments and track weighted sum
        weighted_signal_sum = 0
        for i in valid_segments:
            seg_mask = (mask_liver == i)
            if np.any(seg_mask):
                sig = np.mean(scaled_arr[seg_mask])
                phase_data[f"S{i}"] = sig
                # Add to weighted sum: (Signal * Segment Volume)
                weighted_signal_sum += sig * seg_volumes[i]
            else:
                phase_data[f"S{i}"] = 1e-6

        # 4. Add the Volume-Weighted Whole Liver column
        if total_liver_volume > 0:
            phase_data["Whole_Liver"] = weighted_signal_sum / total_liver_volume
        else:
            phase_data["Whole_Liver"] = 1e-6
            
        results.append(phase_data)

    return pd.DataFrame(results)
        
def main_pipeline_execution():
    for pid in PATIENT_IDS: 
        try:
            patient_search = os.path.join(BASE_DRIVE, f"Patient {pid}*")
            patient_folders = glob.glob(patient_search)
            
            if not patient_folders:
                print(f"❌ Folder not found for Patient {pid}")
                continue
            
            patient_root = patient_folders[0]
            patient_out_dir = os.path.join(OUTPUT_BASE, f"Patient_{pid}")
            os.makedirs(patient_out_dir, exist_ok=True)
            # 1. Target Specific Phases (Fixing the overlap)
            # First, find ALL folders containing "_dyn_"
            dyn_search_pattern = os.path.join(patient_root, "*_dyn_*")
            all_dyn_folders = [f for f in glob.glob(dyn_search_pattern) if os.path.isdir(f)]

            # Now explicitly filter them into pre and post
            pre_folders = [f for f in all_dyn_folders if "_dyn_post" not in os.path.basename(f).lower()]
            post_folders = [f for f in all_dyn_folders if "_dyn_post" in os.path.basename(f).lower()]
            if not pre_folders or not post_folders:
                print(f"⚠️ Missing either the initial 'dyn' or 'dyn_post' for Patient {pid}")
                continue

            # Now pre_folders[0] is guaranteed to be the baseline, 
            # and post_folders[0] is guaranteed to be the post-contrast.
            print(f"📂 Selected PRE: {os.path.basename(pre_folders[0])}")
            print(f"📂 Selected POST: {os.path.basename(post_folders[0])}")
            # 2. Setup output paths
            patient_out_dir = os.path.join(OUTPUT_BASE, f"Patient_{pid}")
            os.makedirs(patient_out_dir, exist_ok=True)
            ts_mask_liver = os.path.join(patient_out_dir, "ts_liver_segments.nii.gz")
            ts_mask_aorta = os.path.join(patient_out_dir, "ts_aorta_segment.nii.gz")
            ts_mask_portal = os.path.join(patient_out_dir, "ts_portal_segment.nii.gz")
            
            # 3. Read File Paths In-Memory
            # We separate them here so we know exactly which one is the 'pre' and 'post' start
            pre_phases = group_dicom_paths_by_time(pre_folders[0])
            post_phases = group_dicom_paths_by_time(post_folders[0])
            
            # Combine them for the full time-course extraction later
            all_phase_paths = pre_phases + post_phases
            
            if not all_phase_paths:
                print("❌ No valid DICOM slices found.")
                continue
            
            # 4. Generate Reference NIfTIs for both Pre and Post
            print("🔄 Creating Pre-Contrast Reference NIfTI...")
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(pre_phases[0]) # First phase of the pre-dynamic set
            ref_img_pre = reader.Execute()
            ref_nifti_pre = os.path.join(patient_out_dir, f"patient_{pid}_mri_PRE.nii.gz")
            sitk.WriteImage(ref_img_pre, ref_nifti_pre)
            
            # --- POST-CONTRAST REFERENCE ---
            print("🔄 Creating Post-Contrast (HBP) Reference NIfTI...")
            reader = sitk.ImageSeriesReader()
            # We take the first phase of the post-contrast folder (usually the HBP or late portal)
            reader.SetFileNames(post_phases[0]) 
            ref_img_post = reader.Execute()
            ref_nifti_post = os.path.join(patient_out_dir, f"patient_{pid}_mri_POST.nii.gz")
            sitk.WriteImage(ref_img_post, ref_nifti_post)

            # --- 5. Registration ---
            print("🔄 Registering Pre and Post-Contrast acquasitions...")
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(post_phases[0])
            post_ref_img = reader.Execute()
            post_ref_arr = sitk.GetArrayFromImage(post_ref_img).astype(np.float32)
            
            pre_out_path = os.path.join(patient_out_dir, f"patient_{pid}_mri_PRE_Registered_4D.nii.gz")
            post_out_path = os.path.join(patient_out_dir, f"patient_{pid}_mri_POST_Registered_4D.nii.gz")
            # 1. Run the registration to get the list of numpy arrays
            register_series(post_ref_arr, post_phases, post_out_path, post_ref_img)
            print(f"✅ Successfully saved 4D registered PRE NIfTI: {post_out_path}")
            register_series(post_ref_arr, pre_phases, pre_out_path, post_ref_img)       
            print(f"✅ Successfully saved 4D registered PRE NIfTI: {pre_out_path}")
            
            visualize_heparim_results(patient_out_dir,pid)
                        
            # Define paths to the 4D files we just saved
            pre_nifti_path = os.path.join(patient_out_dir, f"patient_{pid}_mri_PRE_Registered_4D.nii.gz")
            # Note: Ensure you saved a POST NIfTI using the same logic as the PRE NIfTI
            post_nifti_path = os.path.join(patient_out_dir, f"patient_{pid}_mri_POST_Registered_4D.nii.gz")
            
            # --- 6. Extract Signal Intensity from the saved NIfTIs ---
            print(f"📊 Extracting SI for Patient {pid}...")

            # A. Find the GLOBAL baseline time (t0) from the first PRE phase
            # This ensures PRE and POST are on the same chronological timeline
            first_pre_slice = pre_phases[0][0]
            ds_baseline = pydicom.dcmread(first_pre_slice, stop_before_pixels=True)
            global_t0 = get_precise_seconds_from_float(ds_baseline.AcquisitionTime)
            print(f"Start time {global_t0}...")

            # B. Load your 4D arrays
            pre_4d_arr = sitk.GetArrayFromImage(sitk.ReadImage(pre_nifti_path))
            post_4d_arr = sitk.GetArrayFromImage(sitk.ReadImage(post_nifti_path))
            
            # B. Extract Pre and Post using that same global_t0
            print(f"📊 Extracting PRE-contrast SI for Patient {pid}...")
            df_pre = extract_registered_time_course(pre_4d_arr, pre_phases, ts_mask_liver, ts_mask_aorta, ts_mask_portal, t0=global_t0)
            
            # A. Find the GLOBAL baseline time (t0) from the first PRE phase
            # This ensures PRE and POST are on the same chronological timeline
            first_post_slice = post_phases[0][0]
            ds_baseline = pydicom.dcmread(first_post_slice, stop_before_pixels=True)
            global_t0 = get_precise_seconds_from_float(ds_baseline.AcquisitionTime)
            print(f"Start time {global_t0}...")
            
            
            print(f"📊 Extracting POST-contrast SI for Patient {pid}...")
            df_post = extract_registered_time_course(post_4d_arr, post_phases, ts_mask_liver, ts_mask_aorta, ts_mask_portal, t0=global_t0)
            
            # C. Save
            csv_path = os.path.join(patient_out_dir, f"patient_{pid}_pre_timecourse.csv")
            df_pre.to_csv(csv_path, index=False)
            csv_path = os.path.join(patient_out_dir, f"patient_{pid}_post_timecourse.csv")
            df_post.to_csv(csv_path, index=False)
            
        except Exception as e:
            print(f"❌ Critical failure on Patient {pid}: {e}")
        # Restore stdout
        sys.stdout = sys.__stdout__
                       
def run_batch_logic(ids, base, out):
    """ This function bridges the GUI to your existing processing loop """
    global PATIENT_IDS, BASE_DRIVE, OUTPUT_BASE
    PATIENT_IDS = ids
    BASE_DRIVE = base
    OUTPUT_BASE = out
    
    # Trigger the processing logic. 
    # It no longer needs 'out' passed as an argument because it uses the global OUTPUT_BASE
    main_pipeline_execution()

if __name__ == "__main__":
    root = tk.Tk()
    app = RadiologyControlPanel(root, run_batch_logic)
    root.mainloop()
