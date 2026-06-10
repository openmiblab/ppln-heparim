import pydicom
import sys
from collections import defaultdict
import gc
import os
import time
import threading
import SimpleITK as sitk
import numpy as np
import pandas as pd
import glob
import napari
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tqdm import tqdm

import mdreg
import mdreg.ants as mds

import matplotlib
matplotlib.use('Agg')  # Prevents background thread rendering window locks
import matplotlib.pyplot as plt

# ==========================================
# I/O AND UTILITY METHODS
# ==========================================

class Tee:
    def __init__(self, file):
        self.file = open(file, 'w', encoding='utf-8')
        self.terminal = sys.stdout
  
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.file.flush()


def get_precise_seconds_from_float(ds_or_val):
    val = ds_or_val.AcquisitionTime if hasattr(ds_or_val, 'AcquisitionTime') else ds_or_val
    t_str = f"{float(val):010.3f}"
    hms, ms = t_str.split('.')
    hh, mm, ss = int(hms[:2]), int(hms[2:4]), int(hms[4:6])
    return (hh * 3600) + (mm * 60) + ss + float("0." + ms)


def get_best_time(ds):
    trigger_time = ds.get('TriggerTime')
    if trigger_time is not None:
        return float(trigger_time) / 1000.0

    acq_time = ds.get('AcquisitionTime')
    if acq_time:
        return get_precise_seconds_from_float(acq_time)
    return 0.0


def group_dicom_paths_by_time(dicom_dir):
    print(f"--- Scanning DICOM Headers: {os.path.basename(dicom_dir)} ---")
    dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if os.path.isfile(os.path.join(dicom_dir, f))]
    time_groups = defaultdict(list)
    
    for f in dicom_files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            precise_time = get_best_time(ds)
            t_key = round(precise_time, 2)

            try:
                z_pos = float(ds.ImagePositionPatient[2])
            except:
                z_pos = float(ds.get('InstanceNumber', 0))
                
            time_groups[t_key].append((z_pos, f))
        except:
            continue
            
    sorted_times = sorted(time_groups.keys())
    grouped_filepaths = []
    for t in sorted_times:
        sorted_by_z = sorted(time_groups[t], key=lambda x: x[0])
        just_paths = [x[1] for x in sorted_by_z]
        grouped_filepaths.append(just_paths)
        
    print(f"✅ Found {len(grouped_filepaths)} time points.")
    return grouped_filepaths


# ==========================================
# REFACTORED CORE PIPELINE METHODS
# ==========================================

def load_and_scale_4d_volume(phase_groups, global_t0):
    """
    Assembles a scaled 4D volume from clustered DICOM slices.
    Returns:
        data: NumPy array formatted for mdreg [X, Y, Z, Time]
        tacq: Array of real acquisition times relative to global_t0
    """
    reader = sitk.ImageSeriesReader()
    tacq = []
    volumes = []
    
    for phase in phase_groups:
        ds_slice = pydicom.dcmread(phase[0], stop_before_pixels=True)
        tacq.append(get_precise_seconds_from_float(ds_slice.AcquisitionTime) - global_t0)
        
        reader.SetFileNames(phase)
        vol_arr = sitk.GetArrayFromImage(reader.Execute()).astype(np.float32)
        slope = float(ds_slice.get("RescaleSlope", 1))
        intercept = float(ds_slice.get("RescaleIntercept", 0))
        volumes.append(vol_arr * slope + intercept)
    
    # Transpose from SimpleITK native shape [Z, Y, X, T] to mdreg matrix format [X, Y, Z, T]
    data_cube = np.stack(volumes, axis=-1)
    data = np.transpose(data_cube, (2, 1, 0, 3))
    return data, np.array(tacq)


def execute_model_driven_motion_correction(data, tacq, mask_aorta_t, baseline_frames, output_diagnostics_path):
    """
    Performs independent model-driven coregistration via mdreg.
    Prevents C++ integer overflows by handling diagnostics writing manually.
    Returns:
        coreg: Cleaned and motion-corrected 4D array [X, Y, Z, Time]
    """
    # Dynamically extract the localized initialization AIF signature across the time axis
    aif = np.array([
        np.mean(data[..., t][mask_aorta_t > 0]) if np.any(mask_aorta_t > 0) else 1e-6 
        for t in range(data.shape[-1])
    ])
    
    t_start = time.time()
    
    # 🛑 CRITICAL FIX: Set path=None to stop mdreg from causing an ITK integer overflow
    coreg, fit, transfo, pars = mdreg.fit(
        data,
        fit_image={
            'func': mdreg.fit_2cm_lin, 
            'time': tacq, 
            'aif': aif, 
            'baseline': baseline_frames
        },
        maxit=3,
        path=None, 
        verbose=2,
    )
    print(f"Computation time: {round(time.time() - t_start)} seconds.")
    
    # 💾 SAFE DIAGNOSTICS EXPORT: Manually save small text files instead of huge 4D displacement fields
    try:
        os.makedirs(output_diagnostics_path, exist_ok=True)
        np.save(os.path.join(output_diagnostics_path, "fit_parameters.npy"), pars)
        print(f"📁 Safely exported text diagnostics to {output_diagnostics_path}")
    except Exception as save_err:
        print(f"⚠️ Non-critical diagnostics export failure: {save_err}")

    return coreg


def generate_and_save_mdreg_animation(coreg_data, raw_input_data, destination_path, title_prefix, is_pre_phase=True):
    """
    Generates an mdreg model-driven coregistration evaluation animation 
    and writes it directly to disk as an animated GIF.
    """
    print(f"🎬 Creating motion correction diagnostic animation for {title_prefix}...")
    
    num_frames = coreg_data.shape[-1]
    if is_pre_phase:
        t0, t1 = 0, num_frames
    else:
        t0 = 0
        t1 = min(15, num_frames)  # Focus explicitly on peak dynamic transition frames
        
    # Select absolute center slice [Z] for clean 2D projection
    center_z = coreg_data.shape[2] // 2
    coreg_sliced = coreg_data[:, :, center_z, t0:t1]
    
    try:
        anim = mdreg.plot.animation(
            coreg_sliced,
            title=f'{title_prefix} DCE motion corrected',
            vmin=0,
            vmax=0.9 * np.max(raw_input_data[..., 0]),
        )
        
        output_file = os.path.join(destination_path, f"{title_prefix.lower()}_motion_correction.gif")
        anim.save(output_file, writer='pillow', fps=4)
        print(f"💾 Diagnostic animation written successfully to: {output_file}")
        
        plt.close('all')
        del anim
        
    except Exception as animation_error:
        print(f"⚠️ Animation generation skipped for {title_prefix}: {animation_error}")
        plt.close('all')


def register_inter_series_ants(moving_4d_arr, static_ref_arr):
    """
    Aligns separate dynamic acquisition runs to the specified baseline ref NIfTI.
    Uses ANTs rigid transformations to handle cross-volume spatial drift.
    Returns:
        registered_4d_arr: Aligned array in [X, Y, Z, Time] format
    """
    print("🔄 Calculating ANTs rigid alignment field to baseline reference NIfTI...")
    
    moving_anchor_3d = moving_4d_arr[..., 0].astype(np.float32)
    static_target_3d = static_ref_arr.astype(np.float32)
    
    _, deformation_field = mds.coreg(
        static_target_3d, 
        moving_anchor_3d,
        return_transfo=True, 
        attachment=15, 
        type='rigid'
    )
    
    registered_phases = []
    num_phases = moving_4d_arr.shape[-1]
    
    for t in range(num_phases):
        vol_3d = moving_4d_arr[..., t]
        warped_vol = mds.transform(vol_3d, deformation_field, interpolator='linear')
        registered_phases.append(warped_vol)
        
    return np.stack(registered_phases, axis=-1)


def save_array_as_4d_nifti(data_xyz_t, ref_geom_sitk, destination_path):
    """Converts an [X, Y, Z, Time] matrix back to [T, Z, Y, X] and saves it via SimpleITK."""
    data_t_z_y_x = np.transpose(data_xyz_t, (3, 2, 1, 0))
    sitk_image = sitk.GetImageFromArray(data_t_z_y_x.astype(np.float32))
    sitk_image.SetSpacing((*ref_geom_sitk.GetSpacing(), 1.0))
    sitk_image.SetOrigin((*ref_geom_sitk.GetOrigin(), 0.0))
    sitk.WriteImage(sitk_image, destination_path)
    print(f"💾 NIfTI file cleanly saved: {destination_path}")


def extract_registered_time_course(corrected_xyz_t, phase_paths, liver_nifti, aorta_nifti, pv_nifti, global_t0):
    """Extracts signal-intensity variations over time across spatial segmentations."""
    results = []

    m_liver = sitk.GetArrayFromImage(sitk.ReadImage(liver_nifti))
    m_aorta = sitk.GetArrayFromImage(sitk.ReadImage(aorta_nifti))
    m_pv = sitk.GetArrayFromImage(sitk.ReadImage(pv_nifti))
    
    mask_liver = np.flip(m_liver, axis=1)
    mask_aorta = np.flip(m_aorta, axis=1)
    mask_pv = np.flip(m_pv, axis=1)

    valid_segments = [1, 2, 3, 4, 5, 6, 8]
    seg_volumes = {i: np.sum(mask_liver == i) for i in valid_segments}
    total_liver_volume = sum(seg_volumes.values())

    num_phases = corrected_xyz_t.shape[-1]

    for phase_idx in range(num_phases):
        current_path = phase_paths[phase_idx][0]
        ds = pydicom.dcmread(current_path, stop_before_pixels=True)
        
        current_abs_time = get_precise_seconds_from_float(ds.AcquisitionTime)
        relative_time = current_abs_time - global_t0
        
        slope = float(ds.get("RescaleSlope", 1))
        intercept = float(ds.get("RescaleIntercept", 0))
        
        vol_xyz = corrected_xyz_t[..., phase_idx]
        scaled_arr = np.transpose(vol_xyz, (2, 1, 0)).astype(np.float64) * slope + intercept
        
        phase_data = {
            "Global_Phase": phase_idx,
            "Time": relative_time,
            "Aorta": np.mean(scaled_arr[mask_aorta > 0]) if np.any(mask_aorta > 0) else 1e-6,
            "Portal_Vein": np.mean(scaled_arr[mask_pv > 0]) if np.any(mask_pv > 0) else 1e-6,
        }

        weighted_signal_sum = 0
        for i in valid_segments:
            seg_mask = (mask_liver == i)
            if np.any(seg_mask):
                sig = np.mean(scaled_arr[seg_mask])
                phase_data[f"S{i}"] = sig
                weighted_signal_sum += sig * seg_volumes[i]
            else:
                phase_data[f"S{i}"] = 1e-6

        if total_liver_volume > 0:
            phase_data["Whole_Liver"] = weighted_signal_sum / total_liver_volume
        else:
            phase_data["Whole_Liver"] = 1e-6
            
        results.append(phase_data)

    return pd.DataFrame(results)


def visualize_heparim_results(patient_dir, pid):
    pre_path = os.path.join(patient_dir, f"patient_{pid}_mri_PRE_Registered_4D.nii.gz")
    post_path = os.path.join(patient_dir, f"patient_{pid}_mri_POST_Registered_4D.nii.gz")
    
    liver_nifti = os.path.join(patient_dir, "ts_liver_segments.nii.gz")
    aorta_nifti = os.path.join(patient_dir, "ts_aorta_segment.nii.gz")
    portal_nifti = os.path.join(patient_dir, "ts_portal_segment.nii.gz")

    viewer = napari.Viewer(title=f"HEPARIM Review - Patient {pid}")

    if os.path.exists(pre_path):
        img_pre = sitk.GetArrayFromImage(sitk.ReadImage(pre_path))
        viewer.add_image(img_pre, name='MRI_PRE', colormap='gray', visible=True)
    
    if os.path.exists(post_path):
        img_post = sitk.GetArrayFromImage(sitk.ReadImage(post_path))
        viewer.add_image(img_post, name='MRI_POST', colormap='gray', visible=False)

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


# ==========================================
# MAIN EXECUTION ROUTINE
# ==========================================

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
            
            dyn_search_pattern = os.path.join(patient_root, "*_dyn_*")
            all_dyn_folders = [f for f in glob.glob(dyn_search_pattern) if os.path.isdir(f)]

            pre_folders = [f for f in all_dyn_folders if "_dyn_post" not in os.path.basename(f).lower()]
            post_folders = [f for f in all_dyn_folders if "_dyn_post" in os.path.basename(f).lower()]
            
            if not pre_folders or not post_folders:
                print(f"⚠️ Missing either the initial 'dyn' or 'dyn_post' for Patient {pid}")
                continue
            
            ts_mask_liver = os.path.join(patient_out_dir, "ts_liver_segments.nii.gz")
            ts_mask_aorta = os.path.join(patient_out_dir, "ts_aorta_segment.nii.gz")
            ts_mask_portal = os.path.join(patient_out_dir, "ts_portal_segment.nii.gz")
            
            # Target Co-registration Reference NIfTI Configuration
            ref_nifti = os.path.join(patient_out_dir, f"patient_{pid}_mri.nii.gz")
            
            if not os.path.exists(ref_nifti) or not os.path.exists(ts_mask_aorta):
                print(f"❌ Missing critical reference files for Patient {pid}. Skipping.")
                continue

            # Load spatial mask array details
            m_aorta = sitk.GetArrayFromImage(sitk.ReadImage(ts_mask_aorta))
            mask_aorta_t = np.transpose(np.flip(m_aorta, axis=1), (2, 1, 0))

            # Group temporal acquisitions
            pre_phases = group_dicom_paths_by_time(pre_folders[0])
            post_phases = group_dicom_paths_by_time(post_folders[0])

            # Lock global static timeline baseline coordinate point (t0)
            first_pre_slice = pre_phases[0][0]
            ds_baseline = pydicom.dcmread(first_pre_slice, stop_before_pixels=True)
            global_t0 = get_precise_seconds_from_float(ds_baseline.AcquisitionTime)

            # Read structural physical metadata transformations from target ref_nifti
            ref_geom_sitk = sitk.ReadImage(ref_nifti)
            static_ref_arr = np.transpose(sitk.GetArrayFromImage(ref_geom_sitk), (2, 1, 0))

            # ----------------------------------------------------
            # PROCESSING PHASE 1: PRE-CONTRAST VOLUME
            # ----------------------------------------------------
            print(f"\n📦 Loading & Processing 4D PRE Volume for Patient {pid}...")
            pre_raw_data, pre_tacq = load_and_scale_4d_volume(pre_phases, global_t0)
            
            print("🚀 Executing Model-Driven Motion Correction for PRE Volume...")
            pre_corrected = execute_model_driven_motion_correction(
                data=pre_raw_data, tacq=pre_tacq, mask_aorta_t=mask_aorta_t, 
                baseline_frames=len(pre_tacq), output_diagnostics_path=os.path.join(patient_out_dir, "mdreg_pre_diagnostics")
            )
            
            # Build and save PRE diagnostic registration clip
            generate_and_save_mdreg_animation(pre_corrected, pre_raw_data, patient_out_dir, "PRE", is_pre_phase=True)
            del pre_raw_data  # Clear raw array instantly from workspace
            
            print("🔄 Running ANTs Cross-Series Coregistration to Reference NIfTI...")
            pre_final_xyz_t = register_inter_series_ants(pre_corrected, static_ref_arr)
            del pre_corrected # Clear un-registered dynamic slice cache
            
            pre_nii_path = os.path.join(patient_out_dir, f"patient_{pid}_mri_PRE_Registered_4D.nii.gz")
            save_array_as_4d_nifti(pre_final_xyz_t, ref_geom_sitk, pre_nii_path)
            
            print("📊 Extracting PRE registered timecourse...")
            df_pre = extract_registered_time_course(pre_final_xyz_t, pre_phases, ts_mask_liver, ts_mask_aorta, ts_mask_portal, global_t0)
            df_pre.to_csv(os.path.join(patient_out_dir, f"patient_{pid}_pre_timecourse.csv"), index=False)
            
            del pre_final_xyz_t  # Complete purge of PRE array structures from RAM heap
            gc.collect()         # Force explicit memory cleanup pass

            # ----------------------------------------------------
            # PROCESSING PHASE 2: POST-CONTRAST VOLUME
            # ----------------------------------------------------
            print(f"\n📦 Loading & Processing 4D POST Volume for Patient {pid}...")
            post_raw_data, post_tacq = load_and_scale_4d_volume(post_phases, global_t0)
            
            print("🚀 Executing Model-Driven Motion Correction for POST Volume...")
            post_corrected = execute_model_driven_motion_correction(
                data=post_raw_data, tacq=post_tacq, mask_aorta_t=mask_aorta_t, 
                baseline_frames=0, output_diagnostics_path=os.path.join(patient_out_dir, "mdreg_post_diagnostics")
            )
            
            # Build and save POST diagnostic registration clip
            generate_and_save_mdreg_animation(post_corrected, post_raw_data, patient_out_dir, "POST", is_pre_phase=False)
            del post_raw_data  # Clear raw array instantly from workspace
            
            print("🔄 Running ANTs Cross-Series Coregistration to Reference NIfTI...")
            post_final_xyz_t = register_inter_series_ants(post_corrected, static_ref_arr)
            del post_corrected # Clear un-registered dynamic slice cache
            
            post_nii_path = os.path.join(patient_out_dir, f"patient_{pid}_mri_POST_Registered_4D.nii.gz")
            save_array_as_4d_nifti(post_final_xyz_t, ref_geom_sitk, post_nii_path)
            
            print("📊 Extracting POST registered timecourse...")
            df_post = extract_registered_time_course(post_final_xyz_t, post_phases, ts_mask_liver, ts_mask_aorta, ts_mask_portal, global_t0)
            df_post.to_csv(os.path.join(patient_out_dir, f"patient_{pid}_post_timecourse.csv"), index=False)
            
            del post_final_xyz_t  # Complete purge of POST array structures from RAM heap
            del static_ref_arr    # Clear structural cache reference array
            gc.collect()          # Final cycle clean out before moving to visualization steps

            print(f"✅ Full execution complete for Patient {pid}.\n")
            
            # Open Visualizer window
            visualize_heparim_results(patient_out_dir, pid)

        except Exception as e:
            print(f"❌ Critical failure on Patient {pid}: {e}")
            gc.collect()
            
    sys.stdout = sys.__stdout__


# ==========================================
# GUI INTEGRATION COUPLING
# ==========================================

class RadiologyControlPanel:
    def __init__(self, root, start_callback):
        self.root = root
        self.root.title("HEPARIM - AI Liver Segmentation Control Panel")
        self.root.geometry("600x400")
        self.start_callback = start_callback

        self.base_drive = tk.StringVar(value=r"G:\Shared drives\HEPARIM\Patients 001-029")
        self.output_base = tk.StringVar(value=r"X:\abdominal_imaging\Shared\HEPARIMTS\batch_processing")
        self.patient_list = tk.StringVar(value="001, 004, 005, 008, 010, 013, 019, 020, 022, 027")

        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill="both", expand=True)

        ttk.Label(main_frame, text="Source Data (Base Drive):", font=('Helvetica', 10, 'bold')).pack(anchor="w")
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill="x", pady=(0, 15))
        ttk.Entry(input_frame, textvariable=self.base_drive).pack(side="left", fill="x", expand=True)
        ttk.Button(input_frame, text="Browse", command=self.browse_input).pack(side="right", padx=5)

        ttk.Label(main_frame, text="Save Results To:", font=('Helvetica', 10, 'bold')).pack(anchor="w")
        output_frame = ttk.Frame(main_frame)
        output_frame.pack(fill="x", pady=(0, 15))
        ttk.Entry(output_frame, textvariable=self.output_base).pack(side="left", fill="x", expand=True)
        ttk.Button(output_frame, text="Browse", command=self.browse_output).pack(side="right", padx=5)

        ttk.Label(main_frame, text="Patient IDs (comma separated):", font=('Helvetica', 10, 'bold')).pack(anchor="w")
        ttk.Entry(main_frame, textvariable=self.patient_list).pack(fill="x", pady=(0, 20))

        self.start_btn = ttk.Button(main_frame, text="🚀 Start Processing Pipeline", command=self.run_pipeline)
        self.start_btn.pack(pady=20, ipadx=10, ipady=10)

        self.status = ttk.Label(main_frame, text="Status: Ready", foreground="gray")
        self.status.pack(side="bottom")

    def browse_input(self):
        path = filedialog.askdirectory()
        if path: self.base_drive.set(path)

    def browse_output(self):
        path = filedialog.askdirectory()
        if path: self.output_base.set(path)

    def run_pipeline(self):
        ids = [i.strip() for i in self.patient_list.get().split(",")]
        base = self.base_drive.get()
        out = self.output_base.get()

        if not os.path.exists(base):
            messagebox.showerror("Error", "Source Drive path does not exist!")
            return

        self.start_btn.config(state="disabled")
        self.status.config(text="Status: Processing... check console logs", foreground="blue")
        
        threading.Thread(target=self.worker_thread, args=(ids, base, out), daemon=True).start()

    def worker_thread(self, ids, base, out):
        try:
            self.start_callback(ids, base, out)
            self.root.after(0, self.pipeline_success, len(ids))
        except Exception as e:
            self.root.after(0, self.pipeline_failed, str(e))

    def pipeline_success(self, count):
        self.start_btn.config(state="normal")
        self.status.config(text="Status: Batch Complete", foreground="green")
        messagebox.showinfo("Done", f"Processed {count} patients successfully.")

    def pipeline_failed(self, error_msg):
        self.start_btn.config(state="normal")
        self.status.config(text="Status: Error Occurred", foreground="red")
        messagebox.showerror("Pipeline Failure", f"An error stopped execution:\n{error_msg}")


def run_batch_logic(ids, base, out):
    global PATIENT_IDS, BASE_DRIVE, OUTPUT_BASE
    PATIENT_IDS = ids
    BASE_DRIVE = base
    OUTPUT_BASE = out
    main_pipeline_execution()


if __name__ == "__main__":
    root = tk.Tk()
    app = RadiologyControlPanel(root, run_batch_logic)
    root.mainloop()