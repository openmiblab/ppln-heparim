from pickle import TRUE
import pydicom
import sys
from collections import defaultdict
import gc
import os
import time
import threading
import concurrent.futures  # 🚀 Added for true multi-core parallel processing
import SimpleITK as sitk
import numpy as np
import pandas as pd
import glob
import napari
import zarr  
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import mdreg
import mdreg.ants as mds

import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

# ==========================================
# I/O AND UTILITY METHODS
# ==========================================

class Tee:
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.terminal = sys.stdout
  
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.terminal
        self.file.close()


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

def load_and_scale_4d_volume(phase_groups, global_t0, zarr_store_path):
    reader = sitk.ImageSeriesReader()
    tacq = []
    
    first_phase = phase_groups[0]
    reader.SetFileNames(first_phase)
    sample_vol = sitk.GetArrayFromImage(reader.Execute())
    
    z_dim, y_dim, x_dim = sample_vol.shape
    t_dim = len(phase_groups)
    
    store = zarr.DirectoryStore(zarr_store_path)
    z_data = zarr.create(
        shape=(x_dim, y_dim, z_dim, t_dim), 
        chunks=(x_dim, y_dim, 1, 1), 
        dtype='float32', 
        store=store, 
        overwrite=True
    )
    
    for t_idx, phase in enumerate(phase_groups):
        ds_slice = pydicom.dcmread(phase[0], stop_before_pixels=True)
        tacq.append(get_precise_seconds_from_float(ds_slice.AcquisitionTime) - global_t0)
        
        reader.SetFileNames(phase)
        vol_arr = sitk.GetArrayFromImage(reader.Execute()).astype(np.float32)
        slope = float(ds_slice.get("RescaleSlope", 1))
        intercept = float(ds_slice.get("RescaleIntercept", 0))
        
        scaled_frame = vol_arr * slope + intercept
        z_data[..., t_idx] = np.transpose(scaled_frame, (2, 1, 0))
        
    return z_data, np.array(tacq)


def execute_model_driven_motion_correction(z_data, tacq, mask_aorta_t, baseline_frames, zarr_out_path):
    aif = np.array([
        np.mean(z_data[..., t][mask_aorta_t > 0]) if np.any(mask_aorta_t > 0) else 1e-6 
        for t in range(z_data.shape[-1])
    ])
    
    t_start = time.time()
    
    # 🎯 Writes outputs directly into the Zarr path, avoiding unnecessary returns
    _ = mdreg.fit(
        z_data[:], 
        fit_image={
            'func': mdreg.fit_2cm_lin, 
            'time': tacq, 
            'aif': aif, 
            'baseline': baseline_frames
        },
        parallel=TRUE,
        maxit=3,
        path=zarr_out_path, 
        verbose=2,
    )
    
    print(f"Computation time: {round(time.time() - t_start)} seconds.")
    return zarr.open(zarr_out_path, mode='r')


def generate_and_save_mdreg_animation(z_coreg_data, z_raw_input_data, destination_path, title_prefix, is_pre_phase=True):
    print(f"🎬 Creating motion correction diagnostic animation for {title_prefix}...")
    num_frames = z_coreg_data.shape[-1]
    t0, t1 = (0, num_frames) if is_pre_phase else (0, min(15, num_frames))
        
    center_z = z_coreg_data.shape[2] // 2
    coreg_sliced = z_coreg_data[:, :, center_z, t0:t1]
    vmax_val = 0.9 * np.max(z_raw_input_data[:, :, center_z, 0])
    
    try:
        fig, ax = plt.subplots()
        anim = mdreg.plot.animation(
            coreg_sliced,
            title=f'{title_prefix} DCE motion corrected',
            vmin=0,
            vmax=vmax_val,
        )
        output_file = os.path.join(destination_path, f"{title_prefix.lower()}_motion_correction.gif")
        anim.save(output_file, writer='pillow', fps=4)
        print(f"💾 Diagnostic animation written successfully to: {output_file}")
        
        plt.close(fig)
        plt.close('all')
        del anim
    except Exception as animation_error:
        print(f"⚠️ Animation generation skipped for {title_prefix}: {animation_error}")
        plt.close('all')


def register_inter_series_ants(z_moving_4d, static_ref_arr, zarr_out_path):
    print("🔄 Calculating ANTs rigid alignment field to baseline reference NIfTI...")
    moving_anchor_3d = z_moving_4d[..., 0].astype(np.float32)
    static_target_3d = static_ref_arr.astype(np.float32)
    
    _, deformation_field = mds.coreg(
        static_target_3d, 
        moving_anchor_3d,
        return_transfo=True, 
        attachment=15, 
        type='rigid'
    )
    
    num_phases = z_moving_4d.shape[-1]
    store = zarr.DirectoryStore(zarr_out_path)
    z_registered = zarr.create(shape=z_moving_4d.shape, chunks=(z_moving_4d.shape[0], z_moving_4d.shape[1], 1, 1), dtype='float32', store=store, overwrite=True)
    
    for t in range(num_phases):
        vol_3d = z_moving_4d[..., t]
        warped_vol = mds.transform(vol_3d, deformation_field, interpolator='linear')
        z_registered[..., t] = warped_vol
        
    return z_registered


def save_zarr_as_4d_nifti(z_data_xyz_t, ref_geom_sitk, destination_path):
    data_t_z_y_x = np.transpose(z_data_xyz_t[:], (3, 2, 1, 0))
    sitk_image = sitk.GetImageFromArray(data_t_z_y_x.astype(np.float32))
    sitk_image.SetSpacing((*ref_geom_sitk.GetSpacing(), 1.0))
    sitk_image.SetOrigin((*ref_geom_sitk.GetOrigin(), 0.0))
    sitk_image.SetDirection(ref_geom_sitk.GetDirection())
    sitk.WriteImage(sitk_image, destination_path)
    print(f"💾 NIfTI file cleanly saved: {destination_path}")


def extract_registered_time_course(z_corrected_xyz_t, phase_paths, liver_nifti, aorta_nifti, pv_nifti, global_t0):
    results = []

    mask_liver = sitk.GetArrayFromImage(sitk.ReadImage(liver_nifti))
    mask_aorta = sitk.GetArrayFromImage(sitk.ReadImage(aorta_nifti))
    mask_pv = sitk.GetArrayFromImage(sitk.ReadImage(pv_nifti))
    
    valid_segments = [1, 2, 3, 4, 5, 6, 8]
    seg_volumes = {i: np.sum(mask_liver == i) for i in valid_segments}
    total_liver_volume = sum(seg_volumes.values())

    num_phases = z_corrected_xyz_t.shape[-1]

    for phase_idx in range(num_phases):
        current_path = phase_paths[phase_idx][0]
        ds = pydicom.dcmread(current_path, stop_before_pixels=True)
        
        current_abs_time = get_precise_seconds_from_float(ds.AcquisitionTime)
        relative_time = current_abs_time - global_t0
        
        slope = float(ds.get("RescaleSlope", 1))
        intercept = float(ds.get("RescaleIntercept", 0))
        
        time_frame_xyz = z_corrected_xyz_t[..., phase_idx]
        scaled_arr = np.transpose(time_frame_xyz, (2, 1, 0)).astype(np.float64) * slope + intercept
        
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


# ==========================================
# PARALLELIZED EXECUTOR TARGET
# ==========================================

def process_single_patient(pid, base_drive, output_base):
    """
    Independent wrapper execution block targeting sub-process spawning.
    Returns status codes or error details back to the orchestrator thread.
    """
    logger_inst = None
    try:
        patient_search = os.path.join(base_drive, f"Patient {pid}*")
        patient_folders = glob.glob(patient_search)
        
        if not patient_folders:
            return pid, "FAILED: Patient root folder not found."
        
        patient_root = patient_folders[0]
        patient_out_dir = os.path.join(output_base, f"Patient_{pid}")
        os.makedirs(patient_out_dir, exist_ok=True)
        
        # Divert log outputs to dedicated per-patient files to avoid process cross-talk
        logger_inst = Tee(os.path.join(patient_out_dir, f"pipeline_execution_{pid}.log"))
        sys.stdout = logger_inst

        dyn_search_pattern = os.path.join(patient_root, "*_dyn_*")
        all_dyn_folders = [f for f in glob.glob(dyn_search_pattern) if os.path.isdir(f)]

        pre_folders = [f for f in all_dyn_folders if "_dyn_post" not in os.path.basename(f).lower()]
        post_folders = [f for f in all_dyn_folders if "_dyn_post" in os.path.basename(f).lower()]
        
        if not pre_folders or not post_folders:
            return pid, "FAILED: Missing either the initial 'dyn' or 'dyn_post' series."
        
        ts_mask_liver = os.path.join(patient_out_dir, "ts_liver_segments.nii.gz")
        ts_mask_aorta = os.path.join(patient_out_dir, "ts_aorta_segment.nii.gz")
        ts_mask_portal = os.path.join(patient_out_dir, "ts_portal_segment.nii.gz")
        ref_nifti = os.path.join(patient_out_dir, f"patient_{pid}_mri.nii.gz")
        
        if not os.path.exists(ref_nifti) or not os.path.exists(ts_mask_aorta):
            return pid, "FAILED: Missing structural references or segmentations."

        m_aorta = sitk.GetArrayFromImage(sitk.ReadImage(ts_mask_aorta))
        mask_aorta_t = np.transpose(m_aorta, (2, 1, 0))

        pre_phases = group_dicom_paths_by_time(pre_folders[0])
        post_phases = group_dicom_paths_by_time(post_folders[0])

        first_pre_slice = pre_phases[0][0]
        ds_baseline = pydicom.dcmread(first_pre_slice, stop_before_pixels=True)
        global_t0 = get_precise_seconds_from_float(ds_baseline.AcquisitionTime)

        ref_geom_sitk = sitk.ReadImage(ref_nifti)
        static_ref_arr = np.transpose(sitk.GetArrayFromImage(ref_geom_sitk), (2, 1, 0))

        # Localized Zarr cache zones per process bounds
        zarr_dir_pre_raw = os.path.join(patient_out_dir, "pre_raw.zarr")
        zarr_dir_pre_cor = os.path.join(patient_out_dir, "pre_corrected.zarr")
        zarr_dir_pre_reg = os.path.join(patient_out_dir, "pre_registered.zarr")
        
        zarr_dir_post_raw = os.path.join(patient_out_dir, "post_raw.zarr")
        zarr_dir_post_cor = os.path.join(patient_out_dir, "post_corrected.zarr")
        zarr_dir_post_reg = os.path.join(patient_out_dir, "post_registered.zarr")

        # --- PRE PROCESSING ---
        print(f"\n📦 Loading 4D PRE Zarr Volume for Patient {pid}...")
        z_pre_raw, pre_tacq = load_and_scale_4d_volume(pre_phases, global_t0, zarr_dir_pre_raw)
        
        print(f"🚀 Model-Driven Motion Correction for PRE [{pid}]...")
        z_pre_corrected = execute_model_driven_motion_correction(
            z_data=z_pre_raw, tacq=pre_tacq, mask_aorta_t=mask_aorta_t, 
            baseline_frames=len(pre_tacq), zarr_out_path=zarr_dir_pre_cor
        )
        generate_and_save_mdreg_animation(z_pre_corrected, z_pre_raw, patient_out_dir, "PRE", is_pre_phase=True)
        
        print(f"🔄 ANTs Coregistration for PRE [{pid}]...")
        z_pre_final = register_inter_series_ants(z_pre_corrected, static_ref_arr, zarr_dir_pre_reg)
        save_zarr_as_4d_nifti(z_pre_final, ref_geom_sitk, os.path.join(patient_out_dir, f"patient_{pid}_mri_PRE_Registered_4D.nii.gz"))
        
        print(f"📊 Extrapolating PRE Curves [{pid}]...")
        df_pre = extract_registered_time_course(z_pre_final, pre_phases, ts_mask_liver, ts_mask_aorta, ts_mask_portal, global_t0)
        df_pre.to_csv(os.path.join(patient_out_dir, f"patient_{pid}_pre_timecourse.csv"), index=False)
        
        gc.collect()

        # --- POST PROCESSING ---
        print(f"\n📦 Loading 4D POST Zarr Volume for Patient {pid}...")
        z_post_raw, post_tacq = load_and_scale_4d_volume(post_phases, global_t0, zarr_dir_post_raw)
        
        print(f"🚀 Model-Driven Motion Correction for POST [{pid}]...")
        z_post_corrected = execute_model_driven_motion_correction(
            z_data=z_post_raw, tacq=post_tacq, mask_aorta_t=mask_aorta_t, 
            baseline_frames=0, zarr_out_path=zarr_dir_post_cor
        )
        generate_and_save_mdreg_animation(z_post_corrected, z_post_raw, patient_out_dir, "POST", is_pre_phase=False)
        
        print(f"🔄 ANTs Coregistration for POST [{pid}]...")
        z_post_final = register_inter_series_ants(z_post_corrected, static_ref_arr, zarr_dir_post_reg)
        save_zarr_as_4d_nifti(z_post_final, ref_geom_sitk, os.path.join(patient_out_dir, f"patient_{pid}_mri_POST_Registered_4D.nii.gz"))
        
        print(f"📊 Extrapolating POST Curves [{pid}]...")
        df_post = extract_registered_time_course(z_post_final, post_folders, ts_mask_liver, ts_mask_aorta, ts_mask_portal, global_t0)
        df_post.to_csv(os.path.join(patient_out_dir, f"patient_{pid}_post_timecourse.csv"), index=False)
        
        if logger_inst:
            logger_inst.close()
        return pid, "SUCCESS"

    except Exception as process_error:
        if logger_inst:
            logger_inst.close()
        return pid, f"CRITICAL FAILURE: {str(process_error)}"


# ==========================================
# GUI INTEGRATION COUPLING
# ==========================================

class RadiologyControlPanel:
    def __init__(self, root):
        self.root = root
        self.root.title("HEPARIM - Parallel Processing Engine Panel")
        self.root.geometry("600x420")

        self.base_drive = tk.StringVar(value=r"G:\Shared drives\HEPARIM\Patients 001-029")
        self.output_base = tk.StringVar(value=r"X:\abdominal_imaging\Shared\HEPARIMTS\batch_processing")
        self.patient_list = tk.StringVar(value="001, 004, 005")

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

        self.start_btn = ttk.Button(main_frame, text="🚀 Launch Parallel Workers", command=self.run_parallel_pipeline)
        self.start_btn.pack(pady=10, ipadx=10, ipady=10)

        self.status = ttk.Label(main_frame, text="Status: Engine Idle", foreground="gray")
        self.status.pack(side="bottom")

    def browse_input(self):
        path = filedialog.askdirectory()
        if path: self.base_drive.set(path)

    def browse_output(self):
        path = filedialog.askdirectory()
        if path: self.output_base.set(path)

    def run_parallel_pipeline(self):
        ids = [i.strip() for i in self.patient_list.get().split(",")]
        base = self.base_drive.get()
        out = self.output_base.get()

        if not os.path.exists(base):
            messagebox.showerror("Error", "Source Drive path does not exist!")
            return

        self.start_btn.config(state="disabled")
        self.status.config(text="Status: Multi-core processing running...", foreground="blue")
        
        # Fire off a non-blocking background orchestration thread to keep Tkinter responsive
        threading.Thread(target=self.orchestrator_thread, args=(ids, base, out), daemon=True).start()

    def orchestrator_thread(self, ids, base, out):
        results_summary = []
        
        # 🚀 Use half your system's available CPU threads to balance execution vs I/O bandwidth bounds
        max_workers = max(1, os.cpu_count() // 2) 
        print(f"⚙️ Spawning Process Pool Executor with {max_workers} worker processes...")

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Schedule each patient execution task inside the independent sub-process structures
            futures = {executor.submit(process_single_patient, pid, base, out): pid for pid in ids}
            
            for future in concurrent.futures.as_completed(futures):
                pid = futures[future]
                try:
                    pid, status_msg = future.result()
                    results_summary.append(f"Patient {pid}: {status_msg}")
                    print(f"📊 Worker Complete -> Patient {pid}: {status_msg}")
                except Exception as exc:
                    results_summary.append(f"Patient {pid}: Process Crashed -> {exc}")
                    print(f"❌ Process Level Exception occurred on Patient {pid}: {exc}")

        self.root.after(0, self.pipeline_complete_ui, results_summary)

    def pipeline_complete_ui(self, summaries):
        self.start_btn.config(state="normal")
        self.status.config(text="Status: Batch Run Finished", foreground="green")
        
        report = "\n".join(summaries)
        messagebox.showinfo("Batch Execution Run Report", f"Processing Complete:\n\n{report}")
        
        # Now launch Napari for visual review on patients that finished cleanly
        for summary in summaries:
            if "SUCCESS" in summary:
                target_pid = summary.split(":")[0].replace("Patient ", "").strip()
                target_dir = os.path.join(self.output_base.get(), f"Patient_{target_pid}")
                try:
                    # Napari requires the main loop thread; called back to screen safely
                    print(f"📺 Rendering confirmation UI layout map for Patient {target_pid}...")
                    
                    # Read native arrays directly without index flips
                    pre_path = os.path.join(target_dir, f"patient_{target_pid}_mri_PRE_Registered_4D.nii.gz")
                    liver_nifti = os.path.join(target_dir, "ts_liver_segments.nii.gz")
                    
                    viewer = napari.Viewer(title=f"Parallel Run Quality Assurance Review - ID {target_pid}")
                    if os.path.exists(pre_path):
                        viewer.add_image(sitk.GetArrayFromImage(sitk.ReadImage(pre_path)), name='MRI_PRE')
                    if os.path.exists(liver_nifti):
                        viewer.add_labels(sitk.GetArrayFromImage(sitk.ReadImage(liver_nifti)).astype(np.uint8), name='Liver')
                    napari.run()
                except Exception as render_err:
                    print(f"Review pane window closed or skipped: {render_err}")


if __name__ == "__main__":
    # ⚠️ CRITICAL MANDATE FOR WINDOWS MULTIPROCESSING SAFETY
    # This prevents spawned child processes from infinitely re-spawning nested GUI instances.
    multiprocessing_type = concurrent.futures.process
    root = tk.Tk()
    app = RadiologyControlPanel(root)
    root.mainloop()