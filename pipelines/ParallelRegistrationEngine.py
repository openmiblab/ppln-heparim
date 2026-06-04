import os
import sys
import glob
import json
import logging
import concurrent.futures
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import SimpleITK as sitk
import numpy as np
import napari

# Local Module Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.registration import (
    group_dicom_paths_by_time, get_precise_seconds_from_float, load_and_scale_4d_volume,
    execute_model_driven_motion_correction, register_inter_series_ants, save_zarr_as_4d_nifti,
    extract_registered_time_course
)

def process_single_patient(pid, base_drive, output_base):
    """Decoupled processing worker engine target[cite: 3]."""
    try:
        patient_folders = glob.glob(os.path.join(base_drive, f"Patient {pid}*"))
        if not patient_folders: return pid, "FAILED: Folder missing"
        patient_root = patient_folders[0]
        patient_out_dir = os.path.join(output_base, f"Patient_{pid}")
        os.makedirs(patient_out_dir, exist_ok=True)
        
        # Logging Setup
        logger = logging.getLogger(f"Worker_{pid}")
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(patient_out_dir, f"registration_{pid}.log"), mode='w')
        logger.addHandler(fh)

        all_dyn = [f for f in glob.glob(os.path.join(patient_root, "*_dyn_*")) if os.path.isdir(f)]
        pre_folders = [f for f in all_dyn if "_dyn_post" not in os.path.basename(f).lower()]
        post_folders = [f for f in all_dyn if "_dyn_post" in os.path.basename(f).lower()]
        
        if not pre_folders or not post_folders: return pid, "FAILED: Layout incomplete"
        
        ref_nifti = os.path.join(patient_out_dir, f"patient_{pid}_mri.nii.gz")
        ts_mask_aorta = os.path.join(patient_out_dir, "ts_aorta_segment.nii.gz")
        ts_mask_liver = os.path.join(patient_out_dir, "ts_liver_segments.nii.gz")
        ts_mask_portal = os.path.join(patient_out_dir, "ts_portal_segment.nii.gz")
        
        mask_aorta_t = sitk.GetArrayFromImage(sitk.ReadImage(ts_mask_aorta)).astype(np.uint8) if os.path.exists(ts_mask_aorta) else None
        
        pre_phases = group_dicom_paths_by_time(pre_folders[0])
        post_phases = group_dicom_paths_by_time(post_folders[0])
        
        ds_baseline = pydicom.dcmread(pre_phases[0][0], stop_before_pixels=True)
        global_t0 = get_precise_seconds_from_float(getattr(ds_baseline, 'AcquisitionTime', "000000.00"))
        
        ref_geom_sitk = sitk.ReadImage(ref_nifti)
        static_ref_arr = np.transpose(sitk.GetArrayFromImage(ref_geom_sitk), (2, 1, 0))

        # Core optimization execution runs
        z_pre_raw, pre_tacq = load_and_scale_4d_volume(pre_phases, global_t0, os.path.join(patient_out_dir, "pre_raw.zarr"))
        z_pre_cor = execute_model_driven_motion_correction(z_pre_raw, pre_tacq, mask_aorta_t, len(pre_tacq), os.path.join(patient_out_dir, "pre_cor.zarr"), logger)
        z_pre_fin = register_inter_series_ants(z_pre_cor, static_ref_arr, os.path.join(patient_out_dir, "pre_reg.zarr"))
        save_zarr_as_4d_nifti(z_pre_fin, ref_geom_sitk, os.path.join(patient_out_dir, f"patient_{pid}_mri_PRE_Registered_4D.nii.gz"))
        
        df_pre = extract_registered_time_course(z_pre_fin, pre_phases, ts_mask_liver, ts_mask_aorta, ts_mask_portal, global_t0)
        df_pre.to_csv(os.path.join(patient_out_dir, f"patient_{pid}_pre_timecourse.csv"), index=False)
        
        return pid, "SUCCESS"
    except Exception as e:
        return pid, f"CRITICAL FAILURE: {str(e)}"

class ParallelRegistrationEnginePipeline:
    def __init__(self, root):
        self.root = root
        self.root.title("HEPARIM - Parallel Registration Engine")
        self.root.geometry("600x400")
        
        with open(os.path.join(os.path.dirname(__file__), '../config.json')) as f:
            self.config = json.load(f)
            
        self.base_drive = tk.StringVar(value=self.config["DEFAULT_BASE_DRIVE"])
        self.output_base = tk.StringVar(value=self.config["DEFAULT_OUTPUT_BASE"])
        self.patient_list = tk.StringVar(value=self.config["DEFAULT_PATIENT_LIST"])
        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill="both", expand=True)
        
        ttk.Label(main_frame, text="Source Data Directory:", font=('Helvetica', 10, 'bold')).pack(anchor="w")
        f1 = ttk.Frame(main_frame); f1.pack(fill="x", pady=(0, 10))
        ttk.Entry(f1, textvariable=self.base_drive).pack(side="left", fill="x", expand=True)
        ttk.Button(f1, text="Browse", command=lambda: self.base_drive.set(filedialog.askdirectory())).pack(side="right", padx=5)
        
        ttk.Label(main_frame, text="Save Results Location:", font=('Helvetica', 10, 'bold')).pack(anchor="w")
        f2 = ttk.Frame(main_frame); f2.pack(fill="x", pady=(0, 10))
        ttk.Entry(f2, textvariable=self.output_base).pack(side="left", fill="x", expand=True)
        ttk.Button(f2, text="Browse", command=lambda: self.output_base.set(filedialog.askdirectory())).pack(side="right", padx=5)
        
        ttk.Label(main_frame, text="Target Patient IDs:", font=('Helvetica', 10, 'bold')).pack(anchor="w")
        ttk.Entry(main_frame, textvariable=self.patient_list).pack(fill="x", pady=(0, 20))
        
        self.start_btn = ttk.Button(main_frame, text="🚀 Launch Parallel Workers", command=self.run_pipeline)
        self.start_btn.pack(pady=10, ipadx=10, ipady=10)

    def run_pipeline(self):
        ids = [i.strip() for i in self.patient_list.get().split(",")]
        base = self.base_drive.get()
        out = self.output_base.get()
        
        max_workers = max(1, os.cpu_count() // 2)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_patient, pid, base, out): pid for pid in ids}
            for future in concurrent.futures.as_completed(futures):
                pid, msg = future.result()
                print(f"Tracking ID {pid}: {msg}")
        messagebox.showinfo("Complete", "Parallel registration loops terminated execution safely.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ParallelRegistrationEnginePipeline(root)
    root.mainloop()