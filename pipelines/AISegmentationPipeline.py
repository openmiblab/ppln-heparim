import os
import sys
import glob
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import SimpleITK as sitk
import numpy as np
import napari

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.segmentation import (
    convert_dicom_to_nifti, run_total_segmentator_task, create_multi_organ_mosaic, convert_mitk_suffix_mapping
)

class AISegmentationPipeline:
    def __init__(self, root):
        self.root = root
        self.root.title("HEPARIM - AI Segmentation Infrastructure")
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
        
        ttk.Label(main_frame, text="Base Drive Location:", font=('Helvetica', 10, 'bold')).pack(anchor="w")
        f1 = ttk.Frame(main_frame); f1.pack(fill="x", pady=(0, 10))
        ttk.Entry(f1, textvariable=self.base_drive).pack(side="left", fill="x", expand=True)
        ttk.Button(f1, text="Browse", command=lambda: self.base_drive.set(filedialog.askdirectory())).pack(side="right", padx=5)

        ttk.Label(main_frame, text="Output Directory Target:", font=('Helvetica', 10, 'bold')).pack(anchor="w")
        f2 = ttk.Frame(main_frame); f2.pack(fill="x", pady=(0, 10))
        ttk.Entry(f2, textvariable=self.output_base).pack(side="left", fill="x", expand=True)
        ttk.Button(f2, text="Browse", command=lambda: self.output_base.set(filedialog.askdirectory())).pack(side="right", padx=5)

        ttk.Label(main_frame, text="Patient Mapping Sequence:", font=('Helvetica', 10, 'bold')).pack(anchor="w")
        ttk.Entry(main_frame, textvariable=self.patient_list).pack(fill="x", pady=(0, 20))
        
        ttk.Button(main_frame, text="🚀 Run TotalSegmentator Batch", command=self.execute_batch).pack(pady=10)

    def execute_batch(self):
        ids = [i.strip() for i in self.patient_list.get().split(",")]
        base = self.base_drive.get()
        out = self.output_base.get()
        
        for pid in ids:
            p_folders = glob.glob(os.path.join(base, f"Patient {pid}//Patient {pid}*"))
            if not p_folders: continue
            p_root = p_folders[0]
            
            dicom_folders = [f for f in glob.glob(os.path.join(p_root, "**", "*Dicom.seg*"), recursive=True) if os.path.isdir(f)]
            mitk_files = glob.glob(os.path.join(p_root, "**", f"Patient {pid}.mitk"), recursive=True)
            if not dicom_folders or not mitk_files: continue
            
            p_out = os.path.join(out, f"Patient_{pid}")
            os.makedirs(p_out, exist_ok=True)
            
            ref_nifti = os.path.join(p_out, f"patient_{pid}_mri.nii.gz")
            ts_liver = os.path.join(p_out, "ts_liver_segments.nii.gz")
            ts_aorta = os.path.join(p_out, "ts_aorta_segment.nii.gz")
            ts_portal = os.path.join(p_out, "ts_portal_segment.nii.gz")
            manual_mask = os.path.join(p_out, "manual_liver_segments.nii.gz")

            if convert_dicom_to_nifti(dicom_folders[0], ref_nifti):
                run_total_segmentator_task(ref_nifti, ts_liver, task="liver_segments_mr")
                run_total_segmentator_task(ref_nifti, ts_aorta, task="total_mr", roi_subset=['aorta'])
                run_total_segmentator_task(ref_nifti, ts_portal, task="total_mr", roi_subset=['portal_vein_and_splenic_vein'])
                convert_mitk_suffix_mapping(mitk_files[0], ref_nifti, manual_mask)
                create_multi_organ_mosaic(ref_nifti, ts_liver, ts_aorta, ts_portal, os.path.join(p_out, f"mosaic_{pid}.png"))
                
        messagebox.showinfo("Success", "AI Image Segmentation extraction pipelines complete.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AISegmentationPipeline(root)
    root.mainloop()