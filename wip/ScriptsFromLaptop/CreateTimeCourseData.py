import os
import dicom2nifti
import dicom2nifti.settings as settings
from totalsegmentator.python_api import totalsegmentator
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
from tkinter import filedialog, messagebox, ttk


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

def convert_dicom_to_nifti(dicom_directory, nifti_save_path):
    """
    Converts a directory of DICOM slices into a single NIfTI file.
    Includes settings overrides for Siemens 4D DCE-MRI sequences.
    """
    os.makedirs(os.path.dirname(nifti_save_path), exist_ok=True)
    
    print(f"--- Converting DICOM: {dicom_directory} ---")
    try:
        # --- FIX: Disable strict validation for 4D/Siemens data ---
        settings.disable_validate_slice_increment()
        settings.disable_validate_orientation()
        settings.disable_validate_orthogonal()
        # ----------------------------------------------------------

        # reorient_nifti=True ensures the output is in RAS orientation
        dicom2nifti.dicom_series_to_nifti(dicom_directory, nifti_save_path, reorient_nifti=True)
        print(f"✅ Successfully saved NIfTI to: {nifti_save_path}")
        
        # Re-enable them afterwards (optional, but good practice)
        settings.enable_validate_slice_increment()
        settings.enable_validate_orientation()
        settings.enable_validate_orthogonal()
        
        return True
    
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

def load_4d_dce_dicom(patient_root):
    """
    Finds all phase folders, converts them to arrays, and stacks them into (T, Z, Y, X).
    Assumes folders are named in a way that 'glob' or 'sorted' puts them in temporal order.
    """
    # 1. Find all subfolders containing DICOMs for the DCE sequence
    # Adjust the pattern to match your folder naming (e.g., 'Phase_*' or 'T1_Dynamic_*')
    phase_folders = sorted(glob.glob(os.path.join(patient_root, "DCE_Phases", "*")))
    
    list_of_volumes = []
    
    for folder in phase_folders:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_nifti = os.path.join(tmp_dir, "phase.nii.gz")
            dicom2nifti.dicom_series_to_nifti(folder, temp_nifti)
            img = sitk.ReadImage(temp_nifti)
            list_of_volumes.append(sitk.GetArrayFromImage(img))
    
    # Stack into a 4D numpy array: Shape (Time, Z, Y, X)
    return np.stack(list_of_volumes), img # Return array and the last SITK image for metadata

def extract_time_course_from_4d_niftis(nifti_4d_paths, liver_nifti, aorta_nifti, pv_nifti):
    results = []
    
    # Load masks as numpy arrays (Z, Y, X)
    mask_liver = sitk.GetArrayFromImage(sitk.ReadImage(liver_nifti))
    mask_aorta = sitk.GetArrayFromImage(sitk.ReadImage(aorta_nifti))
    mask_pv = sitk.GetArrayFromImage(sitk.ReadImage(pv_nifti))
    
    global_time_idx = 0  # To keep a continuous count across both files

    for file_idx, nifti_path in enumerate(nifti_4d_paths):
        print(f"   -> Processing 4D NIfTI: {os.path.basename(nifti_path)}")
        
        # Read the image
        img_4d = sitk.ReadImage(nifti_path)
        arr_4d = sitk.GetArrayFromImage(img_4d)
        
        # SimpleITK reads 4D as (Time, Z, Y, X). 
        # If it's 3D, it reads as (Z, Y, X). We force it to 4D for consistency.
        if len(arr_4d.shape) == 3:
            arr_4d = np.expand_dims(arr_4d, axis=0)  # Shape becomes (1, Z, Y, X)
            
        num_timepoints = arr_4d.shape[0]
        
        for t in range(num_timepoints):
            phase_arr = arr_4d[t, ...]  # Extract the 3D volume for this time point
            
            phase_data = {
                "Global_Phase": global_time_idx,
                "Source_Sequence": "dyn_post" if "post" in nifti_path.lower() else "dyn",
                "Local_Timepoint": t
            }
            
            # Vascular Structures
            phase_data["Aorta"] = np.mean(phase_arr[mask_aorta > 0]) if np.any(mask_aorta > 0) else 0
            phase_data["Portal_Vein"] = np.mean(phase_arr[mask_pv > 0]) if np.any(mask_pv > 0) else 0
            
            # Liver Segments S1-S8
            for i in range(1, 9):
                seg_mask = (mask_liver == i)
                phase_data[f"S{i}"] = np.mean(phase_arr[seg_mask]) if np.any(seg_mask) else 0
                
            results.append(phase_data)
            global_time_idx += 1

    return pd.DataFrame(results)


def extract_time_course_from_nifti_masks(dicom_phase_dirs, liver_nifti, aorta_nifti, pv_nifti):
    """
    dicom_phase_dirs: List of paths to folders [Pre, Phase1, Phase2, ...]
    Returns: Pandas DataFrame with Mean Intensities
    """
    
    print(dicom_phase_dirs)
    results = []

    # 1. Load the masks from NIfTI
    # SimpleITK handles the orientation (RAS/LPS) automatically
    mask_liver = sitk.GetArrayFromImage(sitk.ReadImage(liver_nifti))
    mask_aorta = sitk.GetArrayFromImage(sitk.ReadImage(aorta_nifti))
    mask_pv = sitk.GetArrayFromImage(sitk.ReadImage(pv_nifti))

    # 2. Iterate through each DICOM phase folder
    for phase_idx, folder in enumerate(dicom_phase_dirs):
        print(f"Processing Phase {phase_idx}...")
        
        # Load the DICOM phase as a 3D volume
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(folder)
        reader.SetFileNames(dicom_names)
        phase_img = reader.Execute()
        
        # Convert to numpy for math
        phase_arr = sitk.GetArrayFromImage(phase_img)
        
        # 3. Calculate means for this specific timepoint
        phase_data = {"Phase": phase_idx}
        
        # Aorta and Portal Vein (Binary)
        phase_data["Aorta"] = np.mean(phase_arr[mask_aorta > 0])
        phase_data["PortalVein"] = np.mean(phase_arr[mask_pv > 0])
        
        # Liver Segments (Labels 1-8)
        for seg_id in range(1, 9):
            seg_mask = (mask_liver == seg_id)
            if np.any(seg_mask):
                phase_data[f"S{seg_id}"] = np.mean(phase_arr[seg_mask])
            else:
                phase_data[f"S{seg_id}"] = np.nan
        
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
            
            # Keep them in chronological order
            target_dicom_phases = [pre_folders[0], post_folders[0]]
            
            # 3. CONVERT 4D DICOMS TO 4D NIFTIS
            nifti_4d_paths = []
            for dicom_folder in target_dicom_phases:
                # Create a name based on whether it's the pre or post sequence
                seq_name = "dyn_post" if "post" in dicom_folder.lower() else "dyn"
                out_nifti = os.path.join(patient_out_dir, f"4d_sequence_{seq_name}.nii.gz")
                
                # Use your existing wrapper function
                print(f"🔄 Converting 4D DICOM to NIfTI: {seq_name}")
                convert_dicom_to_nifti(dicom_folder, out_nifti)
                nifti_4d_paths.append(out_nifti)
            
            # 4. RUN EXTRACTION ACROSS TIME DIMENSIONS
            ts_mask_liver = os.path.join(patient_out_dir, "ts_liver_segments.nii.gz")
            ts_mask_aorta = os.path.join(patient_out_dir, "ts_aorta_segment.nii.gz")
            ts_mask_portal = os.path.join(patient_out_dir, "ts_portal_segment.nii.gz")
            print(f"📈 Extracting 4D Time Courses for Patient {pid}...")
            df_results = extract_time_course_from_4d_niftis(
                 nifti_4d_paths=nifti_4d_paths,
                 liver_nifti=ts_mask_liver,
                 aorta_nifti=ts_mask_aorta,
                 pv_nifti=ts_mask_portal
                )
                
            # 5. SAVE RESULTS
            csv_path = os.path.join(patient_out_dir, f"patient_{pid}_timecourse.csv")
            df_results.to_csv(csv_path, index=False)
            print(f"✅ CSV saved with {len(df_results)} total time points.")
        except Exception as e:
            print(f"❌ Critical failure on Patient {pid}: {e}")
            
            
def run_batch_logic(ids, base, out):
    """ This function bridges the GUI to your existing processing loop """
    # Map the GUI variables to the global scope of your script
    global PATIENT_IDS, BASE_DRIVE, OUTPUT_BASE
    PATIENT_IDS = ids
    BASE_DRIVE = base
    OUTPUT_BASE = out
    
    # Trigger the processing logic you wrote previously
    # (The code inside your 'if __name__ == "__main__":' block)
    main_pipeline_execution() 

if __name__ == "__main__":
    root = tk.Tk()
    app = RadiologyControlPanel(root, run_batch_logic)
    root.mainloop()