import os
import sys
import glob
import gc
import json
import pydicom
import SimpleITK as sitk
import numpy as np
import pandas as pd
from collections import defaultdict
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tqdm import tqdm

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

class RadiologyControlPanel:
    def __init__(self, root, start_callback):
        self.root = root
        self.root.title("HEPARIM - Raw Signal Timecourse Extractor")
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

        # Input Section
        ttk.Label(main_frame, text="Source Data (Base Drive):", font=('Helvetica', 10, 'bold')).pack(anchor="w")
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill="x", pady=(0, 15))
        ttk.Entry(input_frame, textvariable=self.base_drive).pack(side="left", fill="x", expand=True)
        ttk.Button(input_frame, text="Browse", command=lambda: self.base_drive.set(filedialog.askdirectory())).pack(side="right", padx=5)

        # Output Section
        ttk.Label(main_frame, text="Save Results To:", font=('Helvetica', 10, 'bold')).pack(anchor="w")
        output_frame = ttk.Frame(main_frame)
        output_frame.pack(fill="x", pady=(0, 15))
        ttk.Entry(output_frame, textvariable=self.output_base).pack(side="left", fill="x", expand=True)
        ttk.Button(output_frame, text="Browse", command=lambda: self.output_base.set(filedialog.askdirectory())).pack(side="right", padx=5)

        # Patient IDs
        ttk.Label(main_frame, text="Patient IDs (comma separated):", font=('Helvetica', 10, 'bold')).pack(anchor="w")
        ttk.Entry(main_frame, textvariable=self.patient_list).pack(fill="x", pady=(0, 20))

        # Start Button
        self.start_btn = ttk.Button(main_frame, text="🚀 Extract Raw Signal Timecourse", command=self.run_pipeline)
        self.start_btn.pack(pady=20, ipadx=10, ipady=10)

        # Status Footer
        self.status = ttk.Label(main_frame, text="Status: Ready", foreground="gray")
        self.status.pack(side="bottom")

    def run_pipeline(self):
        ids = [i.strip() for i in self.patient_list.get().split(",") if i.strip()]
        base = self.base_drive.get()
        out = self.output_base.get()

        if not os.path.exists(base):
            messagebox.showerror("Error", "Source Drive path does not exist!")
            return

        self.start_btn.config(state="disabled")
        self.status.config(text="Status: Processing... check console for logs", foreground="blue")
        
        self.root.update()
        self.start_callback(ids, base, out)
        
        self.start_btn.config(state="normal")
        self.status.config(text="Status: Batch Complete", foreground="green")
        messagebox.showinfo("Done", f"Extracted raw timecourses for {len(ids)} patients successfully.")


def get_best_time(ds):
    """Finds the relative time in seconds for DCE fitting from DICOM metadata."""
    trigger_time = ds.get('TriggerTime')
    if trigger_time is not None:
        return float(trigger_time) / 1000.0

    acq_time = ds.get('AcquisitionTime')
    if acq_time:
        t_str = str(acq_time)
        try:
            hms = t_str.split('.')[0]
            ms = t_str.split('.')[1] if '.' in t_str else "0"
            hh, mm, ss = int(hms[:2]), int(hms[2:4]), int(hms[4:6])
            return (hh * 3600) + (mm * 60) + ss + float("0." + ms)
        except Exception:
            return 0.0
            
    return 0.0

def get_precise_seconds_from_float(ds_or_val):
    """Converts DICOM AcquisitionTime (HHMMSS.ffffff) to absolute seconds from midnight."""
    val = ds_or_val.AcquisitionTime if hasattr(ds_or_val, 'AcquisitionTime') else ds_or_val
    t_str = f"{float(val):010.3f}"
    hms, ms = t_str.split('.')
    hh, mm, ss = int(hms[:2]), int(hms[2:4]), int(hms[4:6])
    return (hh * 3600) + (mm * 60) + ss + float("0." + ms)

def group_dicom_paths_by_time(dicom_dir):
    """Groups DICOM files by rounded AcquisitionTime and sorts slices spatially."""
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
            except Exception:
                z_pos = float(ds.get('InstanceNumber', 0))
                
            time_groups[t_key].append((z_pos, f))
        except Exception:
            continue
            
    sorted_times = sorted(time_groups.keys())
    grouped_filepaths = []
    for t in sorted_times:
        sorted_by_z = sorted(time_groups[t], key=lambda x: x[0])
        just_paths = [x[1] for x in sorted_by_z]
        grouped_filepaths.append(just_paths)
        
    print(f"✅ Found {len(grouped_filepaths)} time points.")
    return grouped_filepaths

def stream_dicom_series_to_numpy(phase_paths):
    """Loads a list of lists containing sorted 2D slice paths directly into a 4D NumPy matrix."""
    num_phases = len(phase_paths)
    if num_phases == 0:
        return None
    
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(phase_paths[0])
    sample_img = reader.Execute()
    z, y, x = sitk.GetArrayFromImage(sample_img).shape
    
    array_4d = np.zeros((num_phases, z, y, x), dtype=np.float32)
    
    for idx, paths in enumerate(phase_paths):
        reader.SetFileNames(paths)
        vol_img = reader.Execute()
        array_4d[idx] = sitk.GetArrayFromImage(vol_img).astype(np.float32)
        del vol_img
        
    return array_4d

def extract_raw_time_course(all_raw_arrays, grouped_filepaths, liver_nifti, aorta_nifti, pv_nifti, t0=None):
    """Loops through raw 4D phases, applying Rescale Slope/Intercept metadata to gather signal intensities."""
    results = []

    # Load masks
    m_liver = sitk.GetArrayFromImage(sitk.ReadImage(liver_nifti))
    m_aorta = sitk.GetArrayFromImage(sitk.ReadImage(aorta_nifti))
    m_pv = pv_nifti if isinstance(pv_nifti, np.ndarray) else sitk.GetArrayFromImage(sitk.ReadImage(pv_nifti))
    
    # Standardized LPS -> RAS mapping alignment flip
    mask_liver = np.flip(m_liver, axis=1)
    mask_aorta = np.flip(m_aorta, axis=1)
    mask_pv = np.flip(m_pv, axis=1)

    # Calculate segment volumes (1 through 8)
    valid_segments = [1, 2, 3, 4, 5, 6, 7, 8]
    seg_volumes = {i: np.sum(mask_liver == i) for i in valid_segments}
    total_liver_volume = sum(seg_volumes.values())

    if t0 is None:
        first_path = grouped_filepaths[0][0]
        ds_first = pydicom.dcmread(first_path, stop_before_pixels=True)
        t0 = get_precise_seconds_from_float(ds_first.AcquisitionTime)

    for phase_idx, phase_arr in enumerate(all_raw_arrays):
        current_path = grouped_filepaths[phase_idx][0]
        ds = pydicom.dcmread(current_path, stop_before_pixels=True)
        
        current_abs_time = get_precise_seconds_from_float(ds.AcquisitionTime)
        relative_time = current_abs_time - t0
        
        slope = float(ds.get("RescaleSlope", 1))
        intercept = float(ds.get("RescaleIntercept", 0))
        scaled_arr = phase_arr.astype(np.float64) * slope + intercept
        
        # --- FIXED: Keys verified against exact legacy file parameters ---
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
        
def main_pipeline_execution():
    global PATIENT_IDS, BASE_DRIVE, OUTPUT_BASE
    
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
                print(f"⚠️ Missing either the initial 'dyn' or 'dyn_post' series folders for Patient {pid}")
                continue

            print(f"📂 Selected PRE-Contrast: {os.path.basename(pre_folders[0])}")
            print(f"📂 Selected POST-Contrast: {os.path.basename(post_folders[0])}")
            
            ts_mask_liver = os.path.join(patient_out_dir, "ts_liver_segments.nii.gz")
            ts_mask_aorta = os.path.join(patient_out_dir, "ts_aorta_segment.nii.gz")
            ts_mask_portal = os.path.join(patient_out_dir, "ts_portal_segment.nii.gz")
            
            if not os.path.exists(ts_mask_liver) or not os.path.exists(ts_mask_aorta) or not os.path.exists(ts_mask_portal):
                print(f"⚠️ Segmentation mask files missing for Patient {pid}. Skipping timecourse extraction.")
                continue

            pre_phases = group_dicom_paths_by_time(pre_folders[0])
            post_phases = group_dicom_paths_by_time(post_folders[0])
            
            if not pre_phases or not post_phases:
                print(f"❌ No valid DICOM phases resolved for Patient {pid}.")
                continue
            
            # --- EXTRACT PRE-CONTRAST TIMECOURSE ---
            print(f"🔄 Streaming raw PRE DICOM files on-the-fly for Patient {pid}...")
            pre_4d_arr = stream_dicom_series_to_numpy(pre_phases)
            
            first_pre_slice = pre_phases[0][0]
            ds_baseline_pre = pydicom.dcmread(first_pre_slice, stop_before_pixels=True)
            global_t0_pre = get_precise_seconds_from_float(ds_baseline_pre.AcquisitionTime)
            
            print(f"📊 Analyzing raw PRE-contrast signal arrays...")
            df_pre = extract_raw_time_course(pre_4d_arr, pre_phases, ts_mask_liver, ts_mask_aorta, ts_mask_portal, t0=global_t0_pre)
            
            del pre_4d_arr
            gc.collect()

            # --- EXTRACT POST-CONTRAST TIMECOURSE ---
            print(f"🔄 Streaming raw POST DICOM files on-the-fly for Patient {pid}...")
            post_4d_arr = stream_dicom_series_to_numpy(post_phases)
            
            first_post_slice = post_phases[0][0]
            ds_baseline_post = pydicom.dcmread(first_post_slice, stop_before_pixels=True)
            global_t0_post = get_precise_seconds_from_float(ds_baseline_post.AcquisitionTime)
            
            print(f"📊 Analyzing raw POST-contrast signal arrays...")
            df_post = extract_raw_time_course(post_4d_arr, post_phases, ts_mask_liver, ts_mask_aorta, ts_mask_portal, t0=global_t0_post)
            
            del post_4d_arr
            gc.collect()

            # Export data with exact matching layout format keys
            pre_csv_path = os.path.join(patient_out_dir, f"patient_{pid}_pre_timecourse.csv")
            df_pre.to_csv(pre_csv_path, index=False)
            
            post_csv_path = os.path.join(patient_out_dir, f"patient_{pid}_post_timecourse.csv")
            df_post.to_csv(post_csv_path, index=False)
            print(f"✅ Extracted timecourse metrics saved successfully to {patient_out_dir}")
            
        except Exception as e:
            print(f"❌ Critical extraction failure on Subject {pid}: {e}")

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