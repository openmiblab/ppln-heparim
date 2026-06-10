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
        # Ensure we use utf-8 and 'replace' for the emoji issue
        self.file = open(file, 'w', encoding='utf-8', errors='replace')
        self.terminal = sys.stdout
  
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        # FORCE the data out of memory and into the file immediately
        self.file.flush() 
    
    def flush(self):
        self.terminal.flush()
        if hasattr(self.file, 'flush') and not self.file.closed:
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




def extract_par_from_dicom(dicom_folder):
    """
    Extracts acquisition and patient parameters from a folder of DICOM files.
    """
    # Load the first file in the folder to get headers
    files = [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith('.dcm')]
    if not files:
        raise FileNotFoundError("No DICOM files found in the directory.")
    
    ds = pydicom.dcmread(files[0])

    # Extracting Metadata
    par = {
        # Patient Parameters
        'weight': float(getattr(ds, 'PatientWeight', 70.0)),  # Default to 70kg if missing
        
        # Acquisition Parameters
        'TR': float(ds.RepetitionTime),                      # Repetition Time (ms)
        'FA_1': float(ds.FlipAngle), 
        't0': 0,                                             # Usually set to 0 for start of sequence
        
        # Injection Parameters (Often manual entry, but sometimes in tags)
        'dose_1': 0.025,
        'dose_2': 0.0,                                      # Standard Gadoxetate dose (mmol/kg)
        
        # Tissue Parameters (Defaults - ideally these come from a T1 Map)
        'T1_aorta_1': 1440.0,                                # Typical T1 of blood at 3T (ms)
        'T1_liver_1': 800.0,                                 # Typical T1 of liver at 3T (ms)
        
        # Liver Volume (Requires manual segmentation usually)
        'liver_volume': 1500.0                               # Default average liver volume (mL)
    }

    print("✅ Extracted Parameters from DICOM:")
    for k, v in par.items():
        print(f"  {k}: {v}")
        
    return par

def gadoxate_analysis(patient_out_dir, pid, par, delay_sec=1200):
    try:
        # 1. LOAD DATA
        df_pre = pd.read_csv(os.path.join(patient_out_dir, f"patient_{pid}_pre_timecourse.csv")).dropna()
        df_post = pd.read_csv(os.path.join(patient_out_dir, f"patient_{pid}_post_timecourse.csv")).dropna()
        
        # 2. PREPARE XDATA (Time points for the 4 segments)
        t_a_1 = df_pre['Time'].values.astype(np.float64)
        t_a_2 = df_post['Time'].values.astype(np.float64) + (t_a_1[-1] + delay_sec)
        t_l_1 = t_a_1.copy()
        t_l_2 = t_a_2.copy()
        
        xdata_a = (t_a_1, t_a_2, t_l_1, t_l_2)
        xdata_pv = (np.concatenate((t_a_1, t_a_2), axis=0),np.concatenate((t_a_1, t_a_2), axis=0),np.concatenate((t_a_1, t_a_2), axis=0))

        # 3. PREPARE YDATA (Signals for the 4 segments)
        # Note: Use raw signals; dcmri handles the signal-to-concentration math 
        # internally using the R10 and FA parameters provided in the constructor.
        s_a_1 = df_pre['Aorta'].values.astype(np.float64)
        s_a_2 = df_post['Aorta'].values.astype(np.float64)
        s_pv_1 = df_pre['Portal_Vein'].values.astype(np.float64)
        s_pv_2 = df_post['Portal_Vein'].values.astype(np.float64)
        s_l_1 = df_pre['Whole_Liver'].values.astype(np.float64)
        s_l_2 = df_post['Whole_Liver'].values.astype(np.float64)
        
        ydata_a = (s_a_1, s_a_2, s_l_1, s_l_2)
        ydata_pv = (np.concatenate((s_a_1, s_a_2), axis=0), np.concatenate((s_pv_1, s_pv_2), axis=0), np.concatenate((s_l_1, s_l_2), axis=0))

        # 4. INITIALIZE MODEL
        # Ensure TR is in seconds (DICOM is usually ms)
        tr_sec = float(par['TR']) / 1000.0 if float(par['TR']) > 1.0 else float(par['TR'])

        model_a = dc.AortaLiver2scan(
            weight = float(par['weight']),
            agent = 'gadoxetate',
            dose = float(par['dose_1']),
            dose2 = float(par['dose_2']),
            field_strength = 3.0,
            TR = tr_sec,
            FA = float(par['FA_1']),
            R10a = 1000.0 / float(par['T1_aorta_1']), 
            R10l = 1000.0 / float(par['T1_liver_1']),
            vol = float(par['liver_volume'])
        )
        
        model_pv = dc.AortaPortalLiver(
            weight = float(par['weight']),
            agent = 'gadoxetate',
            dose = float(par['dose_1']),
            field_strength = 3.0,
            TR = tr_sec,
            FA = float(par['FA_1']),
            R10a = 1000.0 / float(par['T1_aorta_1']), 
            R10l = 1000.0 / float(par['T1_liver_1']),
            vol = float(par['liver_volume'])
        )
        sys.stdout = Tee(os.path.join(patient_out_dir, f"Gadoxate_Analysis_patient_{pid}.txt"))


        # 5. TRAIN (Passing the Tuples)
        print(f"🚀 Training  whole liver model for Patient {pid}...")
        model_a.train(xdata_a, ydata_a)
        
        # 6. EXTRACT RESULTS
        params = model_a.export_params(type='dict')
        khe_val = params.get('khe', {}).get('value', 0) if isinstance(params.get('khe'), dict) else params.get('khe', 0)
        print(khe_val)
        khe_scaled = khe_val[1]
        
        # 7. VISUALIZATION
        # To plot the fit, we use the model's internal predict functionality
        fit_a1, fit_a2, fit_l1, fit_l2 = model_a.predict(xdata_a)
        
        model_a.plot(xdata_a, ydata_a,fname=os.path.join(patient_out_dir, f"patient_2_Scan_Aorta_intake_{pid}_fit.png"))
        model_a.print_params(round_to=None)
        
        model_pv.train(xdata_pv, ydata_pv)
        
        # 6. EXTRACT RESULTS
        params = model_pv.export_params(type='dict')
        khe_val = params.get('khe', {}).get('value', 0) if isinstance(params.get('khe'), dict) else params.get('khe', 0)
        print(khe_val)
        khe_scaled = khe_val
        
        # 7. VISUALIZATION
        # To plot the fit, we use the model's internal predict functionality
        fit_a1, fit_pv1,  fit_l2 = model_pv.predict(xdata_pv)
        
        model_pv.plot(xdata_pv, ydata_pv,fname=os.path.join(patient_out_dir, f"patient_dual_intake_{pid}_fit.png"))
        model_pv.print_params(round_to=None)
        #Repeat for each segment 
        for i in range(1, 9):
            col_name = f'S{i}'
            if col_name in df_pre.columns and col_name in df_post.columns:
                s_s_1 = df_pre[f'S{i}'].values.astype(np.float64)
                s_s_2 = df_post[f'S{i}'].values.astype(np.float64)
                
                ydata_s_a = (s_a_1, s_a_2, s_s_1, s_s_2)
                ydata_s_pv = (np.concatenate((s_a_1, s_a_2), axis=0), np.concatenate((s_pv_1, s_pv_2), axis=0), np.concatenate((s_s_1, s_s_2), axis=0))
                # 5. TRAIN (Passing the Tuples)
                print(f"🚀 Training  model for Patient {pid} segment S{i}...")
                model_a.train(xdata_a, ydata_s_a)
                
                # 6. EXTRACT RESULTS
                params = model_a.export_params(type='dict')
                khe_val = params.get('khe', {}).get('value', 0) if isinstance(params.get('khe'), dict) else params.get('khe', 0)
                print(khe_val)
                khe_scaled = khe_val[1]
                
                # 7. VISUALIZATION
                # To plot the fit, we use the model's internal predict functionality
                fit_a1, fit_a2, fit_l1, fit_l2 = model_a.predict(xdata_a)
                
                model_a.plot(xdata_a, ydata_s_a,fname=os.path.join(patient_out_dir, f"patient_2_Scan_Aorta_intake_S{i}_{pid}_fit.png"),show=False)
    
                model_a.print_params(round_to=None)
                # 5. TRAIN (Passing the Tuples)
    
                model_pv.train(xdata_pv, ydata_s_pv)
                
                # 6. EXTRACT RESULTS
                params = model_pv.export_params(type='dict')
                khe_val = params.get('khe', {}).get('value', 0) if isinstance(params.get('khe'), dict) else params.get('khe', 0)
                print(khe_val)
                khe_scaled = khe_val
                
                # 7. VISUALIZATION
                # To plot the fit, we use the model's internal predict functionality
                fit_a1, fit_pv1,  fit_l2 = model_pv.predict(xdata_pv)
                
                model_pv.plot(xdata_pv, ydata_s_pv,fname=os.path.join(patient_out_dir, f"patient_dual_intake_S{i}_{pid}_fit.png"),show=False)
    
                model_pv.print_params(round_to=None)
            else:
               print( f'S{i} segment not found')
                
        print(f"✅ Analysis complete for Patient {pid}.")

    except Exception as e:
        print(f"❌ Failure on Patient {pid}: {e}")
    sys.stdout = sys.__stdout__

        
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
          
            par = extract_par_from_dicom(pre_folders[0])
            
            # D. Plot
            print(f"📊 Gadoxate analysis for Patient {pid}...")
            gadoxate_analysis(patient_out_dir, pid, par)

        except Exception as e:
            print(f"❌ Critical failure on Patient {pid}: {e}")


                       
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
