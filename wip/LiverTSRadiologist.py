import os
import dicom2nifti
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
        


def plot_volume_correlation(manual_path, ts_path, patient_id):
    """
    Creates a scatter plot comparing Manual vs AI volumes for segments S1-S8.
    X-axis: TotalSegmentator Volume
    Y-axis: Manual Volume
    """
    # 1. Load images and calculate voxel volume
    man_img = sitk.ReadImage(manual_path)
    ts_img = sitk.ReadImage(ts_path)
    
    spacing = man_img.GetSpacing()
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    
    man_arr = sitk.GetArrayFromImage(man_img)
    ts_arr = sitk.GetArrayFromImage(ts_img)
    
    # 2. Extract volumes for segments 1-8 (converted to cm3/ml for readability)
    man_volumes = []
    ts_volumes = []
    labels = [f"S{i}" for i in range(1, 9)]
    
    for i in range(1, 9):
        # Convert mm3 to ml (cm3) by dividing by 1000
        man_volumes.append((np.sum(man_arr == i) * voxel_vol) / 1000.0)
        ts_volumes.append((np.sum(ts_arr == i) * voxel_vol) / 1000.0)
    
    # 3. Create Scatter Plot
    plt.figure(figsize=(8, 8))
    
    # Plot each segment with a unique color/label
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for i in range(8):
        plt.scatter(ts_volumes[i], man_volumes[i], color=colors[i], label=labels[i], s=100, edgecolors='black', zorder=3)

    # 4. Add the Identity Line (y = x)
    max_val = max(max(man_volumes), max(ts_volumes)) * 1.1
    plt.plot([0, max_val], [0, max_val], color='gray', linestyle='--', linewidth=1, label='Perfect Agreement (y=x)', zorder=1)
    
    # 5. Formatting
    plt.xlabel('TotalSegmentator Volume (ml)', fontsize=12)
    plt.ylabel('Manual Volume (ml)', fontsize=12)
    plt.title(f'Volume Correlation: Patient {patient_id}', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    
    # 6. Save and Display
    plot_path = os.path.join(os.path.dirname(manual_path), f"volume_scatter_{patient_id}.png")
    plt.savefig(plot_path, dpi=300)
    plt.show()
    
    print(f"📈 Scatter plot saved to: {plot_path}")

def visualize_edit_and_save(image_path, mask_path, save_path=None):
    if save_path is None:
        save_path = mask_path

    print(f"--- Opening Napari for: {os.path.basename(image_path)} ---")
    
    ref_sitk = sitk.ReadImage(image_path)
    image_arr = sitk.GetArrayFromImage(ref_sitk)
    
    # --- FIX START ---
    # Read the mask and explicitly convert to 16-bit or 32-bit integer
    mask_sitk = sitk.ReadImage(mask_path)
    mask_arr = sitk.GetArrayFromImage(mask_sitk).astype(np.int32) 
    # --- FIX END ---
    
    viewer = napari.Viewer()
    viewer.add_image(image_arr, name='MRI Scan', colormap='gray')
    
    # Now this will work without the float64 error
    label_layer = viewer.add_labels(mask_arr, name='Liver Segmentation')
    
    print("\n🖌️ Edit and close Napari to save...")
    napari.run()
    
    # Convert back to SITK and ensure we save as Integer
    edited_data = label_layer.data
    edited_sitk = sitk.GetImageFromArray(edited_data)
    edited_sitk.CopyInformation(ref_sitk)
    
    # Cast back to UInt8 (standard for masks) before saving to disk
    edited_sitk = sitk.Cast(edited_sitk, sitk.sitkUInt8)
    sitk.WriteImage(edited_sitk, save_path)
    
    print(f"✅ Edits saved to: {save_path}")

def convert_dicom_to_nifti(dicom_directory, nifti_save_path):
    """
    Converts a directory of DICOM slices into a single NIfTI file.
    """
    os.makedirs(os.path.dirname(nifti_save_path), exist_ok=True)
    
    print(f"--- Converting DICOM: {dicom_directory} ---")
    try:
        # reorient_nifti=True ensures the output is in RAS orientation, 
        # which is the standard for most AI models.
        dicom2nifti.dicom_series_to_nifti(dicom_directory, nifti_save_path, reorient_nifti=True)
        print(f"Successfully saved NIfTI to: {nifti_save_path}")
        return True
    except Exception as e:
        print(f"Conversion failed: {e}")
        return False
    
def segment_liver_segments(input_nifti_path, output_mask_path):
    """
    Runs TotalSegmentator on a NIfTI file and saves the result using SimpleITK
    to ensure coordinate systems match.
    """
    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)

    print(f"--- Running TotalSegmentator on: {input_nifti_path} ---")
    try:
        # Run the model
        ts_nib_image = totalsegmentator(
            input=input_nifti_path, 
            output=None, 
            task="liver_segments_mr", 
            ml=True
        )
        
        # Convert Nibabel (x, y, z) to SimpleITK (z, y, x)
        ts_data = ts_nib_image.get_fdata()
        ts_sitk_var = sitk.GetImageFromArray(ts_data.transpose(2, 1, 0)) 
        
        # Copy exact Origin, Spacing, and Direction from the source NIfTI
        reference_img = sitk.ReadImage(input_nifti_path)
        ts_sitk_var.CopyInformation(reference_img)

        # Save the result
        sitk.WriteImage(ts_sitk_var, output_mask_path)
        print(f"Segmentation saved to: {output_mask_path}")
        return True

    except Exception as e:
        print(f"Segmentation failed: {e}")
        return False

def convert_mitk_suffix_mapping(mitk_file, reference_nifti, output_path):
    print(f"--- Suffix Mapping: {os.path.basename(mitk_file)} ---")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Extract the .mitk (zip)
        with zipfile.ZipFile(mitk_file, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)
        
        # 2. Setup Reference
        ref_img = sitk.ReadImage(reference_nifti)
        master_mask = sitk.Image(ref_img.GetSize(), sitk.sitkUInt8)
        master_mask.CopyInformation(ref_img)

        # 3. Scan for files ending in s1.nrrd ... s8.nrrd
        found_map = {}
        
        for root, _, files in os.walk(tmp_dir):
            for f in files:
                f_lower = f.lower()
                # Check each possible segment suffix
                for i in range(1, 9):
                    suffix = f"s{i}.nrrd"
                    if f_lower.endswith(suffix):
                        path = os.path.join(root, f)
                        print(f"✅ Match Found: {f} -> Assigned to Label {i}")
                        
                        # Load, Resample, and Weighted Merge
                        seg_img = sitk.ReadImage(path)
                        
                        resampler = sitk.ResampleImageFilter()
                        resampler.SetReferenceImage(ref_img)
                        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                        aligned = resampler.Execute(seg_img)
                        
                        # Convert to binary and multiply by segment ID (i)
                        binary_mask = sitk.Cast(aligned > 0, sitk.sitkUInt8)
                        weighted_mask = binary_mask * i
                        
                        # Layer into master mask
                        master_mask = sitk.Maximum(master_mask, weighted_mask)
                        found_map[i] = f

        # 4. Final Validation
        if not found_map:
            print("❌ Error: No files found ending in s1.nrrd through s8.nrrd.")
            return None

        # Report missing segments if any
        missing = [f"s{i}" for i in range(1, 9) if i not in found_map]
        if missing:
            print(f"⚠️ Warning: Did not find files for: {missing}")

        # 5. Save and Return
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sitk.WriteImage(master_mask, output_path)
        
        final_array = sitk.GetArrayFromImage(master_mask)
        print(f"Success! Final unique labels in array: {np.unique(final_array)}")
        
        return master_mask
    
# --- RUNNING THE CODE ---
# manual_var = convert_mitk_strict_s_labels(MITK_INPUT, REF_NIFTI, FINAL_OUTPUT)
def calculate_dice(manual_path, ts_path, num_labels=8):
    """
    User-provided Dice function using SimpleITK.
    Calculates Whole Liver Dice and individual segment Dice.
    """
    # Load images
    man_img = sitk.GetArrayFromImage(sitk.ReadImage(manual_path))
    ts_img = sitk.GetArrayFromImage(sitk.ReadImage(ts_path))

    results = {}

    # 1. Compare the whole liver (everything > 0)
    man_liver = (man_img > 0).astype(np.uint8)
    ts_liver = (ts_img > 0).astype(np.uint8)
    
    denom_global = np.sum(man_liver) + np.sum(ts_liver)
    if denom_global == 0:
        results['Whole_Liver'] = 1.0
    else:
        results['Whole_Liver'] = (2. * np.sum(man_liver * ts_liver)) / denom_global

    # 2. Compare individual segments (1-8)
    for label in range(1, num_labels + 1):
        man_mask = (man_img == label).astype(np.uint8)
        ts_mask = (ts_img == label).astype(np.uint8)
        
        sum_val = np.sum(man_mask) + np.sum(ts_mask)
        if sum_val == 0:
            dice_val = 1.0  # Both are empty, perfect match
        else:
            dice_val = (2. * np.sum(man_mask * ts_mask)) / sum_val
        
        results[f'Segment_{label}'] = dice_val

    return results

def calculate_best_match_dice(manual_sitk, ts_sitk, num_labels=8):
    # Convert SITK objects to numpy arrays
    man_img = sitk.GetArrayFromImage(sitk.ReadImage(manual_sitk))
    ts_img = sitk.GetArrayFromImage(sitk.ReadImage(ts_sitk))

    results = []

    # Iterate through each manual segment (S1 to S8)
    for m_label in range(1, num_labels + 1):
        man_mask = (man_img == m_label).astype(np.uint8)
        
        # If the manual segment is empty, skip it
        if np.sum(man_mask) == 0:
            continue

        best_dice = -1.0
        best_match_label = None

        # Compare this manual segment against EVERY AI segment (1 to 8)
        for t_label in range(1, num_labels + 1):
            ts_mask = (ts_img == t_label).astype(np.uint8)
            
            sum_val = np.sum(man_mask) + np.sum(ts_mask)
            if sum_val == 0:
                dice = 0.0
            else:
                dice = (2. * np.sum(man_mask * ts_mask)) / sum_val
            
            # Keep track of the highest score
            if dice > best_dice:
                best_dice = dice
                best_match_label = t_label
        
        results.append({
            'Manual_Segment': f'S{m_label}',
            'Best_AI_Match': f'Segment_{best_match_label}',
            'Dice_Score': round(best_dice, 4)
        })

    return results


def main_pipeline_execution():
    all_summary_results = []
    for pid in PATIENT_IDS:
        try:
            # 1. DYNAMIC PATH DISCOVERY
            # Find patient folder (handles "Patient 001", "Patient 001_v2", etc.)
            patient_search = os.path.join(BASE_DRIVE, f"Patient {pid}//Patient {pid}*")
            patient_folders = glob.glob(patient_search)
            if not patient_folders:
                print(f"❌ Folder not found for Patient {pid}")
                continue
            patient_root = patient_folders[0]
        
            # Find DICOM folder containing 'dyn_post' or 'RAVE'
            dicom_search = os.path.join(patient_root, "**", "*Dicom.seg*")
            dicom_folders = [f for f in glob.glob(dicom_search, recursive=True) if os.path.isdir(f)]
            if not dicom_folders:
                print(f"❌ No dyn_post DICOM found for Patient {pid}")
                continue
            input_dicom = dicom_folders[0]
        
            # Find MITK file
            mitk_search = os.path.join(patient_root, "**", f"Patient {pid}.mitk")
            mitk_files = glob.glob(mitk_search, recursive=True)
            if not mitk_files:
                print(f"❌ No .mitk file found for Patient {pid}")
                continue
            mitk_input = mitk_files[0]
        
            # 2. OUTPUT DIRECTORY SETUP
            patient_out_dir = os.path.join(OUTPUT_BASE, f"Patient_{pid}")
            os.makedirs(patient_out_dir, exist_ok=True)
            
            ref_nifti = os.path.join(patient_out_dir, f"patient_{pid}_mri.nii.gz")
            ts_mask = os.path.join(patient_out_dir, "ts_liver_segments.nii.gz")
            manual_mask = os.path.join(patient_out_dir, "manual_liver_segments.nii.gz")
        
            # 3. PROCESSING PIPELINE
            # Step A: DICOM -> NIfTI
            if convert_dicom_to_nifti(input_dicom, ref_nifti):
                
                # Step B: AI Segmentation
                # Note: This is the slowest step (approx 2-5 mins per patient)
                seg_ok = segment_liver_segments(ref_nifti, ts_mask)
                
                # Step C: MITK -> NIfTI
                convert_mitk_suffix_mapping(mitk_input, ref_nifti, manual_mask)
               
                visualize_edit_and_save(ref_nifti, ts_mask)  
                
                #Volume comparison
                if os.path.exists(manual_mask) and os.path.exists(ts_mask):
                        plot_volume_correlation(manual_mask, ts_mask, pid)
        
                # 4. ANALYSIS (Dice Scoring)
                if seg_ok and os.path.exists(manual_mask):
                    dice_metrics = calculate_dice(manual_mask, ts_mask)
                    
                    # Flatten the results for the DataFrame
                    for metric_name, score in dice_metrics.items():
                        all_summary_results.append({
                            "Patient_ID": pid,
                            "Structure": metric_name,
                            "Dice_Score": round(score, 4)
                        })
                    
                    print(f"✅ Patient {pid} completed successfully.")
                else:
                    print(f"⚠️ Patient {pid} failed during mask generation.")
            
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


