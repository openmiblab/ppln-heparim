import os
import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
import zipfile
import tempfile
import matplotlib.pyplot as plt

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
    plot_path = os.path.join(os.path.dirname(manual_path), f"Mohamed_volume_scatter_{patient_id}.png")
    plt.savefig(plot_path, dpi=300)
    plt.show()
    
    print(f"📈 Scatter plot saved to: {plot_path}")



def convert_mitk_suffix_mapping(mitk_file, reference_nifti, output_path):
    print(f"--- Suffix Mapping: {os.path.basename(mitk_file)} ---")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Extract the .mitk (zip)
        with zipfile.ZipFile(mitk_file, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)
            print("Files found in MITK bundle:", os.listdir(tmp_dir)) 
            # Or for nested files:
            for r, d, f in os.walk(tmp_dir):
                print(f"Directory: {r} contains Files: {f}")
        
        # 2. Setup Reference & Master Mask
        ref_img = sitk.ReadImage(reference_nifti)
        master_mask = sitk.Image(ref_img.GetSize(), sitk.sitkUInt8)
        master_mask.CopyInformation(ref_img)

        # Pre-configure resampler for performance
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref_img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)

        found_map = {}
        
        # 3. Scan for files
        for root, _, files in os.walk(tmp_dir):
            for f in files:
                f_lower = f.lower()
                for i in range(1, 9):
                    # Handles both "Segment 1.nrrd" and "s1.nrrd"
                    suffixes = [f"segment_{i}.nrrd", f"s{i}.nrrd"]
                    
                    if any(f_lower.endswith(s) for s in suffixes):
                        path = os.path.join(root, f)
                        print(f"✅ Match Found: {f} -> Label {i}")
                        
                        seg_img = sitk.ReadImage(path)
                        aligned = resampler.Execute(seg_img)
                        
                        # Convert to binary and apply ID
                        # Note: aligned > 0 creates a mask where any non-zero intensity 
                        # is treated as the segment.
                        weighted_mask = sitk.Cast(aligned > 0, sitk.sitkUInt8) * i
                        
                        # Use Maximum to layer. Higher IDs overwrite lower IDs in overlaps.
                        master_mask = sitk.Maximum(master_mask, weighted_mask)
                        found_map[i] = f

        # 4. Final Validation
        if not found_map:
            print("❌ Error: No matching segment files found.")
            return None

        missing = [i for i in range(1, 9) if i not in found_map]
        if missing:
            print(f"⚠️ Warning: Missing labels: {missing}")

        # 5. Save and Return
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sitk.WriteImage(master_mask, output_path)
        
        # Quick summary stats
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(master_mask)
        print(f"Success! Labels present: {stats.GetLabels()}")
        
        return master_mask
    
    
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

def run_mohamed_validation():
    # --- CONFIGURATION ---
    PATIENT_IDS = ["001", "004", "005", "008", "010", "013", "019", "020", "022", "027"]
    AI_BASE_DIR = r"X:\abdominal_imaging\Shared\HEPARIMTS\batch_processing"
    MOHAMED_BASE = r"G:\Shared drives\HEPARIM\FINALISED MISK PATIENTS BY MOHAMED"
    
    all_summary_results = []

    for pid in PATIENT_IDS:
        print(f"\n--- Processing Patient {pid} ---")
        
        # Paths for AI data
        ai_nifti = os.path.join(AI_BASE_DIR, f"Patient_{pid}", "ts_liver_segments.nii.gz")
        ref_mri = os.path.join(AI_BASE_DIR, f"Patient_{pid}", f"patient_{pid}_mri.nii.gz")
        
        # Paths for Mohamed's MITK
        mohamed_folder = MOHAMED_BASE
        
        # Use an f-string to insert the number and add the .mitk extension
        num = str(int(pid))
        mitk_search_pattern = os.path.join(mohamed_folder, f"*{num}*.mitk")
        
        # Find the files
        mitk_files = glob.glob(mitk_search_pattern)
        
        if not mitk_files:
            # If it fails, print the pattern so you can see why it didn't match
            print(f"❌ No .mitk file found for Patient {num} using pattern: {mitk_search_pattern}")
        else:
            mitk_input = mitk_files[0]
            print(f"✅ Found MITK file: {mitk_input}")
        

        # Temporary NIfTI for Mohamed's data
        mohamed_nifti_out = os.path.join(AI_BASE_DIR, f"Patient_{pid}", "mohamed_finalized.nii.gz")

        try:
            # 1. Convert MITK to NIfTI (using your mapping function)
            convert_mitk_suffix_mapping(mitk_input, ref_mri, mohamed_nifti_out)
            
            #Volume comparison
            if os.path.exists(mohamed_nifti_out) and os.path.exists(ai_nifti):
                plot_volume_correlation(mohamed_nifti_out, ai_nifti, pid)
            
            # 2. Run your Custom Dice Function
            if os.path.exists(ai_nifti) and os.path.exists(mohamed_nifti_out):
                dice_metrics = calculate_dice(mohamed_nifti_out, ai_nifti)
                
                # Flatten the results for the DataFrame
                for metric_name, score in dice_metrics.items():
                    all_summary_results.append({
                        "Patient_ID": pid,
                        "Structure": metric_name,
                        "Dice_Score": round(score, 4)
                    })
                print(f"✅ Comparison complete for Patient {pid}")
            
        except Exception as e:
            print(f"❌ Error on Patient {pid}: {e}")

    # --- SAVE AND DISPLAY ---
    if all_summary_results:
        df = pd.DataFrame(all_summary_results)
        # Pivot for easier reading: Patients as rows, Structures as columns
        pivot_df = df.pivot(index='Patient_ID', columns='Structure', values='Dice_Score')
        
        output_path = os.path.join(AI_BASE_DIR, "mohamed_vs_ai_detailed_report.csv")
        pivot_df.to_csv(output_path)
        
        print("\n" + "="*60)
        print("DICE SCORE SUMMARY (Mohamed vs. AI)")
        print("="*60)
        print(pivot_df.to_string())
        print(f"\nReport saved to: {output_path}")

if __name__ == "__main__":
    run_mohamed_validation()