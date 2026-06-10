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

def process_liver_segmentation(dicom_directory, nifti_save_path, output_mask_dir):
    """
    Converts DICOM to NIfTI and segments liver parts.
    
    Args:
        dicom_directory (str): Path to folder containing DICOM slices.
        nifti_save_path (str): Path where the converted NIfTI will be saved.
        output_mask_dir (str): Directory for the segmentation outputs.
    """
    
    # 1. Create directories if they don't exist
    os.makedirs(os.path.dirname(nifti_save_path), exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    # 2. Convert DICOM to NIfTI
    print(f"--- Converting DICOM from {dicom_directory} ---")
    try:
        # This function reads the folder and creates one .nii.gz file
        dicom2nifti.dicom_series_to_nifti(dicom_directory, nifti_save_path, reorient_nifti=True)
        print(f"Successfully saved NIfTI to: {nifti_save_path}")
    except Exception as e:
        print(f"Conversion failed: {e}")
        return

    # 1. Run TotalSegmentator and assign the result to a variable
    # We set output=None to prevent it from auto-saving to a folder immediately
    print("--- Running TotalSegmentator to Variable ---")
    ts_nib_image = totalsegmentator(
        input=nifti_save_path, 
        output=None, 
        task="liver_segments_mr", 
        ml=True
    )
    
    # 2. Convert Nibabel variable to SimpleITK variable
    # This ensures it matches the format of your manual_segmentation_var
    ts_data = ts_nib_image.get_fdata()
    ts_sitk_var = sitk.GetImageFromArray(ts_data.transpose()) # Transpose often needed nib -> sitk
    ts_sitk_var.CopyInformation(sitk.ReadImage(nifti_save_path))
    
    # 3. Save the variable to NIfTI
    # 3. CONSTRUCT PATH AND CREATE DIRECTORY (The Fix)
    ts_save_path = os.path.join(output_mask_dir, "liver_segments_combined.nii.gz")
    
    # This line ensures the folder exists before the writer tries to touch it
    os.makedirs(os.path.dirname(ts_save_path), exist_ok=True)
    
    # 4. Save
    try:
        sitk.WriteImage(ts_sitk_var, ts_save_path)
        print(f"Successfully saved to {ts_save_path}")
    except Exception as e:
        print(f"WriteImage failed. Check if the file is open in another app: {e}")

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
    # Load images
    man_img = sitk.GetArrayFromImage(sitk.ReadImage(manual_path))
    ts_img = sitk.GetArrayFromImage(sitk.ReadImage(ts_path))

    results = {}

    # 1. Compare the whole liver (everything > 0)
    man_liver = (man_img > 0).astype(np.uint8)
    ts_liver = (ts_img > 0).astype(np.uint8)
    
    global_dice = (2. * np.sum(man_liver * ts_liver)) / (np.sum(man_liver) + np.sum(ts_liver))
    results['Whole_Liver'] = global_dice

    # 2. Compare individual segments (1-8)
    for label in range(1, num_labels + 1):
        man_mask = (man_img == label).astype(np.uint8)
        ts_mask = (ts_img == label).astype(np.uint8)
        
        sum_val = np.sum(man_mask) + np.sum(ts_mask)
        if sum_val == 0:
            dice = 1.0  # Both are empty, perfect match
        else:
            dice = (2. * np.sum(man_mask * ts_mask)) / sum_val
        
        results[f'Segment_{label}'] = dice

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



if __name__ == "__main__":
    # CONFIGURATION
    # Ensure this folder contains ONLY the slices for the specific DCE phase (e.g., Portal Venous)
    INPUT_DICOM_FOLDER = "G:\Shared drives\HEPARIM\Patients 001-029\Patient 029\ABDOMINAL_HEPARIM_20201002_151349_198000\RAVE_FB_FA11_DYN_FB_0032" 
    
    # Where to save the converted volume
    SAVE_NIFTI_FILE = "data\converted_mri.nii.gz"
    
    # Where to save the 8 segment masks
    FINAL_MASK_FOLDER = "C:\\Users\md1jbree\output\liver_segments\\"

    #process_liver_segmentation(INPUT_DICOM_FOLDER, SAVE_NIFTI_FILE, FINAL_MASK_FOLDER)
    # Your paths from the previous step
    MITK_INPUT = "G:\\Shared drives\HEPARIM\Patients 001-029\Patient 029\Patient 029\Patient 029.mitk"
    
    # This will be your "TotalSegmentator-style" manual file
    FINAL_OUTPUT = "output\manual_liver_segments\manual_liver_segments.nii"
    conversion_success = convert_dicom_to_nifti(INPUT_DICOM_FOLDER,  SAVE_NIFTI_FILE)
    if conversion_success:
        segment_liver_segments(SAVE_NIFTI_FILE, FINAL_MASK_FOLDER)
    convert_mitk_suffix_mapping(MITK_INPUT, SAVE_NIFTI_FILE, FINAL_OUTPUT)


    dice_scores = calculate_dice(FINAL_OUTPUT, FINAL_MASK_FOLDER + "liver_segments_combined.nii.gz")
    match_results = calculate_best_match_dice(FINAL_OUTPUT, FINAL_MASK_FOLDER + "liver_segments_combined.nii.gz")
    
    # Display results in a clean table
    df = pd.DataFrame(list(dice_scores.items()), columns=['Structure', 'Dice Score'])
    print("\n--- SEGMENTATION COMPARISON RESULTS ---")
    print(df.to_string(index=False))
    
    # Display results in a clean table
    df = pd.DataFrame(match_results)
    print("\n--- BEST-FIT COMPARISON TABLE ---")
    print(df.to_string(index=False))

