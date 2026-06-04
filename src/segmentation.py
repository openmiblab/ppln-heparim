import os
import zipfile
import tempfile
import numpy as np
import SimpleITK as sitk
import dicom2nifti
import matplotlib.pyplot as plt
from totalsegmentator.python_api import totalsegmentator

def convert_dicom_to_nifti(dicom_directory, nifti_save_path):
    """Converts a directory of DICOM slices into a single NIfTI file[cite: 2]."""
    os.makedirs(os.path.dirname(nifti_save_path), exist_ok=True)
    try:
        dicom2nifti.dicom_series_to_nifti(dicom_directory, nifti_save_path, reorient_nifti=True)
        return True
    except Exception as e:
        print(f"Conversion failed: {e}")
        return False

def run_total_segmentator_task(input_nifti_path, output_mask_path, task, roi_subset=None):
    """General abstract wrapper for execution tasks inside TotalSegmentator[cite: 2]."""
    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    try:
        ts_nib_image = totalsegmentator(
            input=input_nifti_path, output=None, task=task, roi_subset=roi_subset, ml=True
        )
        ts_data = ts_nib_image.get_fdata()
        ts_sitk_var = sitk.GetImageFromArray(ts_data.transpose(2, 1, 0))
        
        reference_img = sitk.ReadImage(input_nifti_path)
        ts_sitk_var.CopyInformation(reference_img)
        sitk.WriteImage(ts_sitk_var, output_mask_path)
        return True
    except Exception as e:
        print(f"Segmentation task {task} failed: {e}")
        return False

def create_multi_organ_mosaic(image_path, liver_path, aorta_path, portal_path, output_png, num_slices=12):
    """Creates QC scatter mosaic across targeted structural planes[cite: 2]."""
    img_sitk = sitk.ReadImage(image_path)
    img_arr = sitk.GetArrayFromImage(img_sitk)
    combined_mask = np.zeros_like(img_arr, dtype=np.int16)

    if os.path.exists(liver_path):
        liver_data = sitk.GetArrayFromImage(sitk.ReadImage(liver_path))
        combined_mask[liver_data > 0] = liver_data[liver_data > 0]
    if os.path.exists(aorta_path):
        combined_mask[sitk.GetArrayFromImage(sitk.ReadImage(aorta_path)) > 0] = 9
    if os.path.exists(portal_path):
        combined_mask[sitk.GetArrayFromImage(sitk.ReadImage(portal_path)) > 0] = 10

    z_indices = np.where(np.any(combined_mask > 0, axis=(1, 2)))[0]
    if len(z_indices) == 0: return
    selected_indices = np.linspace(z_indices[0], z_indices[-1], num_slices, dtype=int)
    
    cols = 4
    rows = int(np.ceil(num_slices / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()
    cmap = plt.get_cmap('tab10', 11)

    for i, idx in enumerate(selected_indices):
        ax = axes[i]
        p5, p95 = np.percentile(img_arr[idx], (5, 95))
        ax.imshow(np.clip(img_arr[idx], p5, p95), cmap='gray')
        overlay = np.ma.masked_where(combined_mask[idx] == 0, combined_mask[idx])
        ax.imshow(overlay, cmap=cmap, alpha=0.6, interpolation='nearest', vmin=0, vmax=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_png, bbox_inches='tight', dpi=150)
    plt.close()

def convert_mitk_suffix_mapping(mitk_file, reference_nifti, output_path):
    """Unpacks and aligns native MITK labels maps into standardized segment configurations[cite: 2]."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(mitk_file, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)
        
        ref_img = sitk.ReadImage(reference_nifti)
        master_mask = sitk.Image(ref_img.GetSize(), sitk.sitkUInt8)
        master_mask.CopyInformation(ref_img)
        
        for root, _, files in os.walk(tmp_dir):
            for f in files:
                for i in range(1, 9):
                    if f.lower().endswith(f"s{i}.nrrd"):
                        seg_img = sitk.ReadImage(os.path.join(root, f))
                        resampler = sitk.ResampleImageFilter()
                        resampler.SetReferenceImage(ref_img)
                        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                        aligned = resampler.Execute(seg_img)
                        master_mask = sitk.Maximum(master_mask, sitk.Cast(aligned > 0, sitk.sitkUInt8) * i)
                        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sitk.WriteImage(master_mask, output_path)