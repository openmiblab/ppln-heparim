"""
Segment the liver and liver segments on the harmonized HEPARIM DCE data,
and save a mosaic PNG per visit for visual checking.

Input  (from stage_2_data_harmonization):
    <build>/grasp/stage_2_data_harmonization/HEPARIM   (dbdicom database)
        <patient>/<visit>/DCE_dynamic_MR1   4D series (x, y, z, time)
        <patient>/<visit>/DCE_delayed_<n>   3D delayed scans

Output (this stage):
    <build>/grasp/stage_3_segmentation/HEPARIM          (dbdicom database)
        <patient>/<visit>/DCE_reference     3D image used for segmentation
        <patient>/<visit>/total_mr          all TotalSegmentator total_mr labels (liver = 5)
        <patient>/<visit>/liver             liver mask (0/1)
        <patient>/<visit>/liver_segments    liver segments 1-8      (if SEGMENT_LIVER_SEGMENTS)
        <patient>/<visit>/liver_vessels     liver vessels (CT model) (if SEGMENT_LIVER_VESSELS)
    <build>/grasp/stage_3_segmentation/mosaics/<patient>_<visit>_1.png, _2.png, _3.png
    <build>/grasp/stage_3_segmentation/mosaics/<patient>_<visit>_vessels_1.png ... (if vessels)

Mosaics are drawn with miblab_plot.mosaic_overlay, as in the iBEAt display stage.

Every output series is deleted before it is written, so the stage can be
re-run safely.
"""
import os
import logging

import numpy as np
import nibabel as nib

import dbdicom as db
from totalsegmentator.python_api import totalsegmentator
from miblab_plot import mosaic_overlay      # lab mosaic function (pip install miblab-plot)

from miblab import pipe


PIPELINE = 'grasp'
HARMONIZATION_STAGE = 'stage_2_data_harmonization'
PROJECT = 'HEPARIM'

SUBJECTS = ['HEPARIM_020']
#SUBJECTS = None

# Patients to skip
EXCLUDE = []

# ---- Which 3D image to segment on ----
# 'dynamic': mean of the last DYNAMIC_LAST_FRAMES time points of DYNAMIC_SERIES
#            (late time points have the most liver enhancement)
# 'delayed': one of the delayed scans, DELAYED_SERIES
REFERENCE = 'dynamic'
#REFERENCE = 'delayed'
DYNAMIC_SERIES = 'DCE_dynamic_MR1'      # or 'DCE_dynamic_MR2'
DYNAMIC_LAST_FRAMES = 20                # 215 = average of the whole dynamic
DELAYED_SERIES = 'DCE_delayed_8'        # DCE_delayed_1 ... DCE_delayed_8

# ---- What to segment ----
SEGMENT_LIVER_SEGMENTS = True           # liver_segments_mr: segments 1-8 (MR model)
SEGMENT_LIVER_VESSELS = False           # liver_vessels: CT-only model, untested on MR, change to true if approved

# TotalSegmentator settings
DEVICE = 'cpu'              # 'cpu' on normal nodes, 'gpu' on a GPU node
LIVER_LABEL = 5             # liver in task total_mr

# Mosaic settings (same as the iBEAt display stage)
# One PNG per view: the array is transposed to show the data along each axis
MOSAIC_VIEWS = [(0, 1, 2), (0, 2, 1), (2, 1, 0)]
MOSAIC_MARGIN = [15, 5, 2]


def series_name(entity):
    """('Visit_1', 0) -> 'Visit_1'; 'Visit_1' -> 'Visit_1'"""
    return entity[0] if isinstance(entity, tuple) else entity


def to_nifti(array, affine):
    """
    Wrap an array as a NIfTI image for TotalSegmentator.
    dbdicom affines use DICOM (LPS) patient coordinates; NIfTI expects RAS,
    so the first two axes are flipped. If the mosaics show the liver on the
    wrong side or the segmentation fails badly, check this conversion first.
    """
    lps_to_ras = np.diag([-1, -1, 1, 1])
    return nib.Nifti1Image(array.astype(np.float32), lps_to_ras @ affine)


def run_totseg(image, task):
    """Run one TotalSegmentator task and return a label array on the same grid."""
    seg = totalsegmentator(image, None, task=task, ml=True, device=DEVICE, quiet=True)
    return np.asarray(seg.dataobj).astype(np.int16)


def write(array, affine, series, ref):
    db.delete(series, not_exists_ok=True)
    db.write_volume((array, affine), series, ref=ref)


def rois_from_labels(labels, names):
    """
    Turn a label array into the dict mosaic_overlay expects:
    {roi name: 0/1 mask}. names[i] is the name for label value i.
    """
    rois = {}
    for value, name in names.items():
        mask = (labels == value).astype(np.int16)
        if mask.any():
            rois[name] = mask
    return rois


def save_mosaics(image, rois, png_base, patient_id, study):
    """
    Save one mosaic per view with miblab_plot.mosaic_overlay, like the iBEAt
    display stage: <png_base>_1.png, _2.png, _3.png.
    A failure here is logged but does not undo the segmentation.
    """
    if not rois:
        logging.warning(f'{patient_id} {study}: nothing to draw for {png_base}')
        return
    for cnt, transp in enumerate(MOSAIC_VIEWS, 1):
        png_file = f'{png_base}_{cnt}.png'
        try:
            view_rois = {name: mask.transpose(transp) for name, mask in rois.items()}
            mosaic_overlay(image.transpose(transp), view_rois, png_file, margin=MOSAIC_MARGIN)
            logging.info(f'{patient_id} {study}: mosaic saved to {png_file}')
        except Exception:
            logging.exception(f'{patient_id} {study}: could not make {png_file}')


def reference_series_name():
    if REFERENCE == 'dynamic':
        return DYNAMIC_SERIES
    if REFERENCE == 'delayed':
        return DELAYED_SERIES
    raise ValueError(f"REFERENCE must be 'dynamic' or 'delayed', not {REFERENCE!r}")


def read_reference(source_series):
    """Return the 3D reference image and its affine."""
    if REFERENCE == 'dynamic':
        dyn = db.volume(source_series, 'AcquisitionTime')
        reference = np.mean(dyn.values[..., -DYNAMIC_LAST_FRAMES:], axis=-1)
        return reference, dyn.affine
    vol = db.volume(source_series)          # delayed scans are already 3D
    return vol.values, vol.affine


def segment_visit(harm_db, out_db, mosaic_dir, patient_id, study):
    source_series = [harm_db, patient_id, study, reference_series_name()]

    # 1. Reference image
    logging.info(f'{patient_id} {study}: reading {reference_series_name()} ({REFERENCE})')
    reference, affine = read_reference(source_series)

    write(reference, affine, [out_db, patient_id, study, 'DCE_reference'], source_series)
    nifti = to_nifti(reference, affine)
    dyn_series = source_series      # reference series for writing the masks

    # 2. Whole-body MR labels, and the liver mask from them
    logging.info(f'{patient_id} {study}: running total_mr')
    total = run_totseg(nifti, 'total_mr')
    liver = (total == LIVER_LABEL).astype(np.int16)
    write(total, affine, [out_db, patient_id, study, 'total_mr'], dyn_series)
    write(liver, affine, [out_db, patient_id, study, 'liver'], dyn_series)
    logging.info(f'{patient_id} {study}: liver voxels = {int(liver.sum())}')

    # 3. Liver segments 1-8
    rois = {'liver': liver}
    if SEGMENT_LIVER_SEGMENTS:
        logging.info(f'{patient_id} {study}: running liver_segments_mr')
        segments = run_totseg(nifti, 'liver_segments_mr')
        write(segments, affine, [out_db, patient_id, study, 'liver_segments'], dyn_series)
        rois = rois_from_labels(segments, {i: f'liver_segment_{i}' for i in range(1, 9)})

    # 4. Mosaics for visual checking
    png_base = os.path.join(mosaic_dir, f'{patient_id}_{study}')
    save_mosaics(reference, rois, png_base, patient_id, study)

    # 5. Liver vessels (optional, CT-trained model) with their own mosaics
    if SEGMENT_LIVER_VESSELS:
        logging.info(f'{patient_id} {study}: running liver_vessels (CT model)')
        vessels = run_totseg(nifti, 'liver_vessels')
        write(vessels, affine, [out_db, patient_id, study, 'liver_vessels'], dyn_series)
        vessel_rois = rois_from_labels(vessels, {1: 'liver_vessels_1', 2: 'liver_vessels_2'})
        save_mosaics(reference, vessel_rois, f'{png_base}_vessels', patient_id, study)


def heparim(build, dir_output):
    harm_db = os.path.join(build, PIPELINE, HARMONIZATION_STAGE, PROJECT)
    out_db = os.path.join(dir_output, PROJECT)
    mosaic_dir = os.path.join(dir_output, 'mosaics')
    os.makedirs(out_db, exist_ok=True)
    os.makedirs(mosaic_dir, exist_ok=True)

    # Every visit that has a reference series in the harmonized database
    visits = sorted({
        (s[1], series_name(s[2]))
        for s in db.series(harm_db)
        if series_name(s[3]) == reference_series_name()
    })

    for patient_id, study in visits:
        if SUBJECTS is not None and patient_id not in SUBJECTS:
            continue
        if patient_id in EXCLUDE:
            logging.info(f'Skipping {patient_id} (in EXCLUDE list)')
            continue
        try:
            segment_visit(harm_db, out_db, mosaic_dir, patient_id, study)
            logging.info(f'{patient_id} {study}: done')
        except Exception:
            logging.exception(f'{patient_id} {study}: error')


def run(build, logfile):
    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)
    heparim(build, dir_output)


if __name__ == '__main__':

    build = r"C:\Users\mahil\OneDrive\Documents\Uni Yr 4\Final Proj\VScode - MRI\data"
    pipe.run_stage(run, build, PIPELINE, __file__)