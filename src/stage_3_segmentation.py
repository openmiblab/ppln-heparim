"""
Segment the liver and liver segments on the harmonized HEPARIM DCE data,
and save a mosaic.
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
HARMONIZATION_STAGE = 'stage_2_data_harmonisation'
PROJECT = 'HEPARIM'

SUBJECTS = ['HEPARIM_020']
#SUBJECTS = None

# data to skip
EXCLUDE = []

REFERENCE = 'dynamic'
#REFERENCE = 'delayed' in case i need delayed data
DYNAMIC_SERIES = 'DCE_dynamic_MR1'      # or MR2?
DYNAMIC_LAST_FRAMES = 20                
DELAYED_SERIES = 'DCE_delayed_8'        

SEGMENT_LIVER_SEGMENTS = True           
SEGMENT_LIVER_VESSELS = False           # change to true if approved

DEVICE = 'cpu'            
LIVER_LABEL = 5             

MOSAIC_VIEWS = [(0, 1, 2), (0, 2, 1), (2, 1, 0)]
MOSAIC_MARGIN = [15, 5, 2]


def series_name(entity):
    return entity[0] if isinstance(entity, tuple) else entity


def to_nifti(array, affine):
    lps_to_ras = np.diag([-1, -1, 1, 1])
    return nib.Nifti1Image(array.astype(np.float32), lps_to_ras @ affine)


def run_totseg(image, task):
    seg = totalsegmentator(image, None, task=task, ml=True, device=DEVICE, quiet=True)
    return np.asarray(seg.dataobj).astype(np.int16)


def write(array, affine, series, ref):
    db.delete(series, not_exists_ok=True)
    db.write_volume((array, affine), series, ref=ref)


def rois_from_labels(labels, names):
    rois = {}
    for value, name in names.items():
        mask = (labels == value).astype(np.int16)
        if mask.any():
            rois[name] = mask
    return rois


def save_mosaics(image, rois, png_base, patient_id, study):
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
    if REFERENCE == 'dynamic':
        dyn = db.volume(source_series, 'AcquisitionTime')
        reference = np.mean(dyn.values[..., -DYNAMIC_LAST_FRAMES:], axis=-1)
        return reference, dyn.affine
    vol = db.volume(source_series)          # delayed scans are already 3D
    return vol.values, vol.affine


def segment_visit(harm_db, out_db, mosaic_dir, patient_id, study):
    source_series = [harm_db, patient_id, study, reference_series_name()]

    logging.info(f'{patient_id} {study}: reading {reference_series_name()} ({REFERENCE})')
    reference, affine = read_reference(source_series)

    write(reference, affine, [out_db, patient_id, study, 'DCE_reference'], source_series)
    nifti = to_nifti(reference, affine)
    dyn_series = source_series      

    logging.info(f'{patient_id} {study}: running total_mr')
    total = run_totseg(nifti, 'total_mr')
    liver = (total == LIVER_LABEL).astype(np.int16)
    write(total, affine, [out_db, patient_id, study, 'total_mr'], dyn_series)
    write(liver, affine, [out_db, patient_id, study, 'liver'], dyn_series)
    logging.info(f'{patient_id} {study}: liver voxels = {int(liver.sum())}')

    rois = {'liver': liver}
    if SEGMENT_LIVER_SEGMENTS:
        logging.info(f'{patient_id} {study}: running liver_segments_mr')
        segments = run_totseg(nifti, 'liver_segments_mr')
        write(segments, affine, [out_db, patient_id, study, 'liver_segments'], dyn_series)
        rois = rois_from_labels(segments, {i: f'liver_segment_{i}' for i in range(1, 9)})

    png_base = os.path.join(mosaic_dir, f'{patient_id}_{study}')
    save_mosaics(reference, rois, png_base, patient_id, study)

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