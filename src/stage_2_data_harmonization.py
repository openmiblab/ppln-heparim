"""
Create a clean (harmonized) DICOM database for the HEPARIM GRASP DCE data.

Input  (from stage_1_download_m):
    <build>/grasp/stage_1_download_m/HEPARIM/<patient>/<patient>_<date>/series_*.zip

    Each visit folder contains:
      - series_1001-MR1.zip ... series_1215-MR1.zip : one GRASP time point each (reconstruction 1)
      - series_1001-MR2.zip ... series_1215-MR2.zip : one GRASP time point each (reconstruction 2)
      - series_22.zip ... series_32.zip             : single delayed time points (scanner-reconstructed)

Output (this stage):
    <build>/grasp/stage_2_data_harmonization/HEPARIM  (a dbdicom database)
        <patient>/<visit>/DCE_dynamic_MR1   all MR1 time points merged into one 4D series
        <patient>/<visit>/DCE_dynamic_MR2   all MR2 time points merged into one 4D series
        <patient>/<visit>/DCE_delayed_<n>   each delayed time point as its own series

Each output series is deleted before it is written, so the stage can be
re-run safely without clearing the output folder first.
"""
import os
import re
import shutil
import logging
import zipfile
import tempfile
from pathlib import Path

import dbdicom as db

from miblab import pipe


PIPELINE = 'grasp'
DOWNLOAD_STAGE = 'stage_1_download_m'   # folder name of the download stage output
PROJECT = 'HEPARIM'

# Set to a list such as ['HEPARIM_020'] to test on a few patients first.
# Set to None to process every patient.
SUBJECTS = None

# Patients to skip (e.g. data that needs special handling)
EXCLUDE = []

# Which parts to write to the clean database
INCLUDE_DYNAMIC = True      # GRASP dynamics (MR1 and MR2)
INCLUDE_DELAYED = True      # single delayed time points (series_22 etc.) potentially needs changin to false to putdelayed scans as separate series

# File name patterns from the download stage
DYNAMIC_ZIP = re.compile(r'series_(\d+)-MR(\d+)\.zip$')     # e.g. series_1001-MR1.zip
DELAYED_ZIP = re.compile(r'series_(\d+)\.zip$')             # e.g. series_22.zip


def visit_name(index, date):
    """
    Name of the study (visit) in the clean database.
    index: 1 for the first visit by date, 2 for the second, ...
    date:  'YYYYMMDD' from the folder name

    Change this one line if Steven prefers another naming scheme,
    e.g. return date, or return ['Baseline', 'Followup'][index - 1]
    """
    return f'Visit_{index}'


def visits_of(patient_path):
    """Return the visit folders of a patient sorted by date: [(date, path), ...]"""
    visits = []
    for folder in patient_path.iterdir():
        if not folder.is_dir():
            continue
        date = folder.name.split('_')[-1]     # HEPARIM_020_20200123 -> 20200123
        visits.append((date, folder))
    return sorted(visits)


def copy_zip(zip_path, destination, temp_folder):
    """Unzip one series, copy it into the destination series, then clean up."""
    extract_to = os.path.join(temp_folder, zip_path.stem)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        source_series = db.series(extract_to)
        if len(source_series) != 1:
            raise ValueError(f'expected 1 series in zip, found {len(source_series)}')
        db.copy(source_series[0], destination)
    finally:
        shutil.rmtree(extract_to, ignore_errors=True)


def harmonize_visit(visit_path, database, patient_id, study, temp_folder):
    zips = list(visit_path.glob('series_*.zip'))

    # GRASP dynamics: one destination series per reconstruction (MR1, MR2),
    # time points added in order of series number (= order in time)
    if INCLUDE_DYNAMIC:
        dynamic = {}
        for zip_path in zips:
            m = DYNAMIC_ZIP.match(zip_path.name)
            if m:
                number, recon = int(m.group(1)), int(m.group(2))
                dynamic.setdefault(recon, []).append((number, zip_path))

        for recon, frames in sorted(dynamic.items()):
            frames.sort()
            destination = [database, patient_id, study, f'DCE_dynamic_MR{recon}']
            db.delete(destination, not_exists_ok=True)    # remove any old copy first
            logging.info(f'{patient_id} {study}: writing {len(frames)} time points to DCE_dynamic_MR{recon}')
            for number, zip_path in frames:
                copy_zip(zip_path, destination, temp_folder)

    # Delayed time points: each one kept as its own series
    # (they do not all have the same number of slices)
    if INCLUDE_DELAYED:
        delayed = []
        for zip_path in zips:
            m = DELAYED_ZIP.match(zip_path.name)
            if m:
                delayed.append((int(m.group(1)), zip_path))
        delayed.sort()
        for n, (number, zip_path) in enumerate(delayed, 1):
            destination = [database, patient_id, study, f'DCE_delayed_{n}']
            db.delete(destination, not_exists_ok=True)    # remove any old copy first
            logging.info(f'{patient_id} {study}: writing series_{number} to DCE_delayed_{n}')
            copy_zip(zip_path, destination, temp_folder)


def heparim(build, dir_output):
    download_path = Path(build) / PIPELINE / DOWNLOAD_STAGE / PROJECT
    database = os.path.join(dir_output, PROJECT)
    os.makedirs(database, exist_ok=True)

    patients = sorted(p for p in download_path.iterdir() if p.is_dir())
    if SUBJECTS is not None:
        patients = [p for p in patients if p.name in SUBJECTS]

    with tempfile.TemporaryDirectory() as temp_folder:
        for patient_path in patients:
            patient_id = patient_path.name           # IDs kept as they are, e.g. HEPARIM_020
            if patient_id in EXCLUDE:
                logging.info(f'Skipping {patient_id} (in EXCLUDE list)')
                continue
            for index, (date, visit_path) in enumerate(visits_of(patient_path), 1):
                study = visit_name(index, date)
                try:
                    harmonize_visit(visit_path, database, patient_id, study, temp_folder)
                    logging.info(f'{patient_id} {study} ({date}): done')
                except Exception:
                    logging.exception(f'{patient_id} {study} ({date}): error')


def run(build, logfile):
    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)
    heparim(build, dir_output)


if __name__ == '__main__':

    build = r"C:\Users\mahil\OneDrive\Documents\Uni Yr 4\Final Proj\VScode - MRI\data"
    pipe.run_stage(run, build, PIPELINE, __file__)