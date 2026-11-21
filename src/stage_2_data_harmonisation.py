"""
Create a clean (harmonized) DICOM database for the HEPARIM GRASP DCE data.
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
DOWNLOAD_STAGE = 'stage_1_download_m' 
PROJECT = 'HEPARIM'

#SUBJECTS = ['HEPARIM_020'] for testing
SUBJECTS = None

# data to skip
EXCLUDE = []

# which data to include in the harmonized database
INCLUDE_DYNAMIC = True      # MR1 and MR2
INCLUDE_DELAYED = True      # include dealyed scans

# file name patterns from download stage
DYNAMIC_ZIP = re.compile(r'series_(\d+)-MR(\d+)\.zip$')     
DELAYED_ZIP = re.compile(r'series_(\d+)\.zip$')             


def visit_name(index, date):
    return f'Visit_{index}'


def visits_of(patient_path):
    visits = []
    for folder in patient_path.iterdir():
        if not folder.is_dir():
            continue
        date = folder.name.split('_')[-1]     # HEPARIM_020_20200123 -> 20200123
        visits.append((date, folder))
    return sorted(visits)


def copy_zip(zip_path, destination, temp_folder):
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

    # One empty marker file per finished visit, so a re-run can skip it
    done_dir = os.path.join(dir_output, 'done')
    os.makedirs(done_dir, exist_ok=True)

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
                marker = os.path.join(done_dir, f'{patient_id}_{study}.done')
                if os.path.exists(marker):
                    logging.info(f'{patient_id} {study} ({date}): already done, skipping')
                    continue
                try:
                    harmonize_visit(visit_path, database, patient_id, study, temp_folder)
                    open(marker, 'w').close()
                    logging.info(f'{patient_id} {study} ({date}): done')
                except Exception:
                    logging.exception(f'{patient_id} {study} ({date}): error')


def run(build, logfile):
    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)
    heparim(build, dir_output)


if __name__ == '__main__':

    build = r"C:\Users\mahil\OneDrive\Documents\Uni Yr 4\Final Proj\VScode - MRI\data"
    pipe.run_stage(run, build, PIPELINE, __file__)