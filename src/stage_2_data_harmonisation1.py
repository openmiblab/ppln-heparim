"""
Create a clean (harmonized) DICOM database for the HEPARIM GRASP DCE data.
"""
import os
import re
import csv
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

# any dynamic group with fewer time points than this is incomplete and skipped
MIN_FRAMES = 200

ZIP_NAME = re.compile(r'series_(\d+)(?:-MR(\d+))?\.zip$')


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


def group_series(visit_path):
    dynamic, delayed = {}, []
    for zip_path in visit_path.glob('series_*.zip'):
        m = ZIP_NAME.match(zip_path.name)
        if not m:
            logging.warning(f'Unexpected file name, ignored: {zip_path}')
            continue
        number = int(m.group(1))
        mr = int(m.group(2)) if m.group(2) else 0
        if number >= 1000:
            dynamic.setdefault((number // 1000, mr), []).append((number, zip_path))
        else:
            delayed.append((mr, number, zip_path))
    for frames in dynamic.values():
        frames.sort()
    delayed.sort()
    return dynamic, delayed


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


def source_label(block, mr, frames):
    first, last = frames[0][0], frames[-1][0]
    label = f'series_{first}-{last}'
    return label + (f' (MR{mr})' if mr else '')


def harmonize_visit(visit_path, database, patient_id, study, date, temp_folder, rows):
    dynamic, delayed = group_series(visit_path)

    if INCLUDE_DYNAMIC:
        k = 0
        for (block, mr), frames in sorted(dynamic.items()):
            source = source_label(block, mr, frames)
            if len(frames) < MIN_FRAMES:
                logging.warning(f'{patient_id} {study}: {source} has only {len(frames)} '
                                f'time points (< {MIN_FRAMES}), skipped')
                rows.append([patient_id, study, date, '', source, len(frames), 'skipped: incomplete'])
                continue
            k += 1
            name = f'DCE_dynamic_{k}'
            destination = [database, patient_id, study, name]
            db.delete(destination, not_exists_ok=True)    # remove any old copy first
            logging.info(f'{patient_id} {study}: writing {len(frames)} time points from {source} to {name}')
            for number, zip_path in frames:
                copy_zip(zip_path, destination, temp_folder)
            rows.append([patient_id, study, date, name, source, len(frames), 'ok'])

    if INCLUDE_DELAYED:
        counter = {}
        for mr, number, zip_path in delayed:
            counter[mr] = counter.get(mr, 0) + 1
            name = f'DCE_delayed_{counter[mr]}' + (f'_MR{mr}' if mr else '')
            destination = [database, patient_id, study, name]
            db.delete(destination, not_exists_ok=True)    # remove any old copy first
            logging.info(f'{patient_id} {study}: writing series_{number} to {name}')
            copy_zip(zip_path, destination, temp_folder)
            source = f'series_{number}' + (f' (MR{mr})' if mr else '')
            rows.append([patient_id, study, date, name, source, 1, 'ok'])


def append_rows(csv_file, rows):
    new = not os.path.exists(csv_file)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(['patient', 'visit', 'date', 'output_series', 'source', 'n_series', 'status'])
        writer.writerows(rows)


def heparim(build, dir_output):
    download_path = Path(build) / PIPELINE / DOWNLOAD_STAGE / PROJECT
    databases = os.path.join(dir_output, PROJECT)
    done_dir = os.path.join(dir_output, 'done')
    csv_file = os.path.join(dir_output, 'series_map.csv')
    os.makedirs(databases, exist_ok=True)
    os.makedirs(done_dir, exist_ok=True)

    patients = sorted(p for p in download_path.iterdir() if p.is_dir())
    if SUBJECTS is not None:
        patients = [p for p in patients if p.name in SUBJECTS]

    with tempfile.TemporaryDirectory() as temp_folder:
        for patient_path in patients:
            patient_id = patient_path.name
            if patient_id in EXCLUDE:
                logging.info(f'Skipping {patient_id} (in EXCLUDE list)')
                continue
            database = os.path.join(databases, patient_id)
            for index, (date, visit_path) in enumerate(visits_of(patient_path), 1):
                study = visit_name(index, date)
                marker = os.path.join(done_dir, f'{patient_id}_{study}.done')
                if os.path.exists(marker):
                    logging.info(f'{patient_id} {study} ({date}): already done, skipping')
                    continue
                rows = []
                try:
                    harmonize_visit(visit_path, database, patient_id, study, date, temp_folder, rows)
                    append_rows(csv_file, rows)
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