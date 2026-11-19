"""
Inspect the downloaded DICOM zips for one HEPARIM subject.

For every series_*.zip it reads the DICOM headers (not the pixel data)
and prints one summary row per series, so we can see which series are
the DCE (GRASP) dynamics and how slices and time points are stored.

Usage (on the cluster, with the heparim environment active):
    python inspect_series.py /mnt/parscratch/users/$USER/heparim_data/grasp/stage_1_download_m/HEPARIM/HEPARIM_020

It also saves the table as series_summary.csv in the current folder.
"""
import io
import re
import csv
import sys
import zipfile
from pathlib import Path

import pydicom


FIELDS = [
    'zip', 'n_files', 'SeriesNumber', 'SeriesDescription', 'ImageType',
    'SequenceName', 'Manufacturer', 'Rows', 'Columns',
    'n_SliceLocation', 'n_AcquisitionTime', 'n_TriggerTime',
    'n_TemporalPositionIdentifier', 'first_AcquisitionTime',
    'last_AcquisitionTime',
]


def series_number_from_name(path):
    """Sort key: series_22.zip -> 22, series_1001-MR1.zip -> 1001.1"""
    m = re.match(r'series_(\d+)(?:-MR(\d+))?', path.stem)
    if not m:
        return (float('inf'), path.name)
    return (int(m.group(1)), int(m.group(2) or 0))


def summarise_zip(zip_path):
    """Read every DICOM header in one zip and summarise it."""
    headers = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith('/'):
                continue
            try:
                with zf.open(name) as f:
                    ds = pydicom.dcmread(io.BytesIO(f.read()),
                                         stop_before_pixels=True, force=True)
            except Exception:
                continue
            if 'SOPInstanceUID' not in ds:
                continue
            headers.append(ds)

    row = {'zip': zip_path.name, 'n_files': len(headers)}
    if not headers:
        return row

    first = headers[0]

    def get(ds, key):
        val = ds.get(key, '')
        if isinstance(val, pydicom.multival.MultiValue):
            return '\\'.join(str(v) for v in val)
        return str(val)

    for key in ['SeriesNumber', 'SeriesDescription', 'ImageType',
                'SequenceName', 'Manufacturer', 'Rows', 'Columns']:
        row[key] = get(first, key)

    for key in ['SliceLocation', 'AcquisitionTime', 'TriggerTime',
                'TemporalPositionIdentifier']:
        values = {get(ds, key) for ds in headers} - {''}
        row['n_' + key] = len(values)

    times = sorted({get(ds, 'AcquisitionTime') for ds in headers} - {''})
    row['first_AcquisitionTime'] = times[0] if times else ''
    row['last_AcquisitionTime'] = times[-1] if times else ''
    return row


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    subject_dir = Path(sys.argv[1])
    zips = sorted(subject_dir.rglob('series_*.zip'), key=series_number_from_name)
    if not zips:
        print(f'No series_*.zip files found under {subject_dir}')
        sys.exit(1)

    print(f'Found {len(zips)} zip files. Reading headers...')
    rows = []
    for i, zip_path in enumerate(zips, 1):
        rows.append(summarise_zip(zip_path))
        if i % 50 == 0:
            print(f'  {i}/{len(zips)} done')

    with open('series_summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Short on-screen version
    print()
    print(f"{'zip':<22}{'files':>6}{'slices':>7}{'times':>6}  "
          f"{'ImageType':<38}SeriesDescription")
    for r in rows:
        print(f"{r['zip']:<22}{r['n_files']:>6}{r.get('n_SliceLocation', ''):>7}"
              f"{r.get('n_AcquisitionTime', ''):>6}  "
              f"{r.get('ImageType', '')[:37]:<38}{r.get('SeriesDescription', '')}")
    print()
    print('Full table saved to series_summary.csv')


if __name__ == '__main__':
    main()