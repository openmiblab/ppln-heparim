"""
Compare the MR1 and MR2 versions of one time point.

Usage (on the cluster, heparim environment active):
    python compare_mr1_mr2.py /mnt/parscratch/users/$USER/heparim_data/grasp/stage_1_download_m/HEPARIM/HEPARIM_020 1001
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pydicom


KEYS = ['SeriesNumber', 'SeriesDescription', 'ImageType', 'ImageComments',
        'AcquisitionNumber', 'AcquisitionTime', 'SliceThickness',
        'PixelSpacing', 'ImageOrientationPatient', 'Rows', 'Columns',
        'ProtocolName', 'SeriesInstanceUID']


def read_zip(zip_path):
    datasets = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith('/'):
                continue
            with zf.open(name) as f:
                ds = pydicom.dcmread(io.BytesIO(f.read()), force=True)
            if 'SOPInstanceUID' in ds:
                datasets.append(ds)
    datasets.sort(key=lambda ds: float(ds.get('SliceLocation', 0)))
    return datasets


def describe(label, datasets):
    print(f'===== {label}: {len(datasets)} files =====')
    first = datasets[0]
    for key in KEYS:
        print(f'  {key:<26}{first.get(key, "")}')
    locs = [float(ds.get('SliceLocation', 'nan')) for ds in datasets]
    print(f'  {"SliceLocation range":<26}{min(locs):.2f} to {max(locs):.2f}')
    print(f'  {"First ImagePosition":<26}{first.get("ImagePositionPatient", "")}')
    print(f'  {"Last ImagePosition":<26}{datasets[-1].get("ImagePositionPatient", "")}')
    try:
        pixels = np.stack([ds.pixel_array for ds in datasets])
        print(f'  {"Pixel mean / max":<26}{pixels.mean():.1f} / {pixels.max()}')
        return pixels
    except Exception as e:
        print(f'  Could not read pixels: {e}')
        return None


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    subject_dir = Path(sys.argv[1])
    number = sys.argv[2]

    mr1 = list(subject_dir.rglob(f'series_{number}-MR1.zip'))
    mr2 = list(subject_dir.rglob(f'series_{number}-MR2.zip'))
    if not mr1 or not mr2:
        print(f'Could not find series_{number}-MR1.zip and -MR2.zip under {subject_dir}')
        sys.exit(1)

    p1 = describe('MR1', read_zip(mr1[0]))
    p2 = describe('MR2', read_zip(mr2[0]))

    if p1 is not None and p2 is not None and p1.shape == p2.shape:
        print('===== Comparison =====')
        print(f'  Pixels identical: {np.array_equal(p1, p2)}')
        print(f'  Mean absolute difference: {np.abs(p1.astype(float) - p2.astype(float)).mean():.2f}')


if __name__ == '__main__':
    main()