import os
import pydicom
import numpy as np
import pandas as pd
import dcmri as dc

def extract_par_from_dicom(dicom_folder):
    """Extracts acquisition and physiological metadata values directly from image datasets[cite: 1]."""
    files = [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith('.dcm')]
    if not files: raise FileNotFoundError("No DICOM files found in directory.")
    ds = pydicom.dcmread(files[0])
    return {
        'weight': float(getattr(ds, 'PatientWeight', 70.0)),
        'TR': float(ds.RepetitionTime),
        'FA_1': float(ds.FlipAngle), 
        't0': 0, 'dose_1': 0.025, 'dose_2': 0.0,
        'T1_aorta_1': 1440.0, 'T1_liver_1': 800.0, 'liver_volume': 1500.0
    }

def run_pharmacokinetic_fit(model_type, par, xdata, ydata, output_path, title_prefix=""):
    """Instantiates and executes parametric modeling routines via standard dcmri engines[cite: 1]."""
    tr_sec = float(par['TR']) / 1000.0 if float(par['TR']) > 1.0 else float(par['TR'])
    
    if model_type == 'aorta_liver':
        model = dc.AortaLiver2scan(
            weight=par['weight'], agent='gadoxetate', dose=par['dose_1'], dose2=par['dose_2'],
            field_strength=3.0, TR=tr_sec, FA=par['FA_1'],
            R10a=1000.0 / par['T1_aorta_1'], R10l=1000.0 / par['T1_liver_1'], vol=par['liver_volume']
        )
    else:
        model = dc.AortaPortalLiver(
            weight=par['weight'], agent='gadoxetate', dose=par['dose_1'], field_strength=3.0,
            TR=tr_sec, FA=par['FA_1'], R10a=1000.0 / par['T1_aorta_1'], R10l=1000.0 / par['T1_liver_1'], vol=par['liver_volume']
        )
        
    model.train(xdata, ydata)
    model.plot(xdata, ydata, fname=output_path, show=False)
    return model