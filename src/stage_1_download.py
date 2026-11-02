"""
Automatic download of DIXON data from XNAT.
"""
import logging
from pathlib import Path

from miblab import pipe
from miblab_data.xnat import download_series


PIPELINE = 'dixon'

SIEMENS = {
    "series_description": [
        "T1w_abdomen_dixon_cor_bh", 
        "T1w_abdomen_dixon_cor_bh_opp", 
        "T1w_abdomen_dixon_cor_bh_in",
        "T1w_abdomen_dixon_cor_bh_F",
        "T1w_abdomen_dixon_cor_bh_W",
        "T1w_abdomen_post_contrast_dixon_cor_bh",
        "T1w_abdomen_post_contrast_dixon_cor_bh_opp",
        "T1w_abdomen_post_contrast_dixon_cor_bh_in",
        "T1w_abdomen_post_contrast_dixon_cor_bh_F",
        "T1w_abdomen_post_contrast_dixon_cor_bh_W"
    ]
} 
TURKU_PHILIPS = {
    "series_description": [
        'T1W-abdomen-Dixon-coronal-BH', 
        'T1W-abdomen-Dixon-post-coronal-BH'
    ]
}  
TURKU_GE = {
    "series_description": [
        "WATER: T1_abdomen_dixon_cor_bh", 
        "FAT: T1_abdomen_dixon_cor_bh",
        "InPhase: T1_abdomen_dixon_cor_bh",
        "OutPhase: T1_abdomen_dixon_cor_bh",
        "WATER: T1_abdomen_post_contrast_dixon_cor_bh",
        "FAT: T1_abdomen_post_contrast_dixon_cor_bh",
        "InPhase: T1_abdomen_post_contrast_dixon_cor_bh",
        "OutPhase: T1_abdomen_post_contrast_dixon_cor_bh"
    ]
} 
TURKU_GE_SETUP = {
    "series_description": [
        "WATER: T1_abdomen_dixon_cor_bh_iso", 
        "FAT: T1_abdomen_dixon_cor_bh_iso",
        "InPhase: T1_abdomen_dixon_cor_bh_iso",
        "OutPhase: T1_abdomen_dixon_cor_bh_iso",
        "WATER: T1_abdomen_dixon_cor_bh_npw_fip512",
        "FAT: T1_abdomen_dixon_cor_bh_npw_fip512",
        "InPhase: T1_abdomen_dixon_cor_bh_npw_fip512",
        "OutPhase: T1_abdomen_dixon_cor_bh_npw_fip512"
    ]
} 
SHEFFIELD_SETUP = {
    "series_description": [
        'T2star_map_kidneys_cor-oblique_mbh'
    ]
}
SHEFFIELD_PATIENTS = {
    "series_description": [
        # Philips data
        'T1w_abdomen_dixon_cor_bh', 
        'T1w_abdomen_post_contrast_dixon_cor_bh',
        # GE data
        'WATER: T1_abdomen_dixon_cor_bh',
        'FAT: T1_abdomen_dixon_cor_bh',
        'InPhase: T1_abdomen_dixon_cor_bh',
        'OutPhase: T1_abdomen_dixon_cor_bh',
        'WATER: T1_abdomen_post_contrast_dixon_cor_bh',
        'FAT: T1_abdomen_post_contrast_dixon_cor_bh',
        'InPhase: T1_abdomen_post_contrast_dixon_cor_bh',
        'OutPhase: T1_abdomen_post_contrast_dixon_cor_bh',
    ]
} 
BARI = {
    "series_description": [
        "T1w_abdomen_dixon_cor_bh", 
        "T1w_abdomen_post_contrast_dixon_cor_bh"
    ]
} 
LEEDS = {
    "parameters/sequence": ["*fl3d2"]
}
LEEDS_SETUP = {
    "parameters/sequence": ["*fl3d2"],
    "frames": [144]
} 



DOWNLOAD = {
    'leeds_patients':{
        'project_id': "BEAt-DKD-WP4-Leeds",
        'subject_label':"Leeds_Patients",
        'attr': LEEDS      
    },
    'leeds_volunteers':{
        'project_id': "BEAt-DKD-WP4-Leeds",
        'subject_label':"Leeds_volunteer_repeatability_study",
        'attr': LEEDS      
    },
    'leeds_setup':{
        'project_id': "BEAt-DKD-WP4-Leeds",
        'subject_label':"Leeds_setup_scans",
        'attr': LEEDS_SETUP      
    },
    'bari_patients':{
        'project_id': "BEAt-DKD-WP4-Bari",
        'subject_label':"Bari_Patients",
        'attr': BARI      
    },
    'bari_volunteers':{
        'project_id': "BEAt-DKD-WP4-Bari",
        'subject_label':"Bari_Volunteers_Repeatability",
        'attr': BARI      
    },
    'sheffield_patients':{
        'project_id': "BEAt-DKD-WP4-Sheffield",
        'attr': SHEFFIELD_PATIENTS      
    },
    'sheffield_setup':{
        'project_id': "ibeat_setup",
        'attr': SHEFFIELD_SETUP      
    },
    'turku_ge_patients':{
        'project_id': "BEAt-DKD-WP4-Turku",
        'subject_label':"Turku_Patients_GE",
        'attr': TURKU_GE       
    },
    'turku_ge_repeatability':{
        'project_id': "BEAt-DKD-WP4-Turku",
        'subject_label': "Turku_Volunteers_GE_Repeatability",
        'attr': TURKU_GE     
    },
    'turku_ge_setup':{
        'project_id': "BEAt-DKD-WP4-Turku",
        'subject_label': "Turku_GE_Setup_Tests",
        'attr': TURKU_GE_SETUP     
    },
    'turku_philips_patients':{
        'project_id': "BEAt-DKD-WP4-Turku",
        'subject_label': "Turku_Patients_Philips",
        'attr': TURKU_PHILIPS      
    },
    'turku_philips_repeatability':{
        'project_id': "BEAt-DKD-WP4-Turku",
        'subject_label': "Turku_volunteer_repeatability_study",
        'attr': TURKU_PHILIPS    
    },
    'bordeaux_patients_baseline':{
        'project_id': "BEAt-DKD-WP4-Bordeaux",
        'subject_label': "Bordeaux_Patients_Baseline",
        'attr': SIEMENS      
    },
    'bordeaux_volunteers':{
        'project_id': "BEAt-DKD-WP4-Bordeaux",
        'subject_label': "Bordeaux_Volunteers_Repeatability_Baseline",
        'attr': SIEMENS      
    },
    'bordeaux_patients_followup':{
        'project_id': "BEAt-DKD-WP4-Bordeaux",
        'subject_label': "Bordeaux_Patients_Followup",
        'attr': SIEMENS     
    },
    'exeter_patients_baseline':{
        'project_id': "BEAt-DKD-WP4-Exeter",
        'subject_label': "Exeter_Patients_Baseline",
        'attr': SIEMENS      
    },
    'exeter_patients_followup':{
        'project_id': "BEAt-DKD-WP4-Exeter",
        'subject_label': "Exeter_Patients_Followup",
        'attr': SIEMENS      
    },
    'exeter_volunteers':{
        'project_id': "BEAt-DKD-WP4-Exeter",
        'subject_label': "Exeter_Volunteer",
        'attr': SIEMENS     
    },
    'exeter_setup':{
        'project_id': "BEAt-DKD-WP4-Exeter",
        'subject_label': "Exeter_setup_scans",
        'attr': SIEMENS       
    },
}



def run(build, logfile):


    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)
    dir_ppln = Path(__file__).resolve().parents[2]

    for group, props in DOWNLOAD.items():
        logging.info(f"Start downloading {group}.")
        try:
            download_series(
                xnat_url="https://qib.shef.ac.uk",
                output_dir=dir_output,
                log=True,
                # n_max=1, # FOR DEBUGGING
                n_max=None,  # FINAL RUN
                cred=dir_ppln / 'user_XNAT.txt',
                **props
            )
            logging.info(f"Success downloading {group}.")
        except:
            logging.exception(f"Error downloading {group}.")



if __name__ == '__main__':

    build = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    pipe.run_stage(run, build, PIPELINE, __file__)