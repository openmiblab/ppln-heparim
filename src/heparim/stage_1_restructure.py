import os
import logging
import dbdicom as db

from heparim.utils import pipe

PIPELINE = 'heparim'


def run(build):

    logging.info("Stage 1 --- Restructure ---")

    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)
    dir_input = os.path.join(build, PIPELINE, 'DICOM', 'PATIENT 040 Diagnostic DICOM')
    
    db.copy([dir_input], [dir_output])

    logging.info("Stage 1. Finished Restructure .")



if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data"
    pipe.run_script(run, BUILD, PIPELINE)
