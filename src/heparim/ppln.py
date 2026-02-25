import heparim as ppln
from heparim.utils import pipe

PIPELINE = 'template'

def run(build):
    
    ppln.stage_1_restructure.run(build)


if __name__=='__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\template"
    pipe.run_script(run, BUILD, PIPELINE)