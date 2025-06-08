# dependencies
import os
import cv2
import glob
import tqdm
from connectomics.data.utils.data_io import readvol, savevol

def apply_clahe(in_dir, out_dir, cL=2.0, tGS=(7,7)):
"""
apply_clahe: apply CLAHE on all 3D volumes within some directory. CLAHE is applied to every 2D slice within each volume

Inputs:
    in_dir (str): the directory of raw images to apply CLAHE on
    out_dir (str): the target directory to send CLAHE images to
    cL (float): contrast limit for CLAHE
    tGS (list): grid size for CLAHE
""" 

    clahe = cv2.createCLAHE(clipLimit=cL, tileGridSize=tGS)

    # apply CLAHE to all images in in_dir, saving the result to out_dir
    for fname in tqdm.tqdm([fname for fname in os.listdir(in_dir) if 'im' in fname]):
        data = readvol(os.path.join(in_dir, fname))
        for idx, layer in enumerate(data):
            data[idx] = clahe.apply(layer)
        savevol(os.path.join(out_dir, fname.replace('im', 'clahe')), data)
    
    # exit the function
    return None
