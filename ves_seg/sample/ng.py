import os, sys
from em_util.ng import *
import numpy as np
from connectomics.data.utils.data_io import readvol
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage import morphology

import neuroglancer
port = 9098
neuroglancer.set_server_bind_address(bind_address='localhost',bind_port=port)
viewer=neuroglancer.Viewer()

def screenshot(path='temp.png', save=True, show=True, size=None):
    if size == None: 
        ss = viewer.screenshot().screenshot.image_pixels
    else:
        ss = viewer.screenshot(size=size).screenshot.image_pixels
    if save:
        Image.fromarray(ss).save(path)
    if show:
        plt.imshow(ss)
        plt.show()

data_dir = 'sample'
vol_name = '7-13'

clahe = readvol(os.path.join(data_dir, vol_name+"_clahe.h5"))
mask = readvol(os.path.join(data_dir, vol_name+"_mask.h5"))
gt = readvol(os.path.join(data_dir, vol_name+"_ves.h5"))
pred = readvol(os.path.join(data_dir, vol_name+"_pred.h5"))

mask_orig = mask
for i in range(2):
    mask = morphology.binary_dilation(mask, morphology.ball(radius=2))
gt= gt * mask
pred = pred * mask

with viewer.txn() as s:
    s.layers.append(name='clahe',layer=ng_layer(clahe, res=[8, 8, 30], tt='image'))
    s.layers.append(name='gt',layer=ng_layer(gt, res=[8, 8, 30], tt='segmentation'))
    s.layers.append(name='mask',layer=ng_layer(mask_orig, res=[8, 8, 30]))
    s.layers.append(name='pred',layer=ng_layer(pred, res=[8, 8, 30], tt='segmentation'))

print(viewer)
