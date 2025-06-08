import os,sys
from em_util.ng import *
import numpy as np
from connectomics.data.utils.data_io import readvol
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage import morphology
import neuroglancer

neuroglancer.set_server_bind_address(bind_address='localhost',bind_port=9098)
viewer=neuroglancer.Viewer()


def screenshot(path='temp.png', save=True, show=True, size=None):
"""
screenshot: use the screenshot capabilities built into neuroglancer to take high-resolution screenshots
note that neuroglancer has to load in visible layers before taking a screenshot, so this function may hang on larger volumes

Inputs:
    path (str): the path to save the screenshot to
    save (bool): whether or not to save the screenshot to path
    show (bool): whether or not to show the screenshot with matplotlib
    size ([int, int]): force a specific size of the screenshot. larger size leads to a larger FOV for the screenshot
"""
    if size == None: 
        ss = viewer.screenshot().screenshot.image_pixels
    else:
        ss = viewer.screenshot(size=size).screenshot.image_pixels
    if save:
        Image.fromarray(ss).save(path)
    if show:
        plt.imshow(ss)
        plt.show()

im_path = "image-path.h5"
mask_path = "mask_path.h5"
pred_path = "pred_path.h5"
res = [8, 8, 30] # in [x, y, z]

im = readvol(im_path).astype(np.uint8)
mask = readvol(mask_path).astype(np.uint8)
pred = readvol(pred_path) * mask
with viewer.txn() as s:
    s.layers.append(name='im',layer=ng_layer(im, res=res, tt='image'))
    s.layers.append(name='mask',layer=ng_layer(mask, res=res))
    s.layers.append(name='pred',layer=ng_layer(pred, res=res, tt='segmentation'))

print(viewer)
