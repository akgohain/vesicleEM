import sys
import numpy as np
import h5py
import yaml


def read_h5(filename, dataset=None):
    """
    Read data from an HDF5 file.

    Args:
        filename (str): The path to the HDF5 file.
        dataset (str or list, optional): The name or names of the dataset(s) to read. Defaults to None.
        chunk_id (int, optional): The ID of the chunk to read. Defaults to 0.
        chunk_num (int, optional): The total number of chunks. Defaults to 1.

    Returns:
        numpy.ndarray or list: The data from the HDF5 file.

    """
    fid = h5py.File(filename, "r")
    if dataset is None:
        dataset = fid.keys() if sys.version[0] == "2" else list(fid)
    else:
        if not isinstance(dataset, list):
            dataset = list(dataset)

    out = [None] * len(dataset)
    for di, d in enumerate(dataset):
        out[di] = np.array(fid[d])

    return out[0] if len(out) == 1 else out

def merge_bbox(bbox_a, bbox_b):
    """
    Merge two bounding boxes.

    Args:
        bbox_a (numpy.ndarray): The first bounding box.
        bbox_b (numpy.ndarray): The second bounding box.

    Returns:
        numpy.ndarray: The merged bounding box. Each row: [ymin,ymax,xmin,xmax,count(optional)]    
    """
    num_element = len(bbox_a) // 2 * 2
    out = bbox_a.copy()
    out[: num_element: 2] = np.minimum(bbox_a[: num_element: 2], 
                                       bbox_b[: num_element: 2])
    out[1: num_element: 2] = np.maximum(bbox_a[1: num_element: 2], 
                                        bbox_b[1: num_element: 2])
    if num_element != len(bbox_a): 
        out[-1] = bbox_a[-1] + bbox_b[-1]
    
    return out

def read_yml(filename):
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
    return data

def neuron_name_to_id(name):
    dict = read_yml('/data/projects/weilab/dataset/hydra/mask_mip1/neuron_id.txt')
    if isinstance(name, str):
        name = [name]    
    return [dict[x] for x in name]    

names= ['KR5','KR6']
sid = neuron_name_to_id(names)
bbox = np.loadtxt('/data/projects/weilab/dataset/hydra/mask_mip1/bbox.txt').astype(int)
bb1 = bbox[bbox[:,0]==sid[0], 1:][0]//[1,1,4,4,4,4]
bb2 = bbox[bbox[:,0]==sid[1], 1:][0]//[1,1,4,4,4,4]
bb_all = merge_bbox(bb1, bb2)
bb_all_sz = (bb_all[1::2] - bb_all[::2]) + 1

D0= '/data/projects/weilab/dataset/hydra/results/' 
out = np.zeros(bb_all_sz, np.uint8)
out[bb1[0]-bb_all[0]:bb1[1]-bb_all[0]+1, \
    bb1[2]-bb_all[2]:bb1[3]-bb_all[2]+1, \
    bb1[4]-bb_all[4]:bb1[5]-bb_all[4]+1] = sid[0] * read_h5(f'{D0}neuron_{names[0]}_30-32-32.h5')

out[bb2[0]-bb_all[0]:bb2[1]-bb_all[0]+1, \
    bb2[2]-bb_all[2]:bb2[3]-bb_all[2]+1, \
    bb2[4]-bb_all[4]:bb2[5]-bb_all[4]+1] = sid[1] * read_h5(f'{D0}neuron_{names[1]}_30-32-32.h5')