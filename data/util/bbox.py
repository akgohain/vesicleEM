import os
import numpy as np
import h5py
from .io import read_h5
from tqdm import tqdm

def compute_bbox(seg, do_count=False):
    """
    Compute the bounding box of a binary segmentation.

    Args:
        seg (numpy.ndarray): The binary segmentation.
        do_count (bool, optional): Whether to compute the count of foreground pixels. Defaults to False.

    Returns:
        list: The bounding box of the foreground segment in the format [y0, y1, x0, x1, count (optional)].

    Notes:
        - The input segmentation can have any dimension.
        - If the segmentation is empty (no foreground pixels), None is returned.
        - The bounding box is computed as the minimum and maximum coordinates along each dimension that contain foreground pixels.
        - If `do_count` is True, the count of foreground pixels is included in the output.
    """    
    if not seg.any():
        return None

    out = []
    pix_nonzero = np.where(seg > 0)
    for i in range(seg.ndim):
        out += [pix_nonzero[i].min(), pix_nonzero[i].max()]

    if do_count:
        out += [len(pix_nonzero[0])]
    return out


def compute_bbox_all_chunk(seg_file, do_count=False, uid=None, chunk_num=1, no_tqdm=False):    
    if chunk_num == 1:
        if isinstance(seg_file):
            seg_file = read_h5(seg_file)
        return compute_bbox_all(seg_file, do_count, uid)
    else:
        fid = h5py.File(seg_file, 'r')
        seg = fid[list(fid)[0]]
        num_z = int(np.ceil(seg.shape[0] / float(chunk_num)))
        out = []
        for i in tqdm(range(chunk_num), disable=no_tqdm):
            chunk_bbox = compute_bbox_all(np.array(seg[i*num_z: (i+1)*num_z]), do_count, uid) 
            chunk_bbox[:,1:3] += i*num_z 
            if i == 0:
                out = chunk_bbox.copy()
            else:
                out = merge_bbox_two_matrices(out, chunk_bbox)
            # import pdb;pdb.set_trace()
        fid.close()
        return out 
    
def compute_bbox_all(seg, do_count=False, uid=None):
    """
    Compute the bounding boxes of segments in a segmentation map.

    Args:
        seg (numpy.ndarray): The input segmentation map.
        do_count (bool, optional): Whether to compute the segment counts. Defaults to False.
        uid (numpy.ndarray, optional): The segment IDs to compute the bounding boxes for. Defaults to None.

    Returns:
        numpy.ndarray: An array containing the bounding boxes of the segments.

    Raises:
        ValueError: If the input volume is not 2D or 3D.

    Notes:
        - The function computes the bounding boxes of segments in a segmentation map.
        - The bounding boxes represent the minimum and maximum coordinates of each segment in the map.
        - The function can compute the segment counts if `do_count` is set to True.
        - The bounding boxes are returned as an array.
    """    
    if seg.ndim == 2:
        return compute_bbox_all_2d(seg, do_count, uid)
    elif seg.ndim == 3:
        return compute_bbox_all_3d(seg, do_count, uid)
    else:
        raise "input volume should be either 2D or 3D" 

    
def compute_bbox_all_2d(seg, do_count=False, uid=None):
    """
    Compute the bounding boxes of 2D instance segmentation.

    Args:
        seg (numpy.ndarray): The 2D instance segmentation.
        do_count (bool, optional): Whether to compute the count of each instance. Defaults to False.
        uid (numpy.ndarray, optional): The unique identifier for each instance. Defaults to None.

    Returns:
        numpy.ndarray: The computed bounding boxes of the instances.

    Notes:
        - The input segmentation should have dimensions HxW, where H is the height and W is the width.
        - Each row in the output represents an instance and contains the following information:
            - seg id: The ID of the instance.
            - bounding box: The coordinates of the bounding box in the format [ymin, ymax, xmin, xmax].
            - count (optional): The count of pixels belonging to the instance.
        - If the `uid` argument is not provided, the unique identifiers are automatically determined from the segmentation.
        - Instances with no pixels are excluded from the output.
    """        
    sz = seg.shape
    assert len(sz) == 2
    if uid is None:
        uid = np.unique(seg)
        uid = uid[uid > 0]
    if len(uid) == 0:
        return None
    uid_max = uid.max()
    out = np.zeros((1 + int(uid_max), 5 + do_count), dtype=np.uint32)
    out[:, 0] = np.arange(out.shape[0])
    out[:, 1] = sz[0]
    out[:, 3] = sz[1]
    # for each row
    rids = np.where((seg > 0).sum(axis=1) > 0)[0]
    for rid in rids:
        sid = np.unique(seg[rid])
        sid = sid[(sid > 0) * (sid <= uid_max)]
        out[sid, 1] = np.minimum(out[sid, 1], rid)
        out[sid, 2] = np.maximum(out[sid, 2], rid)
    cids = np.where((seg > 0).sum(axis=0) > 0)[0]
    for cid in cids:
        sid = np.unique(seg[:, cid])
        sid = sid[(sid > 0) * (sid <= uid_max)]
        out[sid, 3] = np.minimum(out[sid, 3], cid)
        out[sid, 4] = np.maximum(out[sid, 4], cid)

    if do_count:
        seg_ui, seg_uc = np.unique(seg, return_counts=True)
        out[seg_ui, -1] = seg_uc
    return out[uid]


def compute_bbox_all_3d(seg, do_count=False, uid=None):
    """
    Compute the bounding boxes of 3D instance segmentation.

    Args:
        seg (numpy.ndarray): The 3D instance segmentation.
        do_count (bool, optional): Whether to compute the count of each instance. Defaults to False.
        uid (numpy.ndarray, optional): The unique identifier for each instance. Defaults to None.

    Returns:
        numpy.ndarray: The computed bounding boxes of the instances.

    Notes:
        - Each row in the output represents an instance and contains the following information:
            - seg id: The ID of the instance.
            - bounding box: The coordinates of the bounding box in the format [ymin, ymax, xmin, xmax, zmin, zmax].
            - count (optional): The count of voxels belonging to the instance.
        - The output only includes instances with valid bounding boxes.
    """

    sz = seg.shape
    assert len(sz) == 3, "Input segment should have 3 dimensions"
    if uid is None:
        uid = seg
    uid_max = int(uid.max())
    out = np.zeros((1 + uid_max, 7 + do_count), dtype=np.int32)
    out[:, 0] = np.arange(out.shape[0])
    out[:, 1] = sz[0]
    out[:, 2] = -1
    out[:, 3] = sz[1]
    out[:, 4] = -1
    out[:, 5] = sz[2]
    out[:, 6] = -1

    # for each slice
    zids = np.where((seg > 0).sum(axis=1).sum(axis=1) > 0)[0]
    for zid in zids:
        sid = np.unique(seg[zid])
        sid = sid[(sid > 0) * (sid <= uid_max)]
        out[sid, 1] = np.minimum(out[sid, 1], zid)
        out[sid, 2] = np.maximum(out[sid, 2], zid)

    # for each row
    rids = np.where((seg > 0).sum(axis=0).sum(axis=1) > 0)[0]
    for rid in rids:
        sid = np.unique(seg[:, rid])
        sid = sid[(sid > 0) * (sid <= uid_max)]
        out[sid, 3] = np.minimum(out[sid, 3], rid)
        out[sid, 4] = np.maximum(out[sid, 4], rid)

    # for each col
    cids = np.where((seg > 0).sum(axis=0).sum(axis=0) > 0)[0]
    for cid in cids:
        sid = np.unique(seg[:, :, cid])
        sid = sid[(sid > 0) * (sid <= uid_max)]
        out[sid, 5] = np.minimum(out[sid, 5], cid)
        out[sid, 6] = np.maximum(out[sid, 6], cid)

    if do_count:
        seg_ui, seg_uc = np.unique(seg, return_counts=True)
        out[seg_ui[seg_ui <= uid_max], -1] = seg_uc[seg_ui <= uid_max]

    return out[np.all(out != -1, axis=-1)].astype(np.uint32)


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

def merge_bbox_one_matrix(bbox):
    out = bbox[0].copy()
    for i in range(1, bbox.shape[0]):
        out = merge_bbox(out, bbox[i])
    return out
        

def merge_bbox_two_matrices(bbox_matrix_a, bbox_matrix_b, do_sort=True):
    """
    Merge two matrices of bounding boxes.

    Args:
        bbox_matrix_a (numpy.ndarray): The first matrix of bounding boxes.
        bbox_matrix_b (numpy.ndarray): The second matrix of bounding boxes.

    Returns:
        numpy.ndarray: The merged matrix of bounding boxes.

    Notes:
        - Each matrix should have dimensions Nx(D+1), where N is the number of bounding boxes and D is the number of dimensions (4 or 5).
        - The first column of each matrix represents the index of the bounding box.
        - The remaining columns represent the coordinates of the bounding box in the format [ymin, ymax, xmin, xmax].
        - If there are bounding boxes with the same index in both matrices, they are merged using the `merge_bbox` function.
        - Bounding boxes that do not have an intersection are concatenated in the output matrix.
    """
    if bbox_matrix_a is None:
        return bbox_matrix_b 
    if bbox_matrix_b is None:
        return bbox_matrix_a 
    bbox_a_id,  bbox_b_id = bbox_matrix_a[:, 0], bbox_matrix_b[:, 0]
    intersect_id = np.in1d(bbox_a_id, bbox_b_id)
    if intersect_id.sum() == 0:
        # no intersection
        return np.vstack([bbox_matrix_a, bbox_matrix_b])
    
    # not changing the input value
    bbox_mb = bbox_matrix_b.copy() 
    for i in np.where(intersect_id)[0]:
        bbox_a = bbox_matrix_a[i, 1:]
        bbox_b_index = bbox_b_id == bbox_a_id[i]
        bbox_b = bbox_mb[bbox_b_index, 1:][0]
        bbox_mb[bbox_b_index, 1:] = merge_bbox(bbox_a, bbox_b)
    
    out = np.vstack([bbox_matrix_a[np.logical_not(intersect_id)], bbox_mb])
    if do_sort:
        out = out[np.argsort(out[:,0])]
    return out

