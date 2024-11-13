import numpy as np
from .bbox import compute_bbox_all

def seg_to_rgb(seg):
    """
    Convert a segmentation map to an RGB image.

    Args:
        seg (numpy.ndarray): The input segmentation map.

    Returns:
        numpy.ndarray: The RGB image representation of the segmentation map.

    Notes:
        - The function converts a segmentation map to an RGB image, where each unique segment ID is assigned a unique color.
        - The RGB image is represented as a numpy array.
    """
    return np.stack([seg // 65536, seg // 256, seg % 256], axis=2).astype(
        np.uint8
    )


def rgb_to_seg(seg):
    """
    Convert an RGB image to a segmentation map.

    Args:
        seg (numpy.ndarray): The input RGB image.

    Returns:
        numpy.ndarray: The segmentation map.

    Notes:
        - The function converts an RGB image to a segmentation map, where each unique color is assigned a unique segment ID.
        - The segmentation map is represented as a numpy array.
    """
    if seg.ndim == 2:
        return seg
    elif seg.shape[-1] == 1:
        return np.squeeze(seg)
    elif seg.ndim == 3:  # 1 rgb image
        if (seg[:, :, 1] != seg[:, :, 2]).any() or (
            seg[:, :, 0] != seg[:, :, 2]
        ).any():
            return (
                seg[:, :, 0].astype(np.uint32) * 65536
                + seg[:, :, 1].astype(np.uint32) * 256
                + seg[:, :, 2].astype(np.uint32)
            )
        else:  # gray image saved into 3-channel
            return seg[:, :, 0]
    elif seg.ndim == 4:  # n rgb image
        return (
            seg[:, :, :, 0].astype(np.uint32) * 65536
            + seg[:, :, :, 1].astype(np.uint32) * 256
            + seg[:, :, :, 2].astype(np.uint32)
        )

def rgb_to_seg(seg):
    """
    Convert an RGB image to a segmentation map.

    Args:
        seg (numpy.ndarray): The input RGB image.

    Returns:
        numpy.ndarray: The segmentation map.

    Notes:
        - The function converts an RGB image to a segmentation map, where each unique color is assigned a unique segment ID.
        - The segmentation map is represented as a numpy array.
    """
    if seg.ndim == 2 or seg.shape[-1] == 1:
        return np.squeeze(seg)
    elif seg.ndim == 3:  # 1 rgb image
        if (seg[:, :, 1] != seg[:, :, 2]).any() or (
            seg[:, :, 0] != seg[:, :, 2]
        ).any():
            return (
                seg[:, :, 0].astype(np.uint32) * 65536
                + seg[:, :, 1].astype(np.uint32) * 256
                + seg[:, :, 2].astype(np.uint32)
            )
        else:  # gray image saved into 3-channel
            return seg[:, :, 0]
    elif seg.ndim == 4:  # n rgb image
        return (
            seg[:, :, :, 0].astype(np.uint32) * 65536
            + seg[:, :, :, 1].astype(np.uint32) * 256
            + seg[:, :, :, 2].astype(np.uint32)
        )

def seg_downsample_all_id(seg, ratio):
    seg_ds = seg[::ratio[0], ::ratio[1], ::ratio[2]]

    bbox = compute_bbox_all(seg)    
    id_ds = np.unique(seg_ds)
    to_add = np.in1d(bbox[:,0], id_ds, invert=True)
    if to_add.sum() != 0:
        # import pdb;pdb.set_trace()
        # some seg ids are lost        
        add_id = bbox[to_add, 0]
        add_loc = np.round(((bbox[to_add,1::2] + bbox[to_add,2::2]) /2) / ratio).astype(int)
        seg_ds[add_loc[:,0], add_loc[:,1], add_loc[:,2]] = add_id        
    return seg_ds
    
def seg_remove_id(seg, bid, invert=False):
    """
    Remove segments from a segmentation map based on their size.

    Args:
        seg (numpy.ndarray): The input segmentation map.
        bid (numpy.ndarray): The segment IDs to be removed. Defaults to None.
        invert (Boolean, optional): Invert the result.

    Returns:
        numpy.ndarray: The updated segmentation map.

    Notes:
        - The function removes segments from the segmentation map based on their size.
        - Segments with a size below the specified threshold are removed.
        - If `bid` is provided, only the specified segment IDs are removed.
    """    
    seg_m = seg.max()
    bid = np.array(bid)
    bid = bid[bid <= seg_m]
    if invert:
        rl = np.zeros(seg_m + 1).astype(seg.dtype)
        rl[bid] = bid
    else:
        rl = np.arange(seg_m + 1).astype(seg.dtype)
        rl[bid] = 0
    return rl[seg]

def seg_remove_small(seg, threshold=100, invert=False):
    uid, uc = np.unique(seg, return_counts=True)
    bid = uid[uc < threshold]
    seg = seg_remove_id(seg, bid, invert)
    return seg

def read_vast_seg(fn):
    a = open(fn).readlines()
    # remove comments
    st_id = 0
    while a[st_id][0] in ["%", "\n"]:
        st_id += 1
    # remove segment name
    out = np.zeros((len(a) - st_id, 24), dtype=int)
    name = [None] * (len(a) - st_id)
    for i in range(st_id, len(a)):
        out[i - st_id] = np.array(
            [int(x) for x in a[i][: a[i].find('"')].split(" ") if len(x) > 0]
        )
        name[i - st_id] = a[i][a[i].find('"') + 1 : a[i].rfind('"')]
    return out, name

def vast_meta_relabel(
    fn, kw_bad=["bad", "del"], kw_nm=["merge"], do_print=False
):
    # if there is meta data
    print("load meta")
    dd, nn = read_vast_seg(fn)
    rl = np.arange(1 + dd.shape[0], dtype=np.uint16)
    pid = np.unique(dd[:, 13])
    if do_print:
        print(
            ",".join(
                [
                    nn[x]
                    for x in np.where(np.in1d(dd[:, 0], pid))[0]
                    if "Imported Segment" not in nn[x]
                ]
            )
        )

    pid_b = []
    if len(kw_bad) > 0:
        # delete seg id
        pid_b = [
            i
            for i, x in enumerate(nn)
            if max([y in x.lower() for y in kw_bad])
        ]
        bid = np.where(np.in1d(dd[:, 13], pid_b))[0]
        bid = np.hstack([pid_b, bid])
        if len(bid) > 0:
            rl[bid] = 0
        print("found %d bad" % (len(bid)))

    # not to merge
    kw_nm += ["background"]
    pid_nm = [
        i for i, x in enumerate(nn) if max([y in x.lower() for y in kw_nm])
    ]
    # pid: all are children of background
    pid_nm = np.hstack([pid_nm, pid_b])
    print("apply meta")
    # hierarchical
    for p in pid[np.in1d(pid, pid_nm, invert=True)]:
        rl[dd[dd[:, 13] == p, 0]] = p
    # consolidate root labels
    for u in np.unique(rl[np.where(rl[rl] != rl)[0]]):
        u0 = u
        while u0 != rl[u0]:
            rl[rl == u0] = rl[u0]
            u0 = rl[u0]
        print(u, rl)
    return rl        
