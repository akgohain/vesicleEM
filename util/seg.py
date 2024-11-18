import os
import numpy as np
import h5py
from .bbox import compute_bbox_all_chunk, merge_bbox_one_matrix, compute_bbox_all
from .arr import UnionFind
from .io import vol_downsample_chunk,read_h5,write_h5
import cc3d


def seg_add_chunk(input_file, chunk_num=1, add_loc=None, add_val=None, seg_file=None, seg_remove=None):
    fid = h5py.File(input_file, 'r+')
    vol = fid[list(fid)[0]]     
    num_z = int(np.ceil(vol.shape[0] / float(chunk_num)))

    if seg_file is not None:
        fid_seg = h5py.File(seg_file, 'r')
        seg = fid_seg[list(fid_seg)[0]]
    
    for i in range(chunk_num): 
        vol_chunk = np.array(vol[i*num_z:(i+1)*num_z])
        if add_loc == 'all':
            vol_chunk[vol_chunk>0] += add_val
        elif isinstance(add_loc, np.ndarray):
            vol_chunk[add_loc[:,0], add_loc[:,1], add_loc[:,2]] = add_val
            
        if seg_file is not None:
            vol_seg = np.array(seg[i*num_z:(i+1)*num_z])
            if seg_remove is not None:
                vol_seg = seg_remove_id(vol_seg, seg_remove)
            vol[vol_seg>0] += vol_seg[vol_seg>0]
    if seg_file is not None:
        fid_seg.close()
         
    fid.close()


def seg_cc_chunk(seg_file, output_file, dt=np.uint16, \
    seg_func=None, chunk_num=1, dust_size=0):
    # first pass: compute the relabel with union find
    max_id = 0
    last_slice = []
    relabel = []    
    fid_seg = h5py.File(seg_file, 'r')
    seg = fid_seg[list(fid_seg)[0]]
    num_z = int(np.ceil(seg.shape[0] / float(chunk_num)))
     
    fid = h5py.File(output_file, 'w')
    out = fid.create_dataset('main', seg.shape, dt)
    for i in range(chunk_num):
        vol = np.array(seg[i*num_z:(i+1)*num_z])
        if seg_func is not None:
            vol = seg_func(vol)
        vol_cc = cc3d.connected_components(vol, connectivity=6)
        mm = vol_cc.max()
        vol_cc[vol_cc>0] += max_id
        out[i*num_z:(i+1)*num_z] = vol_cc        
        if i == 0:
            bb = compute_bbox_all(vol_cc, True)            
            relabel = UnionFind(bb[:,0])
        else:
            bb_chunk = compute_bbox_all(vol_cc, True)
            relabel.add_arr(bb_chunk[:,0])
            # find merge pairs
            id1 = last_slice[last_slice>0].reshape(-1,1)
            id2 = vol_cc[0][last_slice>0].reshape(-1,1)
            to_merge = np.unique(np.hstack([id1, id2]), axis=0)
            relabel.union_arr(to_merge)
            bb = np.vstack([bb, bb_chunk])
        last_slice = vol_cc[-1]
        max_id += mm    
    
    # compute relabel array
    relabel_arr = np.zeros(max_id+1, dt)
    to_merge = [list(x) for x in relabel.components() if len(x)>1]
    for component in to_merge:
        cid = min(component)
        relabel_arr[component] = cid
        all_id = np.in1d(bb[:,0], component)
        merge_id = bb[:,0]==cid        
        bb[merge_id, 1:] = merge_bbox_one_matrix(bb[all_id, 1:])
        bb[all_id, 0] = 0
        bb[merge_id, 0] = cid
    bb = bb[bb[:,0] != 0]
        
    if dust_size > 0:
        relabel[np.in1d(relabel, bb[bb[:,-1] <= dust_size, 0])] = 0        
    
    # second pass: apply relabel
    for i in range(chunk_num): 
        out[i*num_z:(i+1)*num_z] = relabel[np.array(out[i*num_z:(i+1)*num_z])]
        
    fid.close()
    fid_seg.close()

def seg_unique_id_chunk(input_file, chunk_num=1):
    if chunk_num == 1:
        return np.unique(read_h5(input_file))
    else:        
        uid = []
        fid = h5py.File(input_file, 'r')
        seg = fid[list(fid)[0]]
        num_z = int(np.ceil(seg.shape[0] / float(chunk_num)))
        for i in range(chunk_num):
            if i == 0:
                uid = np.unique(np.array(seg[i*num_z:(i+1)*num_z]))
            else:
                uid = np.hstack([uid, np.unique(np.array(seg[i*num_z:(i+1)*num_z]))])
                uid = np.unique(uid)
    return uid
                

def seg_downsample_chunk(input_file, ratio, output_file=None, chunk_num=1):
    # preserve all seg id 
    # o/w for regular downsample: vol_downsample_chunk(input_file, ratio, output_file, chunk_num) 
    if output_file is not None and os.path.exists(output_file):
        print('File exists:', output_file)
        return None
    if output_file is None or chunk_num==1:
        seg = read_h5(input_file)
        bbox = compute_bbox_all(seg)
        seg_ds = seg[::ratio[0], ::ratio[1], ::ratio[2]]
        id_ds = np.unique(seg_ds)
        to_add = np.in1d(bbox[:,0], id_ds, invert=True)
        if to_add.sum() != 0:
            # some seg ids are lost        
            add_id = bbox[to_add, 0]
            add_loc = np.round(((bbox[to_add,1::2] + bbox[to_add,2::2]) /2) / ratio).astype(int)
            seg_ds[add_loc[:,0], add_loc[:,1], add_loc[:,2]] = add_id        
        if output_file is None:
            return seg_ds
        else:
            write_h5(output_file, seg_ds)
    else:
        vol_downsample_chunk(input_file, ratio, output_file, chunk_num)
        # process in chunks
        bbox = compute_bbox_all_chunk(input_file, chunk_num=chunk_num)
        id_ds = seg_unique_id_chunk(output_file, chunk_num)
        to_add = np.in1d(bbox[:,0], id_ds, invert=True)
        if to_add.sum() != 0:
            # some seg ids are lost        
            add_id = bbox[to_add, 0]
            add_loc = np.round(((bbox[to_add,1::2] + bbox[to_add,2::2]) /2) / ratio).astype(int)
            seg_add_chunk(output_file, chunk_num, add_loc, add_id)            
    
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
