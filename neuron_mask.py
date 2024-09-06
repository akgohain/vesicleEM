import sys
from util import *
import numpy as np
from glob import glob

def compute_bbox_tile(conf, job_id, job_num):
    fns = read_txt(conf['mask_filenames'])
    tsz = conf['mask_size'][1:]
    seg_relabel = vast_meta_relabel(conf['mask_meta'])
    bbox_folder = conf['mask_bbox_folder']
    mask_folder = conf['mask_folder'] 
    
    for fn in fns[job_id::job_num]:
        fn = fn[:-1]
        sn = f'{bbox_folder}/{fn}.txt'
        if not os.path.exists(sn):
            im = seg_relabel[rgb_to_seg(read_image(f'{mask_folder}/{fn}.png'))]
            bb = ''
            if im.max() > 0:
                # XY: 1-index
                zz = int(fn[fn.rfind('s')+1:fn.rfind('Y')-1])
                yy = int(fn[fn.rfind('Y')+1:fn.rfind('X')-1])
                xx = int(fn[fn.rfind('X')+1:])
                bb = compute_bbox_all(im)
                bb = np.hstack([bb[:,:1], np.ones([bb.shape[0], 2], int) * zz, bb[:,1:3] + yy*tsz, bb[:,3:5]+xx*tsz])
                np.savetxt(sn, bb, '%d')
            else:
                write_txt(sn, bb)

def merge_bbox_tile(conf):
    fns = read_txt(conf['mask_filenames'])
    bbox_folder = conf['mask_bbox_folder']
    out = np.loadtxt(f'{bbox_folder}/{fns[0][:-1]}.txt').astype(int)
    for fn in fns:
        fn = fn[:-1]
        bbox = np.loadtxt(f'{bbox_folder}/{fn}.txt').astype(int)
        if bbox.shape[0] > 0:
            if bbox.ndim == 1:
                bbox = bbox.reshape(1,-1)
            out = merge_bbox_two_matrices(out, bbox)
    np.savetxt(f'{bbox_folder}/out.txt', out, '%d')

def segid_to_bbox(conf, seg_id):
    bbox = np.loadtxt(f"{conf['mask_bbox_folder']}/out.txt").astype(int)
    # find the bounding box of the input id
    bb = bbox[bbox[:,0] == seg_id, 1:7][0]
    print(f'Neuron {seg_id} bbox: {bb}')
    return bb

def segid_to_neuron(conf, seg_id):    
    # read in the bounding box
    bb = segid_to_bbox(conf, seg_id)
    
    # mip1: [30, 8, 8]
    filenames = [conf['mask_folder'] + conf['mask_template'].format(z) for z in range(conf['mask_size'][0])]
    tile_st = [0,0]   
    tile_sz = 8192
        
    # target: [30, 32, 32]
    zstep, xystep = 1, 4
    tile_sz = tile_sz // xystep
    bb[2:] = bb[2:] // xystep
        
    out = read_tile_volume(filenames, bb[0], bb[1], bb[2], bb[3], bb[4], bb[5], 
                           tile_sz, tile_st, tile_type="seg", tile_ratio=1. / xystep, zstep=zstep)
    
    rl = vast_meta_relabel(conf['mask_meta']).astype(np.uint8)
    out = (rl[out] == seg_id).astype(np.uint8)
    return out

if __name__ == "__main__":
    opt = sys.argv[1]
    conf = read_yml('param.yml')
    if opt[0] == '0':
        # convert VAST export segments into bbox
        if opt == '0':
            # find filenames
            get_filenames(conf['mask_folder'], 'out.txt')
        elif opt == '0.1':
            # compute bbox for each tile
            job_id, job_num = int(sys.argv[2]), int(sys.argv[3])
            compute_bbox_tile(conf, job_id, job_num)
        elif opt == '0.2':
            # merge all bbbox
            merge_bbox_tile(conf)
    elif opt == '1':
        seg_id = int(sys.argv[2])
        seg = segid_to_neuron(conf, seg_id)
        write_h5(f'results/neuron_{seg_id}.h5', seg)
