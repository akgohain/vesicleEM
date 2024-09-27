import sys
import numpy as np
from util import *
import statistics


def vesicle_crop(im, ves, ves_label=None, sz=[5,31,31], sz_thres=5):        
        sz = np.array(sz)
        szh = sz//2
        out_im = np.zeros([0] + list(sz), np.uint8)
        out_mask = np.zeros([0] + list(sz), np.uint8)
        tmp = np.zeros(sz, np.uint8)
        out_l = []
        bbs = compute_bbox_all(ves)
        for bb in bbs:
            bsz = (bb[2::2]-bb[1::2])+1
            # remove too small ones
            if bb[1:].min() > sz_thres:
                if ves_label is not None:
                    tmp = ves_label[ves==bb[0]]
                    # ideally, tmp>0, but some VAST modification
                    ll = statistics.mode(tmp[tmp>0])
                    out_l.append(ll)
                cc = (bb[1::2]+bb[2::2])//2
                crop = im[max(0,cc[0]-szh[0]):cc[0]+szh[0], max(0,cc[1]-szh[1]):cc[1]+szh[1], max(0,cc[2]-szh[2]):cc[2]+szh[2]]
                diff = (sz - crop.shape) // 2
                # much blank                    
                diff2 = sz - crop.shape -diff
                tmp = np.pad(crop, [(diff[0],diff2[0]), (diff[1],diff2[1]), (diff[2],diff2[2])], 'edge')
                # pad: xy-edge, z-reflect
                out_im = np.concatenate([out_im, tmp[None]], axis=0)
                tmp[:] = 0

                crop = ves[max(0,cc[0]-szh[0]):cc[0]+szh[0], max(0,cc[1]-szh[1]):cc[1]+szh[1], max(0,cc[2]-szh[2]):cc[2]+szh[2]]==bb[0]                    
                tmp = np.pad(crop, [(diff[0],diff2[0]), (diff[1],diff2[1]), (diff[2],diff2[2])], 'edge')
                out_mask = np.concatenate([out_mask, tmp[None]], axis=0)
                tmp[:] = 0
        if ves_label is None:
            return out_im, out_mask
        else:
            return out_im, out_mask, out_l

if __name__ == "__main__":
    opt = sys.argv[1]
    if opt == '0':
        fn = 'tile_0-188/6-10.h5'
        im = read_h5(f'/data/projects/weilab/dataset/hydra/im_chunk/{fn}')
        ves = read_h5(f'/data/projects/weilab/hydra/2024_09_13{fn}')
        out = vesicle_crop(im, ves)