from util import *
import numpy as np
Dh = '/data/projects/weilab/dataset/hydra/'

def get_neuron_mask(seg_id):
    rl = vast_meta_relabel(f'{Dh}/mask_mip1/meta.txt').astype(np.uint8)
    # read in the bounding box
    bbox = np.loadtxt(f'{Dh}/mask_mip1/bbox.txt').astype(int)

    # find the bounding box of the input id
    bb = bbox[bbox[:,0] == seg_id, 1:7][0]
    print(f'Neuron {seg_id} bbox: {bb}')

    filenames = [f'/data/projects/weilab/dataset/hydra/mask_mip1/png/Neuron_1_240409_ENDO_no bridge_no ecto.vsseg_export_s{z:04d}_Y%d_X%d.png' for z in range(1830)]

    tile_sz = [8192,8192]
    tile_st = [1,1]
    out = read_tile_volume(filenames, bb[0], bb[1], bb[2], bb[3], bb[4], bb[5], tile_sz, tile_st, tile_type="seg")
    out = rl[out] == seg_id
    return out

if __name__ == "__main__":
    seg_id = 1
    seg = get_neuron_mask(seg_id)
