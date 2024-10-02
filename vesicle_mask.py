import os, sys
from util import *
import numpy as np
from neuron_mask import segid_to_bbox 

def segid_to_vesicle(conf, seg_id, ratio=[1,4,4], opt='big'):
    bb = segid_to_bbox(conf, seg_id)
    # prediction mip1: [30, 8, 8]
    # target: [30,32,32]
    def h5Name(z,y,x):
        z0, z1 = conf['vesicle_zchunk'][z]
        return conf['vesicle_chunk_path_big'].format(z0, z1) % (y, x)
    
    zyx_sz = [100, 4096// ratio[1], 4096// ratio[2]]
    bb[:2] = bb[:2] // ratio[0]
    bb[2:4] = bb[2:4] // ratio[1]
    bb[4:] = bb[4:] // ratio[2]

    out = read_tile_h5(h5Name, bb[0], bb[1], bb[2], bb[3], bb[4], bb[5], 
                       zyx_sz, zz=[88,40,15], tile_step=ratio[1:], zstep=ratio[0], acc_id=True)
    
    return out
    

def crop_to_chunk(conf, opt='big', job_id=0, job_num=1):
    tsz = 256
    pad_z, pad_y, pad_x = conf['vesicle_pad']    
    fn = '%d_%d_%d.png'

    for zz in conf['vesicle_zchunk'][job_id::job_num]:
        input_name = conf['vesicle_pred_path_' + opt].format(zz[0], zz[1])
        output_name = conf['vesicle_chunk_path_' + opt].format(zz[0], zz[1])
        meta = np.loadtxt(conf['vesicle_zchunk_meta'].format(zz[0], zz[1])).astype(int)    
        
        mkdir(output_name, 'parent')
        
        out = np.zeros([zz[1]-zz[0],4096,4096], np.uint16)
        for rc in meta[:, :2]:
            fin = input_name % (rc[0], rc[1])
            fout = output_name % (rc[0], rc[1])
            if not os.path.exists(fin):
                print('missing prediction:', fin)
                continue
            if not os.path.exists(fout):                
                out[:] = 0
                # 8x8x30 nm
                seg = read_h5(fin)
                if (np.array(out.shape) - seg.shape).max() == 0:
                    out = seg
                else:
                    print('working on:', fout, seg.shape)                
                    # crop: 128x128x120 nm               
                    bb = meta[(meta[:,0]==rc[0])*(meta[:,1]==rc[1])][0]
                    z0, z1 = bb[2]-pad_z//4, bb[3]+pad_z//4
                    y0, y1 = bb[4]-pad_y//16, bb[5]+pad_y//16
                    x0, x1 = bb[6]-pad_x//16, bb[7]+pad_x//16
                    out_p = out[max(0,z0)*4:z1*4, max(0,y0)*16:y1*16, max(0,x0)*16:x1*16]

                    # initial offset
                    ## z: not adding extra slices
                    ## xy: use neighboring tiles
                    oset = [0, max(0,-y0), max(0,-x0)]                
                    out_p[:]= seg[oset[0]*4:oset[0]*4+out_p.shape[0],\
                                    oset[1]*16:oset[1]*16+out_p.shape[1],\
                                    oset[2]*16:oset[2]*16+out_p.shape[2]]

                write_h5(fout, out)

if __name__ == "__main__":
    opt = sys.argv[1]
    vesicle = sys.argv[2]
    conf = read_yml('conf/param.yml')
    if opt == '0':        
        # convert deep learning prediction into 100x4096x4096 chunks
        # python vesicle_mask.py 0 big 0 1
        job_id, job_num = int(sys.argv[3]), int(sys.argv[4])
        crop_to_chunk(conf, vesicle, job_id, job_num)
    elif opt == '1':
        # return the vesicle prediction within the neuron bounding box
        # python vesicle_mask.py 1 big 16
        seg_id = int(sys.argv[3])
        ratio = [1,1,1]
        seg = segid_to_vesicle(conf, seg_id, ratio, sys.argv[2])
        if ratio[1] == 1: # save into png for vast
            sn = f'{conf["result_folder"]}/vesicle_{sys.argv[2]}_{seg_id}_{ratio[0]*30}-{ratio[1]*8}-{ratio[2]*8}/'
            mkdir(sn)
            for z in range(seg.shape[0]):
                write_image(f'{sn}seg_{z:04d}.png', seg[z], 'seg')
        else:# save into h5
            write_h5(f'{conf["result_folder"]}/vesicle_{sys.argv[2]}_{seg_id}_{ratio[0]*30}-{ratio[1]*8}-{ratio[2]*8}.h5', seg)
