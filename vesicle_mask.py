import os, sys
from util import *
import numpy as np
from neuron_mask import neuron_id_to_neuron, neuron_id_to_bbox, neuron_to_id_name
import statistics
import cc3d

def crop_to_tile(conf, opt='big', job_id=0, job_num=1):
    pad_z, pad_y, pad_x = conf['vesicle_pad']
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

def vesicle_instance_crop(ves, im=None, ves_label=None, sz=[5,31,31], sz_thres=5):        
    sz = np.array(sz)
    szh = sz//2
    out_im = np.zeros([0] + list(sz), np.uint8)
    out_mask = np.zeros([0] + list(sz), np.uint8)
    tmp = np.zeros(sz, np.uint8)
    out_l = []
    bbs = compute_bbox_all(ves)
    for bb in bbs:        
        # remove small ones
        if bb[1:].min() > sz_thres:
            if ves_label is not None:
                tmp = ves_label[ves==bb[0]]
                # ideally, tmp>0, but some VAST modification
                ll = statistics.mode(tmp[tmp>0])
                out_l.append(ll)
            cc = (bb[1::2] + bb[2::2]) // 2
            crop = ves[max(0,cc[0]-szh[0]):cc[0]+szh[0], max(0,cc[1]-szh[1]):cc[1]+szh[1], max(0,cc[2]-szh[2]):cc[2]+szh[2]]==bb[0]                    
            pad_left = (sz - crop.shape) // 2               
            pad_right = sz - crop.shape - pad_left
            # pad: xy-edge, z-reflect
            tmp = np.pad(crop, [(pad_left[0],pad_right[0]), (pad_left[1],pad_right[1]), (pad_left[2],pad_right[2])], 'edge')
            out_mask = np.concatenate([out_mask, tmp[None]], axis=0)
            tmp[:] = 0
            
            if im is not None:
                crop = im[max(0,cc[0]-szh[0]):cc[0]+szh[0], max(0,cc[1]-szh[1]):cc[1]+szh[1], max(0,cc[2]-szh[2]):cc[2]+szh[2]]
                tmp = np.pad(crop, [(pad_left[0],pad_right[0]), (pad_left[1],pad_right[1]), (pad_left[2],pad_right[2])], 'edge')                
                out_im = np.concatenate([out_im, tmp[None]], axis=0)
                tmp[:] = 0
    if ves_label is None:        
        return out_im, out_mask
    else:
        return out_im, out_mask, out_l
    

def neuron_id_to_vesicle(conf, neuron_id, ratio=[1,4,4], opt='big', output_file=None, neuron_file=None):
    bb = neuron_id_to_bbox(conf, neuron_id)
    # prediction mip1: [30, 8, 8]
    # target: [30,32,32]
    def h5Name(z,y,x):
        z0, z1 = conf['vesicle_zchunk'][z]
        return conf[f'vesicle_chunk_path_{opt}'].format(z0, z1) % (y, x)
    
    zyx_sz = [100, 4096// ratio[1], 4096// ratio[2]]
    bb[:2] = bb[:2] // ratio[0]
    bb[2:4] = bb[2:4] // ratio[1]
    bb[4:] = bb[4:] // ratio[2]

    mask = None
    if neuron_file is not None:
        mask = h5py.File(neuron_file,'r')['main']
    out = read_tile_h5_by_bbox(h5Name, bb[0], bb[1]+1, bb[2], bb[3]+1, bb[4], bb[5]+1, \
                       zyx_sz, zz=[88,40,15], tile_type='seg', tile_step=ratio[1:], zstep=ratio[0], \
                        acc_id=True, output_file=output_file, mask=mask)
    return out

def vesicle_vast_process(seg_file, meta_file, dust_size=50):    
    _, meta_n = read_vast_seg(meta_file)
    relabel = vast_meta_relabel(meta_file)
    ves = relabel[read_h5(seg_file)]
    
    sv_id = [i for i,x in enumerate(meta_n) if x=='SV']    
    out_sv = cc3d.connected_components(ves==sv_id, connectivity=6)
        
    max_id = ves.max()
    lv_id = [i for i,x in enumerate(meta_n) if x=='LV']
    out_lv = cc3d.connected_components(ves==lv_id, connectivity=6)
    # remove small ones
    out_lv = seg_remove_small(out_lv, dust_size)    
    out_lv[out_lv > 0] += max_id
    ves[ves==sv_id] = 0
    ves[ves==lv_id] = 0
    out_lv[ves > 0] = ves[ves > 0]
        
    return out_sv, out_lv
   
    
if __name__ == "__main__":
    conf = read_yml('conf/param.yml')
    args = get_arguments()
    if args.output_folder == '':
        args.output_folder = conf['result_folder']
    if args.task == '0':        
        # convert deep learning prediction into 100x4096x4096 chunks
        # python vesicle_mask.py 0 big 0 1
        job_id, job_num = int(sys.argv[3]), int(sys.argv[4])
        crop_to_tile(conf, vesicle, job_id, job_num)
    elif args.task == 'neuron-vesicle':
        # return the vesicle prediction within the neuron bounding box
        # python vesicle_mask.py -t neuron-vesicle -n 5
        for neuron in args.neuron:
            neuron_id, neuron_name = neuron_to_id_name(conf, neuron)
            # zip -r vesicle_big_5_30-8-8.zip vesicle_big_5_30-8-8
            suff = arr_to_str(np.array(args.ratio) * conf['res'])
            output_file = None
            if args.ratio[1] == 1:
                output_file = f'{conf["result_folder"]}/vesicle_{args.vesicle}_{neuron_name}_{suff}/%04d.png'
                mkdir(output_file, 'parent')
            neuron_file = f'{conf["result_folder"]}/neuron_{neuron_name}_{suff}.h5' 
            seg = neuron_id_to_vesicle(conf, neuron_id, args.ratio, args.vesicle, output_file, neuron_file)        
            if args.ratio[1] != 1:
                write_h5(output_file, seg)

    elif args.task == 'vast-process':
        # python vesicle_mask.py -t vast-process -ir /data/projects/weilab/dataset/hydra/vesicle_pf/ -i KR4_8nm.h5,VAST_segmentation_metadata_KR4.txt -o sv_KR4,lv_KR4 -r 1,4,4
        seg_file, meta_file = [os.path.join(args.input_folder, x) for x in args.input_file.split(',')]
        sv_file, lv_file = [os.path.join(args.output_folder, x) for x in args.output_file.split(',')]
        suffix = arr_to_str(conf['res'])
        sv_file = f'{sv_file}_{suffix}.h5'
        lv_file = f'{lv_file}_{suffix}.h5'
        
        out_sv, out_lv = vesicle_vast_process(seg_file, meta_file)
        write_h5(sv_file, out_sv)
        write_h5(lv_file, out_lv)        
        if max(args.ratio) != 1:
            # large vesicle direct downsample
            out_lv = seg_downsample_all_id(out_lv, args.ratio)
            out_sv = seg_downsample_all_id(out_sv, args.ratio)            
            suffix2 = arr_to_str(np.array(args.ratio)*conf['res'])
            write_h5(sv_file.replace(suffix, suffix2), out_sv)
            write_h5(lv_file.replace(suffix, suffix2), out_lv)
        
    elif args.task == 'chunk':
        # python vesicle_crop.py chunk tile_0-188/6-10.h5
        chunk_name = sys.argv[2]
        im = read_h5(f'/data/projects/weilab/dataset/hydra/im_chunk/{chunk_name}')
        ves = read_h5(f'/data/projects/weilab/hydra/2024_09_13{chunk_name}')
        out = vesicle_crop(im, ves)