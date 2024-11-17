import os
from glob import glob
from util.io import *
from util.tile import *
from util.seg import *
from util.bbox import *
import numpy as np
from neuron_mask import neuron_id_to_bbox, neuron_to_id_name
import statistics
import cc3d

def crop_to_tile(vol, conf, opt, zz, rc):
    meta = np.loadtxt(conf['vesicle_zchunk_meta'].format(zz[0], zz[1])).astype(int)
    pad_z, pad_y, pad_x = conf['vesicle_pad']    
    out_size = [zz[1]-zz[0]] + conf['vesicle_chunk_size']
    if (np.array(out_size) - vol.shape).max() == 0:
        return vol
    else:
        dt = np.uint16 if opt != 'im' else np.uint8    
        out = np.zeros(out_size, dt)
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
        out_p[:]= vol[oset[0]*4:oset[0]*4+out_p.shape[0],\
                        oset[1]*16:oset[1]*16+out_p.shape[1],\
                        oset[2]*16:oset[2]*16+out_p.shape[2]]
        return out

def crop_to_tile_all(conf, opt='big', job_id=0, job_num=1):
    for zz in conf['vesicle_zchunk'][job_id::job_num]:
        input_name = conf['vesicle_pred_path_' + opt].format(zz[0], zz[1])
        output_name = conf['vesicle_chunk_path_' + opt].format(zz[0], zz[1])
        mkdir(output_name, 'parent')
        meta = np.loadtxt(conf['vesicle_zchunk_meta'].format(zz[0], zz[1])).astype(int)    
        for rc in meta[:, :2]:            
            if not os.path.exists(input_name):
                print('missing prediction:', input_name)
                return None
            fin = input_name % (rc[0], rc[1])
            fout = output_name % (rc[0], rc[1])
            if not os.path.exists(fout):                
                vol = read_h5(fin)
                out = crop_to_tile(vol, conf, opt, zz, rc)                
                write_h5(fout, out)

def vesicle_instance_crop(ves, im=None, ves_label=None, sz=[5,31,31], sz_thres=5, chunk_num=1):
    sz = np.array(sz)
    szh = sz//2
    out_im = np.zeros([0] + list(sz), np.uint8)
    out_mask = np.zeros([0] + list(sz), np.uint8)
    tmp = np.zeros(sz, np.uint8)
    out_l = []
    if chunk_num == 1:
        bbs = compute_bbox_all(np.array(ves))
    else:
        bbs = compute_bbox_all_chunk(ves, chunk_num=chunk_num)
    print('# instances:', len(bbs))
    for bb in bbs:        
        # remove small xy size
        if bb[3:].min() > sz_thres:
            if ves_label is not None:
                tmp = ves_label[ves==bb[0]]
                # ideally, tmp>0, but some VAST modification
                ll = statistics.mode(tmp[tmp>0])
                out_l.append(ll)
            cc = (bb[1::2] + bb[2::2]) // 2
            crop = np.array(ves[max(0,cc[0]-szh[0]):cc[0]+szh[0]+1, \
                                max(0,cc[1]-szh[1]):cc[1]+szh[1]+1, \
                                max(0,cc[2]-szh[2]):cc[2]+szh[2]+1])==bb[0]
            pad_left = (sz - crop.shape) // 2               
            pad_right = sz - crop.shape - pad_left
            # pad: xy-edge, z-reflect
            import pdb;pdb.set_trace()
            tmp = np.pad(crop, [(pad_left[0],pad_right[0]), (pad_left[1],pad_right[1]), (pad_left[2],pad_right[2])], 'edge')
            out_mask = np.concatenate([out_mask, tmp[None]], axis=0)
            tmp[:] = 0
            
            if im is not None:
                crop = np.array(im[max(0,cc[0]-szh[0]):cc[0]+szh[0]+1, \
                                   max(0,cc[1]-szh[1]):cc[1]+szh[1]+1, \
                                   max(0,cc[2]-szh[2]):cc[2]+szh[2]+1])
                tmp = np.pad(crop, [(pad_left[0],pad_right[0]), (pad_left[1],pad_right[1]), (pad_left[2],pad_right[2])], 'edge')                
                out_im = np.concatenate([out_im, tmp[None]], axis=0)
                tmp[:] = 0
    if ves_label is None:        
        if im is None:
            return out_mask
        return [out_im, out_mask]
    else:
        if im is None:
            return [out_mask, out_l]
        return [out_im, out_mask, out_l]
    

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

    acc_id, tile_type, h5_func = True, 'seg', None
    if opt == 'im':
        acc_id, tile_type = False, 'image'
        def h5_func(vol, z, y, x):            
            return crop_to_tile(vol, conf, opt, conf['vesicle_zchunk'][z], [y,x])

    mask_data = None
    if neuron_file is not None:
        mask_file = h5py.File(neuron_file,'r')
        mask_data = mask_file['main']
    out = read_tile_h5_by_bbox(h5Name, bb[0], bb[1]+1, bb[2], bb[3]+1, bb[4], bb[5]+1, \
                       zyx_sz, zz=[88,40,15], tile_type=tile_type, tile_step=ratio[1:], zstep=ratio[0], \
                        acc_id=acc_id, output_file=output_file, mask=mask_data, h5_func=h5_func)
    if neuron_file is not None:
        mask_file.close()
    return out

def vesicle_vast_process(seg_file, meta_file, dust_size=50, do_lv=False, do_sv=False):
    out_sv, out_lv = None, None
    if do_lv or do_sv:
        _, meta_n = read_vast_seg(meta_file)
        relabel = vast_meta_relabel(meta_file)
        ves = relabel[read_h5(seg_file)]
         
        if do_sv:
            sv_id = [i for i,x in enumerate(meta_n) if x=='SV']    
            # connected component on each slice
            out_sv = np.zeros(ves.shape, np.uint16)
            max_id = 0
            for z in range(ves.shape[0]):
                if (ves[z]==sv_id).any():
                    out_sv[z] = cc3d.connected_components(ves[z]==sv_id, connectivity=4)
                    mm = out_sv[z].max()
                    out_sv[z][out_sv[z] > 0] += max_id
                    max_id += mm
                
        if do_lv:   
            max_id = ves.max()
            lv_id = [i for i,x in enumerate(meta_n) if x=='LV']
            out_lv = cc3d.connected_components(ves==lv_id, connectivity=6)
            # remove small ones
            out_lv = seg_remove_small(out_lv, dust_size)    
            out_lv[out_lv > 0] += max_id
            ves[ves==sv_id] = 0
            ves[ves==lv_id] = 0
            out_lv[ves > 0] = ves[ves > 0]
            #import pdb;pdb.set_trace() 
            
    return out_sv, out_lv
    
    
if __name__ == "__main__":
    conf = read_yml('conf/param.yml')
    args = get_arguments()
    if args.output_folder == '':
        args.output_folder = conf['result_folder']
    if args.task == 'tile-vesicle':        
        # convert deep learning prediction into 100x4096x4096 chunks
        # python vesicle_mask.py -t tile-vesicle -i /data/projects/weilab/hydra/2024_10_01/tile_188-288/7-12_pred.h5 -o 7-12_pred.h5
        vol = read_h5(args.input_file)
        fn = args.input_file[args.input_file.rfind('tile_'):]
        zz = [int(x) for x in fn[5:fn.find('/')].split('-')]
        rc = [int(x) for x in fn[fn.find('/')+1:fn.rfind('_')].split('-')]
        out = crop_to_tile(vol, conf, args.vesicle, zz, rc)
        write_h5(args.output_file, out)
    elif args.task == 'neuron-vesicle':
        # return the vesicle prediction within the neuron bounding box
        # python vesicle_mask.py -t neuron-vesicle -n 37,38,39 -v im -p "file_type:h5"
        for neuron in args.neuron:
            neuron_id, neuron_name = neuron_to_id_name(conf, neuron)
            # zip -r vesicle_big_5_30-8-8.zip vesicle_big_5_30-8-8
            suff = arr_to_str(np.array(args.ratio) * conf['res'])
            output_file = f'{conf["result_folder"]}/vesicle_{args.vesicle}_{neuron_name}_{suff}'
            if 'file_type' in args.param and args.param['file_type']=='h5':
                output_file = output_file + '.h5'
                if os.path.exists(output_file):
                    continue
            else:                
                mkdir(output_file)
                output_file = output_file + '/%04d.png'            
                fns = glob(output_file + '/*.png')
                if len(fns) != 0:
                    continue
            neuron_file = f'{conf["result_folder"]}/neuron_{neuron_name}_{suff}.h5'
            
            seg = neuron_id_to_vesicle(conf, neuron_id, args.ratio, args.vesicle, output_file, neuron_file)

    elif args.task == 'neuron-vesicle-proofread':
        # python vesicle_mask.py -t neuron-vesicle-proofread -ir /data/projects/weilab/dataset/hydra/vesicle_pf/ -i KR4_8nm.h5,VAST_segmentation_metadata_KR4.txt -o sv_KR4,lv_KR4 -r 1,4,4
        seg_file, meta_file = [os.path.join(args.input_folder, x) for x in args.input_file.split(',')]
        sv_file, lv_file = [os.path.join(args.output_folder, x) for x in args.output_file.split(',')]
        suffix = arr_to_str(conf['res'])        
        sv_file = f'{sv_file}_{suffix}.h5'
        lv_file = f'{lv_file}_{suffix}.h5'
                
        out_sv, out_lv = vesicle_vast_process(seg_file, meta_file, \
                        do_lv=not os.path.exists(lv_file), do_sv=not os.path.exists(sv_file))
        if out_sv is not None:
            write_h5(sv_file, out_sv)
        if out_lv is not None:
            write_h5(lv_file, out_lv)        
        
        if max(args.ratio) != 1:
            # large vesicle direct downsample
            suffix2 = arr_to_str(np.array(args.ratio)*conf['res'])
            if not os.path.exists(lv_file.replace(suffix, suffix2)):
                sv_file2 = sv_file.replace(suffix, suffix2)
                lv_file2 = lv_file.replace(suffix, suffix2)
                if not os.path.exists(sv_file2):
                    out_sv = seg_downsample_all_id(out_sv, args.ratio)
                    write_h5(sv_file2, out_sv)
                if not os.path.exists(lv_file2): 
                    out_lv = seg_downsample_all_id(out_lv, args.ratio)
                    write_h5(lv_file2, out_lv)
        
    elif args.task == 'neuron-vesicle-patch':
        # python vesicle_mask.py -t neuron-vesicle-patch -ir /data/projects/weilab/dataset/hydra/results/ -i lv_KR6_30-8-8.h5,vesicle_im_KR6_30-8-8.h5
        # python vesicle_mask.py -t neuron-vesicle-patch -ir /data/projects/weilab/dataset/hydra/results/ -i sv_KR6_30-8-8.h5,vesicle_im_KR6_30-8-8.h5 -v small -p "chunk_num:5"
        im = None
        if ',' in args.input_file:
            ves_file, im_file = args.input_file.split(',') 
            im = h5py.File(os.path.join(args.input_folder, im_file), 'r')['main']
        else:
            ves_file = args.input_file
        ves = h5py.File(os.path.join(args.input_folder, ves_file), 'r')['main']
        
        # import pdb;pdb.set_trace() 
        patch_sz = [5,31,31] if args.vesicle=='big' else [1,11,11]
        chunk_num = 1 if 'chunk_num' not in args.param else args.param['chunk_num']
        out = vesicle_instance_crop(ves, im, sz=patch_sz, sz_thres=0, chunk_num=chunk_num)
        output_file = args.output_file
        if output_file == '':
            output_file = os.path.join(args.input_folder, ves_file.replace('.h5', '_patch.h5'))
        
        write_h5(output_file, out)