import os,sys
import glob
from util import *
import numpy as np
from neuron_mask import neuron_id_to_bbox, neuron_to_id_name
import statistics
#import cc3d

def crop_to_tile(vol, conf, opt, zz, rc):
    meta = np.loadtxt(conf['vesicle_zchunk_meta'].format(zz[0], zz[1])).astype(int)
    pad_z, pad_y, pad_x = conf['vesicle_pad']    
    out_size = [zz[1]-zz[0]] + conf['vesicle_chunk_size']
    """
    if np.abs(np.array(out_size) - vol.shape).max() == 0:
        # need to shift pad_xy
        return vol
    """
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

def vesicle_instance_crop_chunk(ves_file, im_file=None, bbs_file=None, ves_label=None, sz=[5,31,31], sz_thres=5, chunk_num=1, no_tqdm=False):
    im = None
    if chunk_num == 1: # read in the volume directly
        if isinstance(ves_file, str):
            ves = read_h5(ves_file)
            if im_file is not None:
                im = read_h5(im_file) 
        else: # volume
            ves = ves_file
            im = im_file
        if bbs_file is None or not os.path.exists(bbs_file):
            bbs = compute_bbox_all(ves)
            write_h5(bbs_file, bbs)
        else:
            bbs = read_h5(bbs_file)             
    else: # read in the volume by chunks
        if bbs_file is None or not os.path.exists(bbs_file):    
            bbs = compute_bbox_all_chunk(ves_file, chunk_num=chunk_num, no_tqdm=no_tqdm)
            write_h5(bbs_file, bbs)
        else:
            bbs = read_h5(bbs_file)
        fid_ves = h5py.File(ves_file, 'r')
        ves = fid_ves[list(fid_ves)[0]]
        if im_file is not None:
            fid_im = h5py.File(im_file, 'r')
            im = fid_im[list(fid_im)[0]]
        
    sz = np.array(sz)
    szh = sz//2
    out_im = np.zeros([0] + list(sz), np.uint8)
    out_mask = np.zeros([0] + list(sz), np.uint8)
    tmp = np.zeros(sz, np.uint8)
    out_l = []
    
        
    print('# instances:', len(bbs))
    # import pdb;pdb.set_trace()
    
    #for aa,bb in enumerate(bbs):
    for bb in tqdm(bbs, disable=no_tqdm):
        # remove small xy size
        if (bb[4::2]-bb[3::2]).min() >= sz_thres:
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
            tmp = np.pad(crop, [(pad_left[0],pad_right[0]), (pad_left[1],pad_right[1]), (pad_left[2],pad_right[2])], 'edge')
            out_mask = np.concatenate([out_mask, tmp[None]], axis=0)
            tmp[:] = 0            
            
            if im is not None:
                crop = np.array(im[max(0,cc[0]-szh[0]):cc[0]+szh[0]+1, \
                                   max(0,cc[1]-szh[1]):cc[1]+szh[1]+1, \
                                   max(0,cc[2]-szh[2]):cc[2]+szh[2]+1])
                tmp = np.pad(crop, [(pad_left[0],pad_right[0]), (pad_left[1],pad_right[1]), (pad_left[2],pad_right[2])], 'edge')
                try:
                    out_im = np.concatenate([out_im, tmp[None]], axis=0)
                except:
                    import pdb;pdb.set_trace()
                tmp[:] = 0
        """
        if aa==9:
            import pdb;pdb.set_trace()
        """
    # import pdb;pdb.set_trace()
    if chunk_num != 1: 
        fid_ves.close()
        if im_file is not None:
            fid_im.close()
            
    if ves_label is None:        
        if im is None:
            return out_mask
        return [out_im, out_mask]
    else:
        if im is None:
            return [out_mask, out_l]
        return [out_im, out_mask, out_l]
    

def neuron_id_to_vesicle(conf, neuron_id, ratio=[1,4,4], opt='big', output_file=None, neuron_file=None):
    if output_file is not None:
        if '.png' in output_file:            
            fns = glob.glob(output_file)
            if len(fns) > 0:
                print(f'File exists ({len(fns)}):  {output_file}') 
                return None
        elif os.path.exists(output_file):            
            print('File exists:', output_file)
            return None
                
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

def vesicle_vast_small_vesicle(seg_file, meta_file, output_file=None, output_chunk=8192):
    '''
    _, meta_n = read_vast_seg(meta_file)
    relabel = vast_meta_relabel(meta_file)
    if output_file is None or not os.path.exists(output_file):
        sv_id = [i for i,x in enumerate(meta_n) if x=='SV']        
        # connected component on each slice
        if output_file is None:
            ves = read_h5(seg_file)
            out_sv = np.zeros(ves.shape, np.uint16)
        else:
            ves_fid = h5py.File(seg_file, 'r')
            ves = ves_fid['main']
            fid = h5py.File(output_file, 'w')
            chunk_sz = get_h5_chunk2d(output_chunk, ves.shape[1:])
            out_sv = fid.create_dataset('main', ves.shape, np.uint16, compression="gzip", \
                chunks=(1,chunk_sz[0],chunk_sz[1]))
        max_id = 0
        for z in range(ves.shape[0]):
            slice = relabel[np.array(ves[z])]==sv_id
            if slice.any():
                slice_cc = cc3d.connected_components(slice, connectivity=4)
                mm = slice_cc.max()                                
                slice_cc[slice_cc > 0] += max_id
                out_sv[z] = slice_cc
                max_id += mm
        if output_file is not None:
            ves_fid.close()
            fid.close()
        else:
            return out_sv
    '''
def vesicle_vast_big_vesicle(seg_file, meta_file, dust_size=50, output_file=None, chunk_num=1, no_tqdm=False):
    '''
    meta_d, meta_n = read_vast_seg(meta_file)
    relabel = vast_meta_relabel(meta_file)    
    if output_file is None or not os.path.exists(output_file):
        sv_id = [i for i,x in enumerate(meta_n) if x=='SV']
        lv_id = [i for i,x in enumerate(meta_n) if x=='LV']        
        if output_file is None or chunk_num==1: # direct volume            
            ves = read_h5(seg_file)            
            max_id = ves.max()        
            ves_cc = cc3d.connected_components(ves==lv_id, connectivity=6)
            # remove small ones
            ves_cc = seg_remove_small(ves_cc, dust_size)    
            ves_cc[ves_cc > 0] += max_id
            ves[ves==sv_id] = 0
            ves[ves==lv_id] = 0
            ves_cc[ves > 0] = ves[ves > 0]    
            ves_cc = ves_cc.astype(np.uint16)
            if output_file is not None:
                write_h5(output_file, ves_cc)
            else:
                return ves_cc
        else:
            seg_func = lambda x: relabel[x]==lv_id            
            # write cc into output file
            seg_cc_chunk(seg_file, output_file, dt=np.uint16, seg_func=seg_func, chunk_num=chunk_num, no_tqdm=no_tqdm, dust_size=dust_size)
            seg_rm = [sv_id, lv_id]
            seg_add_chunk(output_file, chunk_num, 'all', np.uint16(meta_d[-1,0]), seg_file, seg_rm, no_tqdm=no_tqdm)
  '''
    
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
        for neuron in args.neuron[args.job_id::args.job_num]:
            neuron_id, neuron_name = neuron_to_id_name(conf, neuron)
            # zip -r vesicle_big_5_30-8-8.zip vesicle_big_5_30-8-8
            suff = arr_to_str(np.array(args.ratio) * conf['res'])
            output_file = f'{conf["result_folder"]}/vesicle_{args.vesicle}_{neuron_name}_{suff}'
            if 'file_type' in args.param and args.param['file_type']=='h5':
                output_file = output_file + '.h5'
            else:                
                mkdir(output_file)
                output_file = output_file + '/%04d.png'            
            neuron_file = f'{conf["result_folder"]}/neuron_{neuron_name}_{suff}.h5' if args.vesicle != 'im' else None
            neuron_id_to_vesicle(conf, neuron_id, args.ratio, args.vesicle, output_file, neuron_file)

    elif args.task == 'neuron-vesicle-proofread':
        # python vesicle_mask.py -t neuron-vesicle-proofread -ir /projects/weilab/dataset/hydra/vesicle_pf/ -n SHL18 -cn 20 -r 1,4,4 -v big
        for neuron in args.neuron[args.job_id::args.job_num]:
            neuron_id, neuron_name = neuron_to_id_name(conf, neuron)
            if args.input_file =='':
                args.input_file = f'{neuron_name}.h5,VAST_segmentation_metadata_{neuron_name}.txt'
            seg_file, meta_file = [os.path.join(args.input_folder, x) for x in args.input_file.split(',')]
            suffix = arr_to_str(conf['res'])
            sv_file, lv_file = [os.path.join(args.output_folder, f'vesicle_{x}_{neuron_name}_{suffix}.h5') for x in ['small','big']]  
            print(sv_file,lv_file)
            if args.vesicle in ['','big']:
                vesicle_vast_big_vesicle(seg_file, meta_file, \
                            output_file=lv_file, chunk_num=args.chunk_num)
            if args.vesicle in ['','small']:
                vesicle_vast_small_vesicle(seg_file, meta_file, output_file=sv_file)
            if max(args.ratio) != 1:
                suffix2 = arr_to_str(np.array(args.ratio)*conf['res'])    
                if args.vesicle in ['','small']:
                    sv_file2 = sv_file.replace(suffix, suffix2)            
                    seg_downsample_chunk(sv_file, args.ratio, sv_file2, args.chunk_num)                
                if args.vesicle in ['','big']:
                    lv_file2 = lv_file.replace(suffix, suffix2)
                    seg_downsample_chunk(lv_file, args.ratio, lv_file2, args.chunk_num)                

    elif args.task == 'neuron-vesicle-patch':
        # python vesicle_mask.py -t neuron-vesicle-patch -ir /projects/weilab/dataset/hydra/results_0408/ -n KR6 -v big -cn 10
        suffix = arr_to_str(conf['res'])
        for neuron in args.neuron[args.job_id::args.job_num]:
            #neuron_id, neuron_name = neuron_to_id_name(conf, neuron)
            neuron_name = args.neuron
            ves_file, im_file, bbs_file = [os.path.join(args.input_folder, f'vesicle_{x}_{neuron_name}_{suffix}.h5') for x in [args.vesicle, 'im', f'{args.vesicle}-bbs']]
            output_file = ves_file.replace('.h5', '_patch.h5')
            if not os.path.exists(output_file):                
                print(neuron_name)
                patch_sz = [5,31,31] if args.vesicle=='big' else [1,11,11]
                out = vesicle_instance_crop_chunk(ves_file, im_file, bbs_file, sz=patch_sz, sz_thres=0, chunk_num=args.chunk_num)
                write_h5(output_file, out)
