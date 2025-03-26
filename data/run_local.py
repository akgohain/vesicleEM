import os, sys
import glob
sys.path.append('../')
from util import *


if __name__ == "__main__":
    args = get_arguments()    
    if args.task == 'im-to-h5':
        # python run_local.py -t im-to-h5 -p "image_type:seg,ran:403-1356" -ir "/data/projects/weilab/dataset/hydra/vesicle_pf/KR4 Proofread 2/*.png" -o /data/projects/weilab/dataset/hydra/vesicle_pf/KR4_8nm.h5 
        # python run_local.py -t im-to-h5 -p "image_type:seg" -ir "/data/projects/weilab/dataset/hydra/vesicle_pf/KR6 PNG/*.png"
        # python run_local.py -t im-to-h5 -ir "/data/projects/weilab/dataset/hydra/results/vesicle_im_KR4_30-8-8/*.png"
        image_type = 'image' if 'image_type' not in args.param else args.param['image_type']
        ratio = None if 'ratio' not in args.param else args.param['ratio']
        image_template = args.input_folder
        if args.output_file == '':
           args.output_file = f'{os.path.dirname(image_template)}.h5' 
        image_index = None
        if 'ran' in args.param:
            ran = [int(x) for x in args.param['ran'].split('-')]
            image_index = range(ran[0], ran[1]+1)
            ims = glob.glob(image_template)
            i1, i2 = ims[0].rfind('s'), ims[0].rfind('.')
            num_digit = i2-i1-1
            image_template = f'{ims[0][:i1+1]}%0{num_digit}d{ims[0][i2:]}'            
         
        dt = np.uint8 if image_type=='image' else np.uint16
        data = read_image_folder(image_template, image_index, \
                        image_type=image_type, ratio=ratio, output_file=args.output_file, dtype=dt)
    elif args.task == 'downsample':
        # python run_local.py -t downsample -i /data/projects/weilab/dataset/hydra/results/neuron_NET11_30-8-8.h5 -r 1,4,4 -o neuron_NET11_30-32-32.h5
        output_file = os.path.join(os.path.dirname(args.input_file), args.output_file)
        vol_downsample_chunk(args.input_file, args.ratio, output_file, args.chunk_num)
    elif args.task == 'vol-shape':
        # python run_local.py -t vol-shape -i /data/projects/weilab/dataset/hydra/results/sv_KR4_30-32-32.h5
        if '.h5' in args.input_file:
            sz = get_vol_shape(args.input_file)
        else:
            data = read_h5(args.input_file)
            sz = data.shape    
        print(sz)
    elif args.task == 'neuron-image':
        # python run_local.py -t neuron-image -n LUX2
        ratio = [1,1,1]
        pad = np.array([100,1000,1000])
        conf = read_yml('conf/param.yml')
        neuron_id, neuron_name = neuron_to_id_name(conf, args.neuron[0])
        bb = neuron_id_to_bbox(conf, neuron_id)
        bb[::2] = np.maximum(bb[::2]-pad, 0)
        bb[1::2] = np.minimum(bb[1::2]+pad, 0)
        zyx_sz = [100, 4096// ratio[1], 4096// ratio[2]]
        bb[:2] = bb[:2] // ratio[0]
        bb[2:4] = bb[2:4] // ratio[1]
        bb[4:] = bb[4:] // ratio[2]    

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

