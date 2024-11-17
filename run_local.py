import os, sys
import glob
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
        # python run_local.py -t downsample -i /data/projects/weilab/dataset/hydra/results/neuron_NET11_30-8-8.h5 -r 1,4,4 -o neuron_NET11_30-32-32.h5 -p "chunk:10"
        sn = os.path.join(os.path.dirname(args.input_file), args.output_file)
        if 'chunk' in args.param:
            # read in chunks
            chunk_num = int(args.param['chunk'])            
            
            fid_in = h5py.File(args.input_file, 'r')
            fid_in_data = fid_in['main']
            fid_out = h5py.File(args.output_file, "w")
            vol_sz = np.array(fid_in_data.shape) // args.ratio
            result = fid_out.create_dataset('main', vol_sz, dtype=fid_in_data.dtype)
            num_z = int(np.ceil(vol_sz[0] / float(chunk_num)))
            for z in range(chunk_num):
                tmp = read_h5_chunk(fid_in_data, z, chunk_num)[::args.ratio[0],::args.ratio[1],::args.ratio[2]]
                result[z*num_z:(z+1)*num_z] = tmp
            fid_in.close()
            fid_out.close()
        else:
            # read in the volume
            vol = read_h5(args.input_file)
            vol = vol[::args.ratio[0], ::args.ratio[1], ::args.ratio[2]]
            write_h5(sn, vol)
    elif args.task == 'vol-shape':
        # python run_local.py -t vol-shape -i /data/projects/weilab/dataset/hydra/results/sv_KR4_30-32-32.h5
        if '.h5' in args.input_file:
            sz = get_volume_size_h5(args.input_file)
        else:
            data = read_h5(args.input_file)
            sz = data.shape    
        print(sz)