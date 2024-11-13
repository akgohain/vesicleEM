import os, sys
import glob
from util import *


if __name__ == "__main__":
    args = get_arguments()    
    if args.task == 'im-to-h5':
        # python run_local.py -t im-to-h5 -p "image_type:seg,ran:403-1356" -ir "/data/projects/weilab/dataset/hydra/vesicle_pf/KR4 Proofread 2/*.png" -o /data/projects/weilab/dataset/hydra/vesicle_pf/KR4_8nm.h5 
        image_type = 'image' if 'image_type' not in args.param else args.param['image_type']
        ratio = None if 'ratio' not in args.param else args.param['ratio']
        image_template = args.input_folder
        image_index = None
        if 'ran' in args.param:
            ran = [int(x) for x in args.param['ran'].split('-')]
            image_index = range(ran[0], ran[1]+1)
            ims = glob.glob(image_template)
            i1, i2 = ims[0].rfind('s'), ims[0].rfind('.')
            num_digit = i2-i1-1
            image_template = f'{ims[0][:i1+1]}%0{num_digit}d{ims[0][i2:]}'            
            
        data = read_image_folder(image_template, image_index, image_type=image_type, ratio=ratio)
        write_h5(args.output_file, data)
    elif args.task == 'downsample':
        # python run_local.py -t downsample -i /data/projects/weilab/dataset/hydra/results/neuron_NET11_30-8-8.h5 -r 1,4,4 -o neuron_NET11_30-32-32.h5
        vol = read_h5(args.input_file)
        vol = vol[::args.ratio[0], ::args.ratio[1], ::args.ratio[2]]
        sn = os.path.join(os.path.dirname(args.input_file), args.output_file)
        write_h5(sn, vol)
    elif args.task == 'vol-shape':
        # python run_local.py -t vol-shape -i /data/projects/weilab/dataset/hydra/results/sv_KR4_30-32-32.h5
        if '.h5' in args.input_file:
            sz = get_volume_size_h5(args.input_file)
        else:
            data = read_h5(args.input_file)
            sz = data.shape    
        print(sz)
        