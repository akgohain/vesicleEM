import os, sys
import argparse
from util.io import *


if __name__ == "__main__":
    args = get_arguments()    
    if args.task == 'im-to-h5':
        image_type = 'image' if 'image_type' not in args.param else args.param['image_type']
        ratio = None if 'ratio' not in args.param else args.param['ratio']               
        data = read_image_folder(args.input, image_type=image_type, ratio=ratio)
        write_h5(args.output, data)
    elif args.task == 'vol-shape':
        if '.h5' in args.input:
            sz = get_volume_size_h5(args.input)
        else:
            data = read_vol(args.input)
            sz = data.shape    
        print(sz)
        