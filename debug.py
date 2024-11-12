import sys
import numpy as np
from util import *

opt = sys.argv[1]

if opt == '0':
    # add 8192 to all bbox: tile_st 1 -> 0
    from glob import glob
    D0 = '/data/projects/weilab/dataset/hydra/mask_mip1/bbox/'
    fns = glob(D0+'*.txt')
    for fn in fns:
        data = np.loadtxt(fn).astype(int)
        if data.ndim == 1:
            data = data[None]
        data[:, 3:] += 8192
        np.savetxt(fn, data, '%d')
elif opt == '1':    
    data = read_yml("/data/projects/weilab/dataset/hydra/mask_mip1/neuron_id.txt")
    kk = np.array(list(data.values()))
    done = [4,15,16,25,36,37,38,39,40,41,52]
    print(sorted(kk[np.in1d(kk, done, invert=True)]))