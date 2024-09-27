import os, sys
import numpy as np

from util import *
from neuron_mask import segid_to_bbox

def ng_layer(data, res, oo=[0, 0, 0], tt="segmentation"):
    # input zyx -> display xyz
    dim = neuroglancer.CoordinateSpace(names=["x", "y", "z"], units="nm", scales=res)
    return neuroglancer.LocalVolume(
        data.transpose([2, 1, 0]), volume_type=tt, dimensions=dim, voxel_offset=oo
    )
    
    
opt = sys.argv[1]

if opt[0] == '0':
    # ng visualization
    import neuroglancer
    ip = 'localhost' # or public IP of the machine for sharable display
    port = 9092 # change to an unused port number
    neuroglancer.set_server_bind_address(bind_address=ip,bind_port=port)
    viewer = neuroglancer.Viewer()
    D0 = 'precomputed://https://rhoana.rc.fas.harvard.edu/ng/hydra/im_w0210'
    with viewer.txn() as s:
        s.layers['image'] = neuroglancer.ImageLayer(source=D0)
    
    conf = read_yml('conf/param.yml')
    if opt == '0': 
        # python -i visualization.py 0
        # visualize bv mask
        res = np.array([32,32,30])
        with viewer.txn() as s:
            s.layers['image'] = neuroglancer.ImageLayer(source=D0) 
        
        def dsp_seg(name, seg_ids):
            for seg_id in seg_ids:
                bb = segid_to_bbox(conf, seg_id)        
                oset = bb[::2]//[1,4,4]        
                mask = read_h5(f'{conf["result_folder"]}/{name}_{seg_id}_30-32-32.h5')
                with viewer.txn() as s:
                    s.layers.append(name='mask',layer=ng_layer(mask, res, oo=oset[::-1]))
        dsp_seg('neuron', [16])
        dsp_seg('vesicle_big', [16])
    print(viewer)
