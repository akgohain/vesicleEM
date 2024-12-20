import os, sys
import numpy as np
sys.path.append('../')
from util import *
from neuron_mask import neuron_id_to_bbox, neuron_name_to_id

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
    port = 9094 # change to an unused port number
    neuroglancer.set_server_bind_address(bind_address=ip,bind_port=port)
    viewer = neuroglancer.Viewer()
    D0 = 'precomputed://https://rhoana.rc.fas.harvard.edu/ng/hydra/im_w0210'
    with viewer.txn() as s:
        s.layers['image'] = neuroglancer.ImageLayer(source=D0)
    
    conf = read_yml('conf/param.yml')
    if opt == '0': 
        # python -i visualization.py 0
        # visualize bv mask
        with viewer.txn() as s:
            s.layers['image'] = neuroglancer.ImageLayer(source=D0) 
        conf = read_yml('conf/param.yml')
        def dsp_seg(name, neuron_names, rr=2):
            for neuron_name in neuron_names:
                neuron_id = neuron_name_to_id(conf, neuron_name)
                bb = neuron_id_to_bbox(conf, neuron_id)        
                if 'im' in name:
                    tt = 'image'
                    oset = bb[::2]
                    mask = read_h5(f'{conf["result_folder"]}/{name}_{neuron_name}_30-8-8.h5')
                    res = np.array([8,8,30])
                else:
                    oset = bb[::2]//[1,4,4]//[rr,rr,rr]
                    mask = read_h5(f'{conf["result_folder"]}/{name}_{neuron_name}_30-32-32.h5')[::rr,::rr,::rr]
                    tt = 'segmentation'
                    res = np.array([32,32,30])*rr
                with viewer.txn() as s:
                    s.layers.append(name='mask',layer=ng_layer(mask, res, tt=tt, oo=oset[::-1]))
        with viewer.txn() as s:
            s.layers.append(name='ph',layer=ng_layer(np.zeros([1,1,1]), [8,8,30]))
        #dsp_seg('neuron', ['KR5', 'KR6'])
        #dsp_seg('neuron', ['KR4']);dsp_seg('sv', ['KR4']);dsp_seg('lv', ['KR4'])
        #dsp_seg('neuron', ['KR4']);dsp_seg('vesicle_big', ['KR4'])
        dsp_seg('vesicle_big', ['SHL24'])
        dsp_seg('vesicle_im', ['SHL24'], 1)
        #dsp_seg('neuron', ['RGC7'],2)
    print(viewer)
