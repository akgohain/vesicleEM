import sys
import numpy as np
import h5py
sys.path.append('../')
from util import *
from glob import glob
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
    print(','.join([str(x) for x in kk]))
    """
    done = [4,15,16,25,36,37,38,39,40,41,52]
    print(sorted(kk[np.in1d(kk, done, invert=True)]))
    """
elif opt == '1.1': # check bbox
    nid = int(sys.argv[2])
    seg = read_h5("/data/projects/weilab/dataset/hydra/mask_mip5.h5")
    print(compute_bbox(seg==nid) * np.array([4,32,32]))
    
elif opt == '2':    
    #aa = [x for x in glob('/data/projects/weilab/dataset/hydra/vesicle_pf/*') if '.' not in x]
    aa = [x[x.rfind('ll_')+3:x.rfind('_30')] for x in glob('/data/projects/weilab/dataset/hydra/results/vesicle_small_*-32.h5')]
    bb = ','.join(aa)
    print(len(aa),bb)
    # print(f'python neuron_mask.py -t neuron-mask -n {bb}')
    # print(f'python vesicle_mask.py -t neuron-vesicle -n {bb} -v im -p "file_type:h5"')    
    print(f'python vesicle_mask.py -t neuron-vesicle-patch -ir /data/projects/weilab/dataset/hydra/results/ -n {bb} -v small')
    """
    for bb in aa:
        #print(f'python run_local.py -t im-to-h5 -p "image_type:seg" -ir "{bb}/*.png"')
        try:
            s1 = np.array(get_vol_shape(f'/data/projects/weilab/dataset/hydra/results/vesicle_im_{bb}_30-8-8.h5'))
            s2 = np.array(get_vol_shape(f'/data/projects/weilab/dataset/hydra/vesicle_pf/{bb}_8nm.h5'))
            if np.abs(s1-s2).max()!=0:
                print(bb,s1,s2)
        except:
            print(bb)    
        #pass
    """
        
elif opt == '2.1':
    fn = '/data/projects/weilab/dataset/hydra/vesicle_pf/*.h5'; tdt=np.uint16
    # fn = '/data/projects/weilab/dataset/hydra/results/vesicle_im_*.h5'; tdt=np.uint8
    fn = '/data/projects/weilab/dataset/hydra/results/sv_*_30-32-32.h5'; tdt=np.uint16
    aa = glob(fn)
    for bb in aa:        
        # check dtype and shape
        try:
            print(bb, read_h5(bb).max())
            """
            fid = h5py.File(bb,'r')['main']
            if fid.dtype != tdt:
                print(bb, fid.dtype)                
                if fid.dtype == np.uint32:
                    vol = read_h5(bb).astype(np.uint16)
                    write_h5(bb+'_bk', vol)
                else:
                    print(f'rm {bb}')
            """
        except:
           print(bb, 'bug') 
        """
        # generate proofread volume
        fn = bb[bb.rfind('/')+1:bb.rfind('_')]        
        print(f'python vesicle_mask.py -t neuron-vesicle-proofread -ir /data/projects/weilab/dataset/hydra/vesicle_pf/ -i {fn}_8nm.h5,VAST_segmentation_metadata_{fn}.txt -o sv_{fn},lv_{fn} -r 1,4,4')
        """
elif opt == '3': # check vast process
    from vesicle_mask import *
    fn = 'KR4'
    # check small vesicle
    D0 = '/data/projects/weilab/dataset/hydra/vesicle_pf/'
    Dr = '/data/projects/weilab/dataset/hydra/results/'
    seg_file = f'{D0}{fn}_8nm.h5' 
    meta_file = f'{D0}VAST_segmentation_metadata_{fn}.txt'
    """
    out_sv = vesicle_vast_small_vesicle(seg_file, meta_file)
    out_sv_pre = read_h5(f'{Dr}sv_{fn}_30-8-8.h5')
    print('sv:', np.abs(out_sv_pre[::8,::8,::8].astype(float)-out_sv[::8,::8,::8]).max())    
    out_sv4 = seg_downsample_chunk(f'{Dr}sv_{fn}_30-8-8.h5', [1,4,4],chunk_num=5)
    out_sv4_pre = read_h5(f'{Dr}sv_{fn}_30-32-32.h5')
    print('sv4:', np.abs(out_sv4_pre.astype(float)-out_sv4).max())
    """
    # check big vesicle
    out_lv = vesicle_vast_big_vesicle(seg_file, meta_file, chunk_num=5)
    out_lv_pre = read_h5(f'{Dr}lv_{fn}_30-8-8.h5')
    ui, uc = np.unique(out_lv, return_counts=True)
    ui2, uc2 = np.unique(out_lv_pre, return_counts=True)
    print('lv:', np.abs(np.sort(uc)-np.sort(uc2)).max())
    out_lv4 = seg_downsample_chunk(f'{Dr}lv_{fn}_30-8-8.h5', [1,4,4],chunk_num=5)
    out_lv4_pre = read_h5(f'{Dr}lv_{fn}_30-32-32.h5')
    ui3, uc3 = np.unique(out_lv4, return_counts=True)
    ui4, uc4 = np.unique(out_lv4_pre, return_counts=True)
    print('lv4:', np.abs(np.sort(uc3)-np.sort(uc4)).max())    
elif opt == '4':    
    from imageio import imwrite
    from skimage.color import label2rgb
    Dr = '/data/projects/weilab/dataset/hydra/results/'
    nn = 'KR4'
    seg_file = f'{Dr}vesicle_big_{nn}_30-8-8_patch.h5' 
    result = read_h5(seg_file)
    ii = 0
    def compare(ii):
        imwrite('test_im_%d.png'%ii, result[0][ii,3])
        imwrite('test_mask_%d.png'%ii, label2rgb(result[1][ii,3], image=result[0][ii,3]))
        #imwrite('test_seg_%d.png'%ii, result[1][ii,3])
    compare(ii)
elif opt == '5':# image volume drift by 128 in xy    
    from vesicle_mask import crop_to_tile 
    zz, rc = [388,488], [8,13]
    opt = 'im'
    fn = f'/data/projects/weilab/dataset/hydra/im_chunk/tile_{zz[0]}-{zz[1]}/{rc[0]}-{rc[1]}.h5'
    conf = read_yml('conf/param.yml')
    vol = read_h5(fn)
    out = crop_to_tile(vol, conf, opt, zz, rc)
    write_h5('ha.h5', out)
    """
    out = read_h5('ha.h5')
    out0 = read_h5('/data/projects/weilab/dataset/hydra/im_chunk/tile_388-488/8-13.h5')
    with viewer.txn() as s:
        s.layers.append(name='ii',layer=ng_layer(out, [8,8,30], tt='image',oo=[13*4096,8*4096,388]))
        s.layers.append(name='i0',layer=ng_layer(out0, [8,8,30], tt='image',oo=[13*4096,8*4096,388]))
    """

elif opt == '6': # vesicle small all 0
    Dr = '/data/projects/weilab/dataset/hydra/results/'
    nn = 'KR5'
    im, seg = read_h5(f'{Dr}vesicle_small_{nn}_30-8-8_patch.h5')
    np.squeeze(im.max(axis=2).max(axis=2))
    np.squeeze(seg.max(axis=2).max(axis=2))
    