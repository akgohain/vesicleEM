import sys
import numpy as np
import h5py
sys.path.append('../')
from util import *
from glob import glob
opt = sys.argv[1]

Dd = '/projects/weilab/dataset/hydra/'
if opt == '0':
    # add 8192 to all bbox: tile_st 1 -> 0
    from glob import glob
    D0 = f'{Dd}mask_mip1/bbox/'
    fns = glob(D0+'*.txt')
    for fn in fns:
        data = np.loadtxt(fn).astype(int)
        if data.ndim == 1:
            data = data[None]
        data[:, 3:] += 8192
        np.savetxt(fn, data, '%d')
elif opt == '1':    
    data = read_yml("/projects/weilab/dataset/hydra/mask_mip1/neuron_id.txt")
    kk = np.array(list(data.values()))
    print(','.join([str(x) for x in kk]))
    """
    done = [4,15,16,25,36,37,38,39,40,41,52]
    print(sorted(kk[np.in1d(kk, done, invert=True)]))
    """
elif opt == '1.1': # check bbox
    nid = int(sys.argv[2])
    seg = read_h5("/projects/weilab/dataset/hydra/mask_mip5.h5")
    print(compute_bbox(seg==nid) * np.array([4,32,32]))
    
elif opt == '1.2':    
    aa = [x[x.rfind('neuron')+7:x.rfind('3')-1] for x in glob('/projects/weilab/dataset/hydra/results/neuron_*_30-8-8.h5')]
    cc = [x[x.rfind('neuron')+7:x.find('30')-1] for x in glob('/projects/weilab/dataset/hydra/results/neuron_*_30-32-32.h5')]
    for bb in aa:
        if bb not in cc:
            print(f'/projects/weilab/weidf/lib/miniconda3/envs/emu/bin/python run_local.py -t downsample -i /projects/weilab/dataset/hydra/results/neuron_{bb}_30-8-8.h5 -r 1,4,4 -o neuron_{bb}_30-32-32.h5 -cn 10')
elif opt == '1.21':    
    aa = [x[x.rfind('/')+1:x.rfind('_')] for x in glob('/projects/weilab/dataset/hydra/vesicle_pf/*_8nm.h5')]
    #aa = [x[x.rfind('ll_')+3:x.rfind('_30')] for x in glob('/projects/weilab/dataset/hydra/results/vesicle_small_*-32.h5')]
    bb = ','.join(aa)
    print(len(aa),bb)
    # print(f'python neuron_mask.py -t neuron-mask -n {bb}')
    # print(f'python vesicle_mask.py -t neuron-vesicle -n {bb} -v im -p "file_type:h5"')    
    print(f'python vesicle_mask.py -t neuron-vesicle-patch -ir /projects/weilab/dataset/hydra/results/ -n {bb} -v small')
    """
    for bb in aa:
        #print(f'python run_local.py -t im-to-h5 -p "image_type:seg" -ir "{bb}/*.png"')
        try:
            s1 = np.array(get_vol_shape(f'/projects/weilab/dataset/hydra/results/vesicle_im_{bb}_30-8-8.h5'))
            s2 = np.array(get_vol_shape(f'/projects/weilab/dataset/hydra/vesicle_pf/{bb}_8nm.h5'))
            if np.abs(s1-s2).max()!=0:
                print(bb,s1,s2)
        except:
            print(bb)    
        #pass
    """
elif opt == '1.3':
    aa = ['SHL29','SHL53','SHL52','PN8','SHL26','SHL51','KM2','SHL54']    
    fn = '/projects/weilab/dataset/hydra/results/neuron_%s_30-8-8.h5'
    for bb in aa:
        if not os.path.exists(fn%bb):
            print(bb)
        
elif opt == '2.1':
    fn = '/projects/weilab/dataset/hydra/vesicle_pf/*.h5'; tdt=np.uint16
    # fn = '/projects/weilab/dataset/hydra/results/vesicle_im_*.h5'; tdt=np.uint8
    fn = '/projects/weilab/dataset/hydra/results/sv_*_30-32-32.h5'; tdt=np.uint16
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
        print(f'python vesicle_mask.py -t neuron-vesicle-proofread -ir /projects/weilab/dataset/hydra/vesicle_pf/ -i {fn}_8nm.h5,VAST_segmentation_metadata_{fn}.txt -o sv_{fn},lv_{fn} -r 1,4,4')
        """
elif opt == '3': # check vast process
    from vesicle_mask import *
    fn = 'KR4'
    # check small vesicle
    D0 = '/projects/weilab/dataset/hydra/vesicle_pf/'
    Dr = '/projects/weilab/dataset/hydra/results/'
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
    Dr = '/projects/weilab/dataset/hydra/results/'
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
    fn = f'/projects/weilab/dataset/hydra/im_chunk/tile_{zz[0]}-{zz[1]}/{rc[0]}-{rc[1]}.h5'
    conf = read_yml('conf/param.yml')
    vol = read_h5(fn)
    out = crop_to_tile(vol, conf, opt, zz, rc)
    write_h5('ha.h5', out)
    """
    out = read_h5('ha.h5')
    out0 = read_h5('/projects/weilab/dataset/hydra/im_chunk/tile_388-488/8-13.h5')
    with viewer.txn() as s:
        s.layers.append(name='ii',layer=ng_layer(out, [8,8,30], tt='image',oo=[13*4096,8*4096,388]))
        s.layers.append(name='i0',layer=ng_layer(out0, [8,8,30], tt='image',oo=[13*4096,8*4096,388]))
    """

elif opt[0] == '6':
    Dr = '/projects/weilab/dataset/hydra/results/'
    nns = ['KR6','NET12','SHL55','KR11','KR10','SHL20','PN3','LUX2','KR4','KR5','KM4','RGC2','SHL17']
    nns=['NET10','NET11','SHL18','SHL24','SHL28','PN7','RGC7']
    nns=['SHL24']
    fns = ['big','small']
    if opt == '6': # vesicle small all 0
        # nns = ['KR5']
        for nn in nns:
            for fn in fns:
                sn = f'{Dr}vesicle_{fn}_{nn}_30-8-8_patch.h5'
                if not os.path.exists(sn):
                    print('No', sn)
                else:
                    im, seg = read_h5(sn)
                    num1 = (im.max(axis=1).max(axis=1).max(axis=1)==0).sum()
                    num2 = (seg.max(axis=1).max(axis=1).max(axis=1)==0).sum()
                    print(nn,fn,num1,num2)
                    import pdb; pdb.set_trace()
                    # bb=read_h5(f'{Dr}vesicle_big-bbs_SHL24_30-8-8.h5')
    elif opt == '6.1': # vesicle small all 0
        # python vesicle_mask.py -t neuron-vesicle-patch -ir /projects/weilab/dataset/hydra/results/ -n KR6 -v big
        nn = 'KR11'
        sn = f'{Dr}vesicle_{fn}_{nn}_30-8-8_patch.h5'
        im, seg = read_h5(sn)
        num2 = (seg.max(axis=1).max(axis=1).max(axis=1)==0)
    elif opt == '6.2': # vesicle small all 0
        nn = 'KR4'
        sn = read_h5(f'{Dr}vesicle_{fn}_{nn}_30-8-8.h5')
        chunk_num=10
        bbs = compute_bbox_all_chunk(sn, chunk_num=chunk_num)
        bbs2 = compute_bbox_all_chunk(sn, chunk_num=1)
        diff = np.abs(bbs-bbs2).max()
        print(diff)
    
elif opt[0] == '7': # vesicle pf
    nns = ['NET10', 'SHL28', 'RGC7', 'SHL18', 'SHL24', 'KR5', 'SHL55', 'PN7', 'NET11', 'PN3', 'KR4', 'RGC2', 'LUX2', 'NET12', 'KR6', 'KR10', 'SHL20', 'SHL17', 'KM4', 'KR11']
    nns = ['KM4', 'SHL55', 'KR5', 'KR11', 'PN3', 'KR10', 'NET12', 'KR6', 'KR4', 'NET11', 'PN7', 'RGC2', 'SHL20', 'LUX2']
    if opt == '7':# check unique id
        nn = 'KR5'
        fn = f'{Dd}/vesicle_pf/NET10.h5'
        fn = f'{Dd}/results_0408/vesicle_big_{nn}_30-32-32.h5'
        chunk_num = 5
        
        vol = h5py.File(fn, 'r')['main']
        num_z = int(np.ceil(vol.shape[0] / float(chunk_num)))
        uid = np.zeros([0])
        for i in range(chunk_num):
            seg = np.array(vol[i*num_z:(i+1)*num_z])
            uid = np.unique(np.hstack([uid, np.unique(seg)]))
            print(i, seg.max(), len(uid))
    elif opt == '7.1':
        from scipy.spatial.distance import cdist
        for nn in nns:
            nn = 'PN7'
            vd, vn = read_vast_seg(f'{Dd}vesicle_pf/VAST_segmentation_metadata_{nn}.txt')
            mm = vd[-1,0]
            bb0 = read_h5(f'{Dd}results/vesicle_big-bbs_{nn}_30-8-8.h5')
            bb = read_h5(f'{Dd}results_0408/vesicle_big-bbs_{nn}_30-8-8.h5')
            print(nn, len(bb), len(bb0),)
            bb0 = bb0[bb0[:,0]>mm]
            ind_mm = bb[:,0]>mm
            bb = bb[ind_mm]
            #ind = np.in1d(bb[:,0],bb0[:,0])
            #ind2 = np.in1d(bb0[:,0],bb[:,0])
            dis = cdist(bb0[:,1:], bb[:,1:], 'cityblock')
            ind = dis.min(axis=0)!=0
            print(ind.sum())
            write_h5(f'{Dd}results_0408/extra_patch/vesicle_big-bbs_{nn}_30-8-8.h5',bb[ind])
            patch = read_h5(f'{Dd}results_0408/vesicle_big_{nn}_30-8-8_patch.h5')
            import pdb; pdb.set_trace()
            patch[0] = patch[0][ind_mm][ind]
            patch[1] = patch[1][ind_mm][ind]
            write_h5(f'{Dd}results_0408/extra_patch/vesicle_big_{nn}_30-8-8_patch.h5',patch)

            yy, xx = np.where(dis==0)
            rl = np.zeros(bb0[-1,0]+1, np.uint16)
            rl[:mm+1] = np.arange(mm+1)
            rl[bb0[yy,0]] = bb[xx,0]
            write_h5(f'{Dd}results_0408/extra_patch/vesicle_big_{nn}_30-8-8_rl.h5',rl)
    elif opt == '7.2':# check unique id
        from glob import glob
        fns = [x[x.find('big')+4:-12] for x in glob(Dd+'results_0408/vesicle_big_*_30-32-32.h5')]
        print()
        import pdb; pdb.set_trace()
