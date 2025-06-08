# dependencies 
import os
import sys
import glob
import tqdm
import yaml
import shutil
import subprocess
import numpy as np
import logging as log
from connectomics.utils.evaluate import adapted_rand, get_binary_jaccard
from connectomics.data.utils.data_io import readvol, savevol
from connectomics.utils.process import binary_watershed, bc_watershed, bcd_watershed



def do_inference(
        im_path: str,
        pred_path: str,
        config_list: list,
        checkpoint: str,
        batch_size: int=392,
        device=0,
        watershed=True,
        show=False):
    """
    do_inference: perform a single instance of model inference. attempts to
        detect an appropriate watershed technique based on the shape of the
        prediction.

    Arguments:
        im_path (str): the path to the image volume
        pred_path (str): the path to folder that contains the prediction volume
        config_list (list): paths to configurations files for PyTC. Should have
            a length of 1 or 2. if 1 file is give, that file is passed in via 
            --config-file. if 2 files are given, the first is passed in via
            --config-base and the second is passed in via --config-file.
        checkpoint (str): the path to the checkpoint file to load
        batch_size (int): the batch size for inference
        device (int): the CUDA device to use
        watershed (bool): whether or not to do watershed
        show (bool): whether or not to suppress outputs
    """

    # split im_path and pred_path
    im_path, im_name = '/'.join(im_path.split('/')[:-1]), im_path.split('/')[-1]
    pred_path, pred_name = '/'.join(pred_path.split('/')[:-1]), pred_path.split('/')[-1]

    # create pred_path is necessary
    subprocess.run(['mkdir', '-p', pred_path])   

    # set config(s)
    assert len(config_list) in [1, 2]
    config_base = config_list[0]
    if len(config_list) == 2:
        config_file = config_list[1]

    # load the config(s) and modify base_config as necessary
    with open(config_base) as fp:
        config_base = yaml.safe_load(fp)
    if len(config_list) == 2:
        with open(config_file) as fp:
            config_file = yaml.safe_load(fp)
    config_base['INFERENCE']['INPUT_PATH'] = im_path
    config_base['INFERENCE']['IMAGE_NAME'] = im_name
    config_base['INFERENCE']['OUTPUT_PATH'] = pred_path
    config_base['INFERENCE']['OUTPUT_NAME'] = pred_name
    config_base['INFERENCE']['SAMPLES_PER_BATCH'] = batch_size

    # save temporary copies of config(s)
    with open(os.path.join(pred_path, 'base.yaml'), 'w') as fp:
        dump = yaml.dump(config_base)
        fp.write(dump)
    if len(config_list) == 2:
        shutil.copy(config_list[1], os.path.join(pred_path, 'extra.yaml'))

    # execute inference
    if len(config_list) == 1:
        cmd = f"CUDA_VISIBLE_DEVICES={device} python scripts/main.py --config-file {os.path.join(pred_path, 'base.yaml')} --checkpoint {checkpoint} --inference SYSTEM.NUM_CPUS 2 SYSTEM.NUM_GPUS 1 SYSTEM.DISTRIBUTED False SYSTEM.PARALLEL 'DP'"
    else:
        cmd = f"CUDA_VISIBLE_DEVICES={device} python scripts/main.py --config-base {os.path.join(pred_path, 'base.yaml')} --config-file {os.path.join(pred_path, 'extra.yaml')} --checkpoint {checkpoint} --inference SYSTEM.NUM_CPUS 2 SYSTEM.NUM_GPUS 1 SYSTEM.DISTRIBUTED False SYSTEM.PARALLEL 'DP'"
    try:
        subprocess.run(cmd, shell=True, stdout=None if show else subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(e.output)
        sys.exit(1)

    # perform watershed if necessary
    if watershed == True:
        pred = readvol(os.path.join(pred_path, pred_name))
        # watershed values were found through trial and error
        if pred.shape[0] == 1:
            pred = binary_watershed(pred, thres2=0.5)
        elif pred.shape[0] == 2:
            pred = bc_watershed(pred, thres3=0.3, thres4=0.5)
        elif pred.shape[0] == 3:
            pred = bcd_watershed(pred, thres3=0.3, thres4=0.35)
        savevol(os.path.join(pred_path, pred_name), pred)



def get_iou(
        gt_path: str,
        pred_path: str,
        mask_path: str,
        print_stats: bool=False,
        scale: list=None):
    """
    get_iou: calculate IoU for binary segementations.
    
    Arguments:
        gt_path (str): the path to the ground truth volume
        pred_path (str): the path the prediction volume
        mask_path (str): the path to the mask volume
        print_stats (bool): whether or not to print stats out
        scale (list): how much to scale each dimension of the gt and mask by
    """
    
    # these can be instance segmentations or semantic segmentation maps
    gt = readvol(gt_path).squeeze()
    pred = readvol(pred_path).squeeze()
    if mask_path is not None:
        mask = readvol(mask_path).squeeze()

    # scale if necessary
    if scale is not None:
        gt = np.tile(gt, scale)
        if mask_path is not None:
            mask = np.tile(mask, scale)
        
    # apply mask if necessary
    if mask_path is not None:
        pred = pred * np.clip(mask, 0, 1)
    
    # converts segmentations into binary segmentations, if necessary
    gt = np.clip(gt, 0, 1)
    pred = np.clip(pred, 0, 1)
    
    # calculate stats
    stats = get_binary_jaccard(pred, gt)[0]

    # print stats, if necessary
    if print_stats:
        print('#####################')
        print(f'positive IoU:\t{stats[0]:.03}')
        print(f'overall IoU:\t{stats[1]:.03}')
        print(f'precision:\t{stats[2]:.03}')
        print(f'recall:\t\t{stats[3]:.03}')
        print('#####################')
    
    # return IoU stats in order [positive IoU, overall IoU, precision, recall]
    return stats



def get_rand_index(
        gt_path: str,
        pred_path: str,
        mask_path = str,
        print_stats: bool=False,
        scale: list=None):
    """
    get_rand_index: calculate Rand score for instance segmetnations.
    
    Arguments:
        gt_path (str): the path to the ground truth volume
        pred_path (str): the path to the prediction volume
        mask_path (str): the path to the mask volume
        print_stats (bool): whether or not to print stats out
        scale (list): how much to scale each dimension of the gt and mask by
    """
    
    # these should be semantic segmentation maps
    gt = readvol(gt_path).squeeze()
    pred = readvol(pred_path).squeeze()
    if mask_path is not None:
        mask = readvol(mask_path).squeeze()

    # scale if necessary
    if scale is not None:
        gt = np.tile(gt, scale)
        if mask_path is not None:
            mask = np.tile(mask, scale)
        
    # apply mask if necessary
    if mask_path is not None:
        pred = pred * np.clip(mask, 0, 1)

    # calculate Rand index
    stats = adapted_rand(pred, gt, all_stats=True)
    
    # print stats, if necessary
    if print_stats:
        print('#####################')
        print(f'Rand index:\t{stats[0]:.03}')
        print(f'precision:\t{stats[1]:.03}')
        print(f'recall:\t\t{stats[2]:.03}')
        print('#####################')

    # return stats in order [rand_index, precision, recall]
    return stats



def infer_and_stats(
        im_list: list,
        gt_list: list,
        mask_list: list,
        config_list: list,
        checkpoint: str,
        batch_size: int=392,
        metric: str='rand',
        print_stats: bool=False,
        print_progress: bool=True,
        scale: list=None,
        device=0):
    """
    infer_and_stats: calculate a specific error metric using a given list of
        images and groud truths using one specific checkpoint. images and
        ground truths should correspond to one another.

        im_list (list): list of paths to image volumes
        gt_list (list): list of paths to ground truth volumes
        mask_list (list): list of paths to mask volumes. can be None
        config_list (list): list of PyTC configurations; see do_inference for
            more information
        checkpoint (str): path to model checkpoint to use
        batch_size (int): batch size to use for inference
        metric (str): metric to calculate. currently accepts 'iou' and 'rand'
        print_stats (bool): whether or not to print stats out
        print_progress (bool): whether to use tqdm or not
        scale (list): whether to scale the gt and mask or not
        device (int): which CUDA device to use
    """

    # sanity check; each image should have a corresponding ground truth
    assert len(im_list) == len(gt_list)
    
    # make sure mask list has correct size if None
    if mask_list == None:
        mask_list = [None] * len(im_list)

    # check for correct metric
    assert metric in ['iou', 'rand']

    # loop through pairs of images and ground truths. for each pair, calculate
    # the error metric
    metric_list = []
    for im, gt, mask in tqdm.tqdm(zip(im_list, gt_list, mask_list), total=len(im_list)) if print_progress else zip(im_list, gt_list, mask_list):
       fname_id = np.random.randint(10000)
       fname = f'/tmp/pred_{fname_id}.h5'
       do_inference(im, fname, config_list, checkpoint, batch_size=batch_size, device=device)
       if metric == 'iou':
           metric_list.append([im, gt, get_iou(gt, fname, mask, scale=scale)])
       if metric == 'rand':
           metric_list.append([im, gt, get_rand_index(gt, fname, mask, scale=scale)])

    # print if necessary. each metric has its own case
    if print_stats:
        for im, gt, stats in metric_list:
            im, gt = im.split('/')[-1], gt.split('/')[-1]
            # IoU
            if metric == 'iou':
                print('#####################')
                print(f'im:\t\t{im}')
                print(f'gt:\t\t{gt}')
                print(f'positive IoU:\t{stats[0]:.03}')
                print(f'overall IoU:\t{stats[1]:.03}')
                print(f'precision:\t{stats[2]:.03}')
                print(f'recall:\t\t{stats[3]:.03}')
                print('#####################')
            # Rand index
            if metric == 'rand':
                print('#####################')
                print(f'im:\t\t{im}')
                print(f'gt:\t\t{gt}')
                print(f'Rand index:\t{stats[0]:.03}')
                print(f'precision:\t{stats[1]:.03}')
                print(f'recall:\t\t{stats[2]:.03}')
                print('#####################')
   
    # return the list of metrics
    return metric_list
