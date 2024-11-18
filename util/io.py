import os, sys
import glob
import numpy as np
import imageio
from scipy.ndimage import zoom
import yaml
import h5py
from tqdm import tqdm
import argparse
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def get_arguments():
    """
    The function `get_arguments()` is used to parse command line arguments for the evaluation on AxonEM.
    :return: The function `get_arguments` returns the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="argument parser"
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        help="task",
        default="",
    )
    parser.add_argument(
        "-ir",
        "--input-folder",
        type=str,
        help="path to input folder",
        default="",
    )
    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        help="path to input file",
        default="",
    )
    parser.add_argument(
        "-or",
        "--output-folder",
        type=str,
        help="path to output folder",
        default="",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        help="path to output file",
        default="",
    )
    parser.add_argument(
        "-p",
        "--param",
        type=str,
        help="extra parameter",
        default="",
    )
    parser.add_argument(
        "-r",
        "--ratio",
        type=str,
        help="ratio",
        default="1,1,1",
    )
    parser.add_argument(
        "-n",
        "--neuron",
        type=str,
        help="neuron name or id",
        default="",
    )
    parser.add_argument(
        "-v",
        "--vesicle",
        type=str,
        help="big or small vesicle",
        default="big",
    )
    parser.add_argument(
        "-ji",
        "--job-id",
        type=int,        
        default=0,
    )
    parser.add_argument(
        "-jn",
        "--job-num",
        type=int,        
        default=1,
    )
    args = parser.parse_args()
    if args.param != "":
        args.param = str2dict(args.param)
    args.neuron = args.neuron.split(',')        
    args.ratio = [int(x) for x in args.ratio.split(',')]
    return args


def mkdir(foldername, opt=""):
    """
    Create a directory.

    Args:
        fn (str): The path of the directory to create.
        opt (str, optional): The options for creating the directory. Defaults to "".

    Returns:
        None

    Raises:
        None
    """
    if opt == "parent":  # until the last /
        foldername = os.path.dirname(foldername)
    if not os.path.exists(foldername):
        if "all" in opt or "parent" in opt:
            os.makedirs(foldername)
        else:
            os.mkdir(foldername)
            
def read_yml(filename):
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
    return data

def arr_to_str(arr):
    return '-'.join([str(x) for x in arr])

def read_image(filename, image_type="image", ratio=None, resize_order=None, data_type="2d", crop=None):
    """
    Read an image from a file.

    Args:
        filename (str): The path to the image file.
        image_type (str, optional): The type of image to read. Defaults to "image".
        ratio (int or list, optional): The scaling ratio for the image. Defaults to None.
        resize_order (int, optional): The order of interpolation for scaling. Defaults to 1.
        data_type (str, optional): The type of image data to read. Defaults to "2d".

    Returns:
        numpy.ndarray: The image data.

    Raises:
        AssertionError: If the ratio dimensions do not match the image dimensions.
    """
    if data_type == "2d":
        # assume the image of the size M x N x C
        image = imageio.imread(filename)
        if image_type == "seg":
            image = rgb_to_seg(image)
        if ratio is not None:
            if str(ratio).isnumeric():
                ratio = [ratio, ratio]
            if ratio[0] != 1:
                if resize_order is None:
                    resize_order = 0 if image_type == "seg" else 1
                if image.ndim == 2:
                    image = zoom(image, ratio, order=resize_order)
                else:
                    # do not zoom the color channel
                    image = zoom(image, ratio + [1], order=resize_order)
        if crop is not None:
            image = image[crop[0]: crop[1], crop[2]: crop[3]]
    else:
        # read in nd volume
        image = imageio.volread(filename)
        if ratio is not None:
            assert (
                str(ratio).isnumeric() or len(ratio) == image.ndim
            ), f"ratio's dim {len(ratio)} is not equal to image's dim {image.ndim}"
            image = zoom(image, ratio, order=resize_order)
        if crop is not None:
            obj = tuple(slice(crop[x*2], crop[x*2+1]) for x in range(image.ndim))
            image = image[obj]
    return image

def get_file_number(filename, index):
    return len(filename) if isinstance(filename, list) else len(index)

def get_filename(filename, index, x):
    return filename[x] if isinstance(filename, list) else filename % index[x]

def read_image_folder(
    filename, index=None, image_type="image", ratio=None, resize_order=None, crop=None, no_tqdm=False, dtype=None, output_file=None
):
    """
    Read a folder of images.

    Args:
        filename (str or list): The path to the image folder or a list of image file paths.
        index (int or list, optional): The index or indices of the images to read. Defaults to None.
        image_type (str, optional): The type of image to read. Defaults to "image".
        ratio (list, optional): The downsampling ratio for the images. Defaults to None.
        resize_order (int, optional): The order of interpolation for scaling. Defaults to 1.

    Returns:
        numpy.ndarray: The folder of images.

    Raises:
        None
    """
    if ratio is None:
        ratio = [1, 1]
    # either filename or index is a list
    if '*' in filename:
        filename = sorted(glob.glob(filename))    
    num_image = get_file_number(filename, index)
    
    i = 0
    while not os.path.exists(get_filename(filename, index, i)):
        i += 1        
    im0 = read_image(get_filename(filename, index, i), image_type, ratio, resize_order, crop=crop)
    sz = list(im0.shape)
    dt = dtype if dtype is not None else im0.dtype
    out_sz = [num_image] + sz
    if output_file is None:
        out = np.zeros(out_sz, dt)
    else:
        fid = h5py.File(output_file, 'w')
        out = fid.create_dataset('main', out_sz, dtype=dt)
    for i in tqdm(range(num_image), disable=no_tqdm):
        fn = get_filename(filename, index, i)
        if os.path.exists(fn):
            out[i] = read_image(fn, image_type, ratio, resize_order, crop=crop).astype(dt)
            
    if output_file is None: 
        return out
    else:
        fid.close()
        
def get_filenames(folder_name, output_name=None):
    fns = [x[x.rfind('/')+1:x.rfind('.')] for x in glob(f'{folder_name}/*.png')]
    if output_name is None:
        if folder_name[-1] == '/':
            folder_name = folder_name[:-1]
        output_name =f'{folder_name}.txt'
    write_txt(output_name, fns)        

def get_seg_dtype(mid):
    """
    Get the appropriate data type for a segmentation based on the maximum ID.

    Args:
        mid (int): The maximum ID in the segmentation.

    Returns:
        numpy.dtype: The appropriate data type for the segmentation.

    Notes:
        - The function determines the appropriate data type based on the maximum ID in the segmentation.
        - The data type is selected to minimize memory usage while accommodating the maximum ID.
    """    
    m_type = np.uint64
    if mid < 2**8:
        m_type = np.uint8
    elif mid < 2**16:
        m_type = np.uint16
    elif mid < 2**32:
        m_type = np.uint32
    return m_type


def read_txt(filename):
    """
    Args:
    filename (str): The path to the text file.

    Returns:
    "main",  list: The lines of the text file as a list of strings.

    Raises:
        None
    """
    with open(filename, "r") as a:
        content = a.readlines()
    return content
def write_image(filename, image, image_type="image"):
    if image_type=='seg':
        image = seg_to_rgb(image)
    imageio.imsave(filename, image)

def write_txt(filename, content):
    """
    Write content to a text file.

    Args:
        filename (str): The path to the text file.
        content (str or list): The content to write. If a list, each element will be written as a separate line.

    Returns:
        None

    Raises:
        None
    """
    with open(filename, "w") as a:
        if isinstance(content, (list,)):
            for ll in content:
                a.write(ll)
                if "\n" not in ll:
                    a.write("\n")
        else:
            a.write(content)

def read_h5_chunk(file_handler, chunk_id=0, chunk_num=1):
    """
    Read a chunk of data from a file handler.

    Args:
        file_handler: The file handler object.
        chunk_id: The ID of the chunk to read. Defaults to 0.
        chunk_num: The total number of chunks. Defaults to 1.

    Returns:
        numpy.ndarray: The read chunk of data.
    """
    if chunk_num == 1:
        # read the whole chunk
        return np.array(file_handler)
    elif chunk_num == -1:
        # read a specific slice
        return np.array(file_handler[chunk_id])
    else:
        # read a chunk
        num_z = int(np.ceil(file_handler.shape[0] / float(chunk_num)))
        return np.array(file_handler[chunk_id * num_z : (chunk_id + 1) * num_z])
    
def read_h5(filename, dataset=None):
    """
    Read data from an HDF5 file.

    Args:
        filename (str): The path to the HDF5 file.
        dataset (str or list, optional): The name or names of the dataset(s) to read. Defaults to None.
        chunk_id (int, optional): The ID of the chunk to read. Defaults to 0.
        chunk_num (int, optional): The total number of chunks. Defaults to 1.

    Returns:
        numpy.ndarray or list: The data from the HDF5 file.

    """
    fid = h5py.File(filename, "r")
    if dataset is None:
        dataset = fid.keys() if sys.version[0] == "2" else list(fid)
    else:
        if not isinstance(dataset, list):
            dataset = [dataset]

    out = [None] * len(dataset)
    for di, d in enumerate(dataset):
        out[di] = np.array(fid[d])
    
    fid.close()
    return out[0] if len(out) == 1 else out

def write_h5(filename, data, dataset="main"):
    """
    Write data to an HDF5 file.

    Args:
        filename (str): The path to the HDF5 file.
        data (numpy.ndarray or list): The data to write.
        dataset (str or list, optional): The name or names of the dataset(s) to create. Defaults to "main".

    Returns:
        None

    Raises:
        None
    """
    fid = h5py.File(filename, "w")
    if isinstance(data, (list,)):
        if not isinstance(dataset, (list,)): 
            num_digit = int(np.floor(np.log10(len(data)))) + 1
            dataset = [('key%0'+str(num_digit)+'d')%x for x in range(len(data))]        
        for i, dd in enumerate(dataset):
            ds = fid.create_dataset(
                dd,
                data[i].shape,
                compression="gzip",
                dtype=data[i].dtype,
            )
            ds[:] = data[i]
    else:
        ds = fid.create_dataset(
            dataset, data.shape, compression="gzip", dtype=data.dtype
        )
        ds[:] = data
    fid.close()


def get_vol_shape(filename, dataset_name=None):
    """
    The function `get_vol_size` returns the size of a dataset in an HDF5 file, or the size of the
    first dataset if no dataset name is provided.

    :param filename: The filename parameter is the name of the HDF5 file that you want to read
    :param dataset_name: The parameter `dataset_name` is an optional argument that specifies the name of
    the dataset within the HDF5 file. If it is not provided, the function will retrieve the first
    dataset in the file and return its shape as the volume size
    :return: the size of the volume as a list.
    """
    volume_size = []
    fid = h5py.File(filename, "r")
    if dataset_name is None:
        dataset_name = fid.keys() if sys.version[0] == "2" else list(fid)
        if len(dataset_name) > 0:
            volume_size = fid[dataset_name[0]].shape
    fid.close()
    return volume_size

def str2dict(input):
    dict ={}
    for x in input.split(','):
        y= x.split(":")
        if y[1].isnumeric():
            y[1] = float(y[1])
        dict[y[0]] = y[1]
    return dict

    
def vol_downsample_chunk(input_file, ratio, output_file=None, chunk_num=1):
    if output_file is None or chunk_num==1:
        vol = read_h5(input_file)
        vol = vol[::ratio[0], ::ratio[1], ::ratio[2]]
        if output_file is None:
            return vol
        else:
            write_h5(output_file, vol)
    else:
        fid_in = h5py.File(input_file, 'r')
        fid_in_data = fid_in[fid_in.keys()[0]]
        fid_out = h5py.File(output_file, "w")
        vol_sz = np.array(fid_in_data.shape) // ratio
        result = fid_out.create_dataset('main', vol_sz, dtype=fid_in_data.dtype)
        
        num_z = int(np.ceil(vol_sz[0] / float(chunk_num)))
        for z in range(chunk_num):
            tmp = read_h5_chunk(fid_in_data, z, chunk_num)[::ratio[0],::ratio[1],::ratio[2]]
            result[z*num_z:(z+1)*num_z] = tmp
            
        fid_in.close()
        fid_out.close()


def seg_to_rgb(seg):
    """
    Convert a segmentation map to an RGB image.

    Args:
        seg (numpy.ndarray): The input segmentation map.

    Returns:
        numpy.ndarray: The RGB image representation of the segmentation map.

    Notes:
        - The function converts a segmentation map to an RGB image, where each unique segment ID is assigned a unique color.
        - The RGB image is represented as a numpy array.
    """
    return np.stack([seg // 65536, seg // 256, seg % 256], axis=2).astype(
        np.uint8
    )


def rgb_to_seg(seg):
    """
    Convert an RGB image to a segmentation map.

    Args:
        seg (numpy.ndarray): The input RGB image.

    Returns:
        numpy.ndarray: The segmentation map.

    Notes:
        - The function converts an RGB image to a segmentation map, where each unique color is assigned a unique segment ID.
        - The segmentation map is represented as a numpy array.
    """
    if seg.ndim == 2:
        return seg
    elif seg.shape[-1] == 1:
        return np.squeeze(seg)
    elif seg.ndim == 3:  # 1 rgb image
        if (seg[:, :, 1] != seg[:, :, 2]).any() or (
            seg[:, :, 0] != seg[:, :, 2]
        ).any():
            return (
                seg[:, :, 0].astype(np.uint32) * 65536
                + seg[:, :, 1].astype(np.uint32) * 256
                + seg[:, :, 2].astype(np.uint32)
            )
        else:  # gray image saved into 3-channel
            return seg[:, :, 0].astype(np.uint32)
    elif seg.ndim == 4:  # n rgb image
        return (
            seg[:, :, :, 0].astype(np.uint32) * 65536
            + seg[:, :, :, 1].astype(np.uint32) * 256
            + seg[:, :, :, 2].astype(np.uint32)
        )