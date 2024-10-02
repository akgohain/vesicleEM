import os, sys
import numpy as np
import imageio
from scipy.ndimage import zoom
import yaml
import h5py

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



        
def get_filenames(folder_name, output_name):            
    fns = [x[x.rfind('/')+1:x.rfind('.')] for x in glob(f'{folder_name}/*.png')]
    write_txt(f'{folder_name}/{output_name}', fns)        



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
            dataset = list(dataset)

    out = [None] * len(dataset)
    for di, d in enumerate(dataset):
        out[di] = np.array(fid[d])

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
