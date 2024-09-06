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

def read_vast_seg(fn):
    a = open(fn).readlines()
    # remove comments
    st_id = 0
    while a[st_id][0] in ["%", "\\"]:
        st_id += 1
    
    st_id -= 1
    # remove segment name
    out = np.zeros((len(a) - st_id - 1, 24), dtype=int)
    name = [None] * (len(a) - st_id - 1)
    for i in range(st_id + 1, len(a)):
        out[i - st_id - 1] = np.array(
            [int(x) for x in a[i][: a[i].find('"')].split(" ") if len(x) > 0]
        )
        name[i - st_id - 1] = a[i][a[i].find('"') + 1 : a[i].rfind('"')]
    return out, name

def vast_meta_relabel(
    fn, kw_bad=["bad", "del"], kw_nm=["merge"], do_print=False
):
    # if there is meta data
    print("load meta")
    dd, nn = read_vast_seg(fn)
    rl = np.arange(1 + dd.shape[0], dtype=np.uint16)
    pid = np.unique(dd[:, 13])
    if do_print:
        print(
            ",".join(
                [
                    nn[x]
                    for x in np.where(np.in1d(dd[:, 0], pid))[0]
                    if "Imported Segment" not in nn[x]
                ]
            )
        )

    pid_b = []
    if len(kw_bad) > 0:
        # delete seg id
        pid_b = [
            i
            for i, x in enumerate(nn)
            if max([y in x.lower() for y in kw_bad])
        ]
        bid = np.where(np.in1d(dd[:, 13], pid_b))[0]
        bid = np.hstack([pid_b, bid])
        if len(bid) > 0:
            rl[bid] = 0
        print("found %d bad" % (len(bid)))

    # not to merge
    kw_nm += ["background"]
    pid_nm = [
        i for i, x in enumerate(nn) if max([y in x.lower() for y in kw_nm])
    ]
    # pid: all are children of background
    pid_nm = np.hstack([pid_nm, pid_b])
    print("apply meta")
    # hierarchical
    for p in pid[np.in1d(pid, pid_nm, invert=True)]:
        rl[dd[dd[:, 13] == p, 0]] = p
    # consolidate root labels
    for u in np.unique(rl[np.where(rl[rl] != rl)[0]]):
        u0 = u
        while u0 != rl[u0]:
            rl[rl == u0] = rl[u0]
            u0 = rl[u0]
        print(u, rl)
    return rl

def get_tile_name(pattern, row=None, column=None):
    """
    Generate the tile name based on the pattern and indices.

    Args:
        pattern (str): The pattern for the tile name.
        row (int, optional): The row index. Defaults to None.
        column (int, optional): The column index. Defaults to None.

    Returns:
        str: The generated tile name.
    """    
    
    if "%" in pattern:
        return pattern % (row, column)
    elif "{" in pattern:
        return pattern.format(row=row, column=column)
    else:
        return pattern


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
    if seg.ndim == 2 or seg.shape[-1] == 1:
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
            return seg[:, :, 0]
    elif seg.ndim == 4:  # n rgb image
        return (
            seg[:, :, :, 0].astype(np.uint32) * 65536
            + seg[:, :, :, 1].astype(np.uint32) * 256
            + seg[:, :, :, 2].astype(np.uint32)
        )

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

def read_tile_volume(filenames, z0p, z1p, y0p, y1p, x0p, x1p, tile_sz, tile_st=None, 
                     tile_dtype=np.uint8, tile_type="image", tile_ratio=1, 
                     tile_resize_mode=1, tile_border_padding="reflect", tile_blank="", 
                     volume_sz=None, zstep=1):
    """
    Read and assemble a volume from a set of tiled images.

    Args:
        filenames (list): The list of file names or patterns for the tiled images.
        z0p (int): The starting index of the z-axis.
        z1p (int): The ending index of the z-axis.
        y0p (int): The starting index of the y-axis.
        y1p (int): The ending index of the y-axis.
        x0p (int): The starting index of the x-axis.
        x1p (int): The ending index of the x-axis.
        tile_sz (int or list): The size of each tile in pixels. If an integer is provided, the same size is used for both dimensions.
        tile_st (list, optional): The starting index of the tiles. Defaults to [0, 0].
        tile_dtype (numpy.dtype, optional): The data type of the tiles. Defaults to np.uint8.
        tile_type (str, optional): "image" or "seg"
        tile_ratio (float or list, optional): The scaling factor for resizing the tiles. If a float is provided, the same ratio is used for both dimensions. Defaults to 1.
        tile_resize_mode (int, optional): The interpolation mode for resizing the tiles. Defaults to 1.
        tile_seg (bool, optional): Whether the tiles represent segmentation maps. Defaults to False.
        tile_border_padding (str, optional): The padding mode for tiles at the boundary. Defaults to "reflect".
        tile_blank (str, optional): The value or pattern to fill empty tiles. Defaults to "".
        volume_sz (list, optional): The size of the volume in each dimension. Defaults to None.
        zstep (int, optional): The step size for the z-axis. Defaults to 1.

    Returns:
        numpy.ndarray: The assembled volume.

    Notes:        
        - The tiles are specified by file names or patterns in the `filenames` parameter.
        - The volume is constructed by arranging the tiles according to their indices.
        - The size of each tile is specified by the `tile_sz` parameter.
        - The tiles can be resized using the `tile_ratio` parameter.
        - The tiles can be interpolated using the `tile_resize_mode` parameter.
        - The tiles can represent either grayscale images or segmentation maps.
        - The volume can be padded at the boundary using the `tile_bd` parameter.
        - Empty tiles can be filled with a value or pattern using the `tile_blank` parameter.
        - The size of the volume can be specified using the `volume_sz` parameter.
        - The step size for the z-axis can be adjusted using the `zstep` parameter.
    """
    if tile_st is None:
        tile_st = [0, 0]
    if not isinstance(tile_sz, (list,)):
        tile_sz = [tile_sz, tile_sz]
    if not isinstance(tile_ratio, (list,)):
        tile_ratio = [tile_ratio, tile_ratio]
    # [row,col]
    # no padding at the boundary
    # st: starting index 0 or 1

    bd = None
    if volume_sz is not None:
        bd = [
            max(-z0p, 0),
            max(0, z1p - volume_sz[0]),
            max(-y0p, 0),
            max(0, y1p - volume_sz[1]),
            max(-x0p, 0),
            max(0, x1p - volume_sz[2]),
        ]
        z0, y0, x0 = max(z0p, 0), max(y0p, 0), max(x0p, 0)
        z1, y1, x1 = (
            min(z1p, volume_sz[0]),
            min(y1p, volume_sz[1]),
            min(x1p, volume_sz[2]),
        )
    else:
        z0, y0, x0, z1, y1, x1 = z0p, y0p, x0p, z1p, y1p, x1p

    result = np.zeros(
        ((z1 - z0 + zstep - 1) // zstep, y1 - y0, x1 - x0), tile_dtype
    )
    c0 = x0 // tile_sz[1]  # floor
    c1 = (x1 + tile_sz[1] - 1) // tile_sz[1]  # ceil
    r0 = y0 // tile_sz[0]
    r1 = (y1 + tile_sz[0] - 1) // tile_sz[0]
    z1 = min(len(filenames) - 1, z1)
    for i, z in enumerate(range(z0, z1, zstep)):
        pattern = filenames[z]
        for row in range(r0, r1):
            for column in range(c0, c1):
                filename = get_tile_name(pattern, row + tile_st[0], column + tile_st[1])
                if os.path.exists(filename):
                    patch = read_image(filename, tile_type, tile_ratio, tile_resize_mode)
                    # exception: last tile may not have the right size
                    psz = patch.shape
                    xp0 = column * tile_sz[1]
                    xp1 = min(xp0 + psz[1], (column + 1) * tile_sz[1])
                    yp0 = row * tile_sz[0]
                    yp1 = min(yp0 + psz[0], (row + 1) * tile_sz[0])
                    x0a = max(x0, xp0)
                    x1a = min(x1, xp1)
                    y0a = max(y0, yp0)
                    y1a = min(y1, yp1)
                    result[i, y0a - y0 : y1a - y0, x0a - x0 : x1a - x0] = (
                            patch[y0a - yp0 : y1a - yp0, x0a - xp0 : x1a - xp0]
                    )                    
                else:
                    print(f"Non-exist: {filename}")
    # blank case
    if tile_blank != "":
        blank_st = 0
        blank_lt = result.shape[0] - 1
        while blank_st <= blank_lt and not np.any(result[blank_st] > 0):
            blank_st += 1
        if blank_st == blank_lt + 1:
            print("!! This volume is all 0 !!")
        else:
            result[:blank_st] = result[blank_st : blank_st + 1]
            while blank_lt >= blank_st and not np.any(result[blank_lt] > 0):
                blank_lt -= 1
            result[blank_lt:] = result[blank_lt - 1 : blank_lt]
            for z in range(blank_st + 1, blank_lt):
                if not np.any(result[z] > 0):
                    result[z] = result[z - 1]

    # boundary case
    if bd is not None and max(bd) > 0:
        result = np.pad(
            result, ((bd[0], bd[1]), (bd[2], bd[3]), (bd[4], bd[5])), tile_border_padding
        )
    return result

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
        
def get_filenames(folder_name, output_name):            
    fns = [x[x.rfind('/')+1:x.rfind('.')] for x in glob(f'{folder_name}/*.png')]
    write_txt(f'{folder_name}/{output_name}', fns)        


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
    if seg.ndim == 2 or seg.shape[-1] == 1:
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
            return seg[:, :, 0]
    elif seg.ndim == 4:  # n rgb image
        return (
            seg[:, :, :, 0].astype(np.uint32) * 65536
            + seg[:, :, :, 1].astype(np.uint32) * 256
            + seg[:, :, :, 2].astype(np.uint32)
        )



def compute_bbox(seg, do_count=False):
    """
    Compute the bounding box of a binary segmentation.

    Args:
        seg (numpy.ndarray): The binary segmentation.
        do_count (bool, optional): Whether to compute the count of foreground pixels. Defaults to False.

    Returns:
        list: The bounding box of the foreground segment in the format [y0, y1, x0, x1, count (optional)].

    Notes:
        - The input segmentation can have any dimension.
        - If the segmentation is empty (no foreground pixels), None is returned.
        - The bounding box is computed as the minimum and maximum coordinates along each dimension that contain foreground pixels.
        - If `do_count` is True, the count of foreground pixels is included in the output.
    """    
    if not seg.any():
        return None

    out = []
    pix_nonzero = np.where(seg > 0)
    for i in range(seg.ndim):
        out += [pix_nonzero[i].min(), pix_nonzero[i].max()]

    if do_count:
        out += [len(pix_nonzero[0])]
    return out


def compute_bbox_all(seg, do_count=False, uid=None):
    """
    Compute the bounding boxes of segments in a segmentation map.

    Args:
        seg (numpy.ndarray): The input segmentation map.
        do_count (bool, optional): Whether to compute the segment counts. Defaults to False.
        uid (numpy.ndarray, optional): The segment IDs to compute the bounding boxes for. Defaults to None.

    Returns:
        numpy.ndarray: An array containing the bounding boxes of the segments.

    Raises:
        ValueError: If the input volume is not 2D or 3D.

    Notes:
        - The function computes the bounding boxes of segments in a segmentation map.
        - The bounding boxes represent the minimum and maximum coordinates of each segment in the map.
        - The function can compute the segment counts if `do_count` is set to True.
        - The bounding boxes are returned as an array.
    """    
    if seg.ndim == 2:
        return compute_bbox_all_2d(seg, do_count, uid)
    elif seg.ndim == 3:
        return compute_bbox_all_3d(seg, do_count, uid)
    else:
        raise "input volume should be either 2D or 3D" 

def compute_bbox_all_2d(seg, do_count=False, uid=None):
    """
    Compute the bounding boxes of 2D instance segmentation.

    Args:
        seg (numpy.ndarray): The 2D instance segmentation.
        do_count (bool, optional): Whether to compute the count of each instance. Defaults to False.
        uid (numpy.ndarray, optional): The unique identifier for each instance. Defaults to None.

    Returns:
        numpy.ndarray: The computed bounding boxes of the instances.

    Notes:
        - The input segmentation should have dimensions HxW, where H is the height and W is the width.
        - Each row in the output represents an instance and contains the following information:
            - seg id: The ID of the instance.
            - bounding box: The coordinates of the bounding box in the format [ymin, ymax, xmin, xmax].
            - count (optional): The count of pixels belonging to the instance.
        - If the `uid` argument is not provided, the unique identifiers are automatically determined from the segmentation.
        - Instances with no pixels are excluded from the output.
    """        
    sz = seg.shape
    assert len(sz) == 2
    if uid is None:
        uid = np.unique(seg)
        uid = uid[uid > 0]
    if len(uid) == 0:
        return None
    uid_max = uid.max()
    out = np.zeros((1 + int(uid_max), 5 + do_count), dtype=np.uint32)
    out[:, 0] = np.arange(out.shape[0])
    out[:, 1] = sz[0]
    out[:, 3] = sz[1]
    # for each row
    rids = np.where((seg > 0).sum(axis=1) > 0)[0]
    for rid in rids:
        sid = np.unique(seg[rid])
        sid = sid[(sid > 0) * (sid <= uid_max)]
        out[sid, 1] = np.minimum(out[sid, 1], rid)
        out[sid, 2] = np.maximum(out[sid, 2], rid)
    cids = np.where((seg > 0).sum(axis=0) > 0)[0]
    for cid in cids:
        sid = np.unique(seg[:, cid])
        sid = sid[(sid > 0) * (sid <= uid_max)]
        out[sid, 3] = np.minimum(out[sid, 3], cid)
        out[sid, 4] = np.maximum(out[sid, 4], cid)

    if do_count:
        seg_ui, seg_uc = np.unique(seg, return_counts=True)
        out[seg_ui, -1] = seg_uc
    return out[uid]


def compute_bbox_all_3d(seg, do_count=False, uid=None):
    """
    Compute the bounding boxes of 3D instance segmentation.

    Args:
        seg (numpy.ndarray): The 3D instance segmentation.
        do_count (bool, optional): Whether to compute the count of each instance. Defaults to False.
        uid (numpy.ndarray, optional): The unique identifier for each instance. Defaults to None.

    Returns:
        numpy.ndarray: The computed bounding boxes of the instances.

    Notes:
        - Each row in the output represents an instance and contains the following information:
            - seg id: The ID of the instance.
            - bounding box: The coordinates of the bounding box in the format [ymin, ymax, xmin, xmax, zmin, zmax].
            - count (optional): The count of voxels belonging to the instance.
        - The output only includes instances with valid bounding boxes.
    """

    sz = seg.shape
    assert len(sz) == 3, "Input segment should have 3 dimensions"
    if uid is None:
        uid = seg
    uid_max = int(uid.max())
    out = np.zeros((1 + uid_max, 7 + do_count), dtype=np.int32)
    out[:, 0] = np.arange(out.shape[0])
    out[:, 1] = sz[0]
    out[:, 2] = -1
    out[:, 3] = sz[1]
    out[:, 4] = -1
    out[:, 5] = sz[2]
    out[:, 6] = -1

    # for each slice
    zids = np.where((seg > 0).sum(axis=1).sum(axis=1) > 0)[0]
    for zid in zids:
        sid = np.unique(seg[zid])
        sid = sid[(sid > 0) * (sid <= uid_max)]
        out[sid, 1] = np.minimum(out[sid, 1], zid)
        out[sid, 2] = np.maximum(out[sid, 2], zid)

    # for each row
    rids = np.where((seg > 0).sum(axis=0).sum(axis=1) > 0)[0]
    for rid in rids:
        sid = np.unique(seg[:, rid])
        sid = sid[(sid > 0) * (sid <= uid_max)]
        out[sid, 3] = np.minimum(out[sid, 3], rid)
        out[sid, 4] = np.maximum(out[sid, 4], rid)

    # for each col
    cids = np.where((seg > 0).sum(axis=0).sum(axis=0) > 0)[0]
    for cid in cids:
        sid = np.unique(seg[:, :, cid])
        sid = sid[(sid > 0) * (sid <= uid_max)]
        out[sid, 5] = np.minimum(out[sid, 5], cid)
        out[sid, 6] = np.maximum(out[sid, 6], cid)

    if do_count:
        seg_ui, seg_uc = np.unique(seg, return_counts=True)
        out[seg_ui[seg_ui <= uid_max], -1] = seg_uc[seg_ui <= uid_max]

    return out[np.all(out != -1, axis=-1)].astype(np.uint32)


def merge_bbox_two_matrices(bbox_matrix_a, bbox_matrix_b):
    """
    Merge two matrices of bounding boxes.

    Args:
        bbox_matrix_a (numpy.ndarray): The first matrix of bounding boxes.
        bbox_matrix_b (numpy.ndarray): The second matrix of bounding boxes.

    Returns:
        numpy.ndarray: The merged matrix of bounding boxes.

    Notes:
        - Each matrix should have dimensions Nx(D+1), where N is the number of bounding boxes and D is the number of dimensions (4 or 5).
        - The first column of each matrix represents the index of the bounding box.
        - The remaining columns represent the coordinates of the bounding box in the format [ymin, ymax, xmin, xmax].
        - If there are bounding boxes with the same index in both matrices, they are merged using the `merge_bbox` function.
        - Bounding boxes that do not have an intersection are concatenated in the output matrix.
    """
    if bbox_matrix_a is None:
        return bbox_matrix_b 
    if bbox_matrix_b is None:
        return bbox_matrix_a 
    bbox_a_id,  bbox_b_id = bbox_matrix_a[:, 0], bbox_matrix_b[:, 0]
    intersect_id = np.in1d(bbox_a_id, bbox_b_id)
    if intersect_id.sum() == 0:
        # no intersection
        return np.vstack([bbox_matrix_a, bbox_matrix_b])
    for i in np.where(intersect_id)[0]:
        bbox_a = bbox_matrix_a[i, 1:]
        bbox_b_index = bbox_b_id == bbox_a_id[i]
        bbox_b = bbox_matrix_b[bbox_b_index, 1:][0]
        bbox_matrix_b[bbox_b_index, 1:] = merge_bbox(bbox_a, bbox_b)
    return np.vstack([bbox_matrix_a[np.logical_not(intersect_id)], bbox_matrix_b])


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