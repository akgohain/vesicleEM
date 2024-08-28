import os
import numpy as np
import imageio
from scipy.ndimage import zoom

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

def read_tile_volume(filenames, z0p, z1p, y0p, y1p, x0p, x1p, tile_sz, tile_st=None, tile_dtype=np.uint8, tile_type="image", tile_ratio=1, tile_resize_mode=1, tile_border_padding="reflect", tile_blank="", volume_sz=None, zstep=1):
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


