import logging
import os
import shutil

# from json_minify import json_minify
import numpy as np
from PIL import Image


def project3d_to_2d(pt, dx, dy, origin, normal=None):
    shifted_pt = pt - origin
    coord_x = np.dot(dx, shifted_pt)
    coord_y = np.dot(dy, shifted_pt)
    if normal is not None:
        dist_from_plane = np.dot(normal, shifted_pt)
        assert dist_from_plane < 1e-10, f"shifted point is not on plane, distance was {dist_from_plane}"
    return np.array([coord_x, coord_y])


def solve_affine(old_landmarks, new_landmarks):
    x = np.vstack(new_landmarks).T
    y = np.vstack(old_landmarks).T
    difference_in_locs = np.mean(y, axis=1) - np.mean(x, axis=1)
    x += np.expand_dims(difference_in_locs, 1)
    x = np.vstack((x, np.ones(len(old_landmarks))))
    y = np.vstack((y, np.ones(len(new_landmarks))))
    A = np.matmul(y, np.linalg.inv(x))
    return A


def separate_affine(affine):
    """ separate affine into scaling and rotation components
    then construct a new affine matrix which is rotate -> scale -> rotate back
    https://colab.research.google.com/drive/1ImBB-N6P9zlNMCBH9evHD6tjk0dzvy1_#scrollTo=VTa-YKGPEYXK
    """
    # first remove translation (if present)
    affine[:-1, -1] = 0
    from scipy.linalg import polar
    # decompose to rotation (R) and stretch (K) matrices
    R, K = polar(affine)
    # determinant of R should be positive. If not then we need to change the signs
    if np.linalg.det(R) < 0:
        K[:-1, :-1] *= -1
    assert np.allclose(affine, R @ K)  # check that everything worked
    inv_R = R * np.array(
        [[1, -1, 1], [-1, 1, 1], [1, 1, 1]])  # to invert rotation matrix we just need to switch sin components
    scale = R @ K @ inv_R  # rotate, scale, rotate back
    return scale


def add_margin(img, top, right, bottom, left, color=0):
    """ add margins to an image """
    width, height = img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(img.mode, (new_width, new_height), color)
    result.paste(img, (left, top))
    return result


def shift_to_center(img: Image.Image, background=0) -> Image.Image:
    """ shift the contents of an image (img) to the center. background describes the current background pixel val """
    loc_image = Image.fromarray((np.array(img) != background))
    bbox = loc_image.getbbox()
    width, height = img.size
    new_left = int((width - bbox[2] + bbox[0]) / 2)
    new_top = int((height - bbox[3] + bbox[1]) / 2)
    new_left = new_left - bbox[0]  # need to give coordinate of top left of image not top left of non-background
    new_top = new_top - bbox[1]
    result = Image.new(img.mode, (width, height), background)
    result.paste(img, (new_left, new_top))
    return result


def save_images(df, output_dir, save_label=True):
    def _save(im, name):
        im = im.convert('L')
        im.save(name)

    for i, row in df.iterrows():
        # output_name = Path(output_dir).joinpath(fi.filename.parts[-2]+'_'+fi.filename.stem+'_'+str(fi.ind)+'.png')
        output_name = os.path.join(output_dir, row.gen_savename)
        _save(row.img, output_name)
        if save_label:
            label_name = os.path.join(output_dir, row.lbl_savename)
            _save(row.label_img, label_name)


def remove_img_cols(df):
    cols = list(df.columns)
    cols.remove('vtk_img')
    cols.remove('img')
    return cols


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


def parse_dir(dir, ending='.png'):
    files = os.listdir(dir)
    files = [os.path.join(dir, f) for f in files if ending in f]
    return files


def clean_dict_for_json(d):
    """
    prepares a dictionary for json by:
        - converting numpy arrays to lists
    """
    for k, v in d.items():
        if type(v) is np.ndarray:
            d[k] = v.tolist()
    return d


def check_angle(a):
    """ given an angle in radians convert it to the range -pi to pi"""
    while a > np.pi:
        a -= 2*np.pi
    while a < -np.pi:
        a += 2*np.pi
    return a


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def rmdir(path):
    """  Removes a directory
    """
    if os.path.exists(path):
        shutil.rmtree(path)
