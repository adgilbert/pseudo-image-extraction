import logging

import cv2
import numpy as np
from PIL import Image

from mesh_utils import Tags
from mesh_utils.image_operations import crop_to_mask, resize

__all__ = ["get_full_mask", "generate_US_cones"]


def get_circle_mask(xx, yy, origin, radius):
    """ Gets a circular mask at the specified origin and radius.
     xx and yy define the coordinate grid for the mask so origin should be in reference to these two.
     """
    mask = np.zeros(xx.shape)
    mask[np.sqrt((yy - origin[0]) ** 2 + (xx - origin[1]) ** 2) < radius] = 1
    return mask


def get_angle_mask(xx, yy, origin, width, tilt):
    """ Gets a triangular mask of the specified origin, width, and tilt.
    xx and yy provide the grid for the mask and the parameters should be listed in reference to this.
    """
    mask = np.zeros(xx.shape)
    half_width = width/2
    lower_bound = -np.pi/2 - half_width + tilt
    upper_bound = -np.pi/2 + half_width + tilt
    mask[(np.arctan2(origin[0]-yy, xx-origin[1]) < upper_bound) &
         (np.arctan2(origin[0]-yy, xx-origin[1]) > lower_bound)] = 1
    return mask


def get_full_mask(inp_size, origin, radius, width, tilt, ax=None):
    """ calls both circle mask and angle mask to get a mask of an ultrasound cone.
    if ax is not None than the mask will be plotted.
    returns a mask with 0 being the region outside the cone and 1 the region inside it
    """
    assert radius < inp_size, "radius should be smaller than image"
    max_side_pt = max([radius * np.sin(width / 2 + tilt), radius * np.sin(-width / 2 - tilt)])
    diff = max_side_pt - inp_size / 2
    if diff > 0:
        inp_size = int(np.ceil(max_side_pt * 2))
        origin[1] += int(diff)
    xx, yy = np.meshgrid(range(inp_size), range(inp_size))
    circle_mask = get_circle_mask(xx, yy, origin, radius)
    angle_mask = get_angle_mask(xx, yy, origin, width, tilt)
    full_mask = circle_mask + angle_mask
    if ax is not None:
        ax.imshow(full_mask)
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
    full_mask /= 2
    full_mask = full_mask.astype(int)
    return full_mask


def check_mask_bounds(img, r, c, radius):
        if r < np.ceil(radius):
            r += np.ceil(radius)
        if r >= img.shape[0] - np.ceil(radius):
            r -= np.ceil(radius)
        if c < np.ceil(radius):
            c += np.ceil(radius)
        if c >= img.shape[1] - np.ceil(radius):
            c -= np.ceil(radius)
        return r, c


def generate_US_cones(num_cones, inp_size=1024):
    """ randomly samples radius, width, tilt, origin, and rotations to generate a ultrasound mask. """
    radii = np.random.uniform(.93 * inp_size, 1.0 * inp_size, size=num_cones)  # 976
    widths = np.random.normal(6 * np.pi / 12, np.pi / 30, size=num_cones)  # TODO make this adjustable with view
    # tilts = np.random.normal(0, np.pi / 32, size=num_cones)
    tilts = np.zeros(num_cones)
    origin_ys = np.ones(shape=(num_cones,))
    origin_xs = inp_size / 2 * np.ones(shape=(num_cones,))
    # origin_ys = np.random.normal(16, 4, size=num_cones)
    # origin_xs = np.random.normal(512, 16, size=num_cones)
    # rotations = np.random.normal(-90, 1, size=num_cones)
    for radius, width, tilt, origin_y, origin_x in zip(radii, widths, tilts, origin_ys, origin_xs):
        params = dict(radius=radius, width=width, tilt=tilt, origin_x=origin_x, origin_y=origin_y)
        mask = get_full_mask(inp_size, [origin_y, origin_x], radius, width, tilt)
        mask = crop_to_mask(mask, mask)  # crop to mask and reshape
        mask = resize(mask, (inp_size, inp_size))
        yield np.array(mask), params


def check_pixel_count(image, tissue_tag):
    rows, cols = image.shape
    pixel_count = np.sum(1.0 * image == tissue_tag)
    return pixel_count/(rows*cols)  # return as a percentage or image size


def get_max_pixels_to_change(image, tissue_tag):
    tissue_locs = np.where(image==tissue_tag)
    min_loc = np.min(tissue_locs, axis=1)
    max_loc = np.max(tissue_locs, axis=1)
    max_area = np.sum((max_loc - min_loc)**2)
    return max_area


def get_center_of_mass(image, tissue_tag):
    """ get center of mass of given tissue type"""
    tissue_locs = np.where(image == tissue_tag)
    center_of_mass = np.mean(tissue_locs, axis=1)
    com_x, com_y = int(center_of_mass[1]), int(center_of_mass[0])
    return com_x, com_y


def get_center_by_contouring(image, tissue_tag, starting_guess):
    contour_image = np.array(image == tissue_tag).astype(np.uint8)
    contours, _ = cv2.findContours(contour_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) <= 1:  # will always find 1 contour (background) but this isnt the one we want
        return None
    # discard largest contour since this is the full background and that one might mess things up
    biggest, _ = get_largest_contour(contours)
    contours = [c for c in contours if c is not biggest]
    # now find which contours center is closest
    min_dist, closest = None, None
    rows, cols = image.shape
    for c in contours:
        try:
            center = get_contour_center(c)
        except ZeroDivisionError:
            logging.info("zero division error in finding contour center, skipping this contour")
            continue  # contour is too small
        if cv2.contourArea(c) < .001 * rows * cols:
            continue  # contour is too small
        dist = np.linalg.norm(np.array(center) - starting_guess)
        if min_dist is None or dist < min_dist:
            closest = center
            min_dist = dist
    return closest


def add_blood_pool_mask(image, surrounding_tissue_tag, new_tissue_tag, verbose=False):
    """
    creates a mask where a blood pool can be filled in with a new tissue_tag.
    :param image: vtk image
    :param surrounding_tissue_tag: the tissue type surrounding the blood pool.
    This tissue type does not have to completely enclose it, but the blood pool should be completely enclosed by
    some tissue. This center of mass of this tissue type used as the initialization point
    :param new_tissue_tag: the tag to assign to the new tissue
    :param verbose: print out extra info
    :return:
    """
    if type(image) == Image.Image:
        image = np.array(image)
    assert type(image) == np.ndarray, f"image should be numpy array, but was {type(image)}"
    assert len(np.squeeze(image).shape) == 2, "only built to work with single channel images"
    rows, cols = image.shape
    tag_name = next(iter([k for k, v in Tags.items() if v == surrounding_tissue_tag]))  # should only be one
    if check_pixel_count(image, surrounding_tissue_tag) < 0.001:
        logging.warning("<.5% points with tag {} found in image, skipping...".format(tag_name))
        return image
    init_x, init_y = get_center_of_mass(image, surrounding_tissue_tag)
    if image[init_y, init_x] != Tags["background"]:
        logging.warning(
            "center of mass method didn't work to find starting point for {}. trying contouring".format(tag_name))
        center = get_center_by_contouring(image, 0, np.array([init_x, init_y]))  # get contours of background
        if center is None:
            logging.warning("contour centering also failed to find starting point for {}, skipping".format(tag_name))
            return image
        init_x, init_y = center
        if image[init_y, init_x] != Tags["background"]:
            logging.warning("contour centering also failed to find starting point for {}, skipping".format(tag_name))
            # image[init_y-20:init_y+20, init_x-20:init_x+20] = 255
            return image
    max_count = get_max_pixels_to_change(image, surrounding_tissue_tag)
    image_copy = image.copy()
    _ = cv2.floodFill(image_copy.reshape((rows, cols, 1)), None, (init_x, init_y), (new_tissue_tag,))
    new_count = np.sum(1.0*(image_copy==new_tissue_tag))
    if new_count < max_count:
        if verbose:
            logging.info("added {} pixels to image for {}".format(new_count, tag_name))
        return image_copy.reshape(rows, cols)
    else:
        logging.warning("centering found wrong contour, aborting")
        return image


def filter_locs(self, locs, type):
    if type == "bottom":
        rows = self.img.shape[0]
        cols = self.img.shape[1]
        mask = ((locs[0] > 0.50 * rows) & ((locs[0] > 0.67*rows) | (locs[1] < 0.5*cols)))
        locs = (locs[0][mask], locs[1][mask])
        return locs


def filter_plax_pericardium(image):
    """ For the plax view the pericardium should only be seen at the bottom of the image"""
    locs = np.where(image == Tags["pericardium"])
    rows, cols = image.shape
    mask = ((locs[0] < 0.50 * rows) | ((locs[0] < 0.67 * rows) & (locs[1] > 0.5 * cols)))
    locs = (locs[0][mask], locs[1][mask])
    image[locs[0], locs[1]] = Tags["background"]
    return image


def get_largest_contour(contours):
    """ find the largest contour among all contours """
    biggest_area, biggest_contour = 0, None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > biggest_area:
            biggest_area = area
            biggest_contour = contour
    return biggest_contour, biggest_area


def get_contour_center(contour):
    """ find the center of a contour using the moments function
    https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=moments#moments
    """
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY


def increase_attribute_thickness(image, tag_name="pericardium", additional_width=10):
    """ The models include a pericardium. This function increases the thickness.
    This function adds width 10 pixels to the pericardium because more than this starts to bleed into the other tissue
    types. However, it can be called multiple times to continually increase the thickness.
    """

    def get_bordered(contour_image, width):
        """ modified version of:
        https://stackoverflow.com/questions/51541754/how-to-change-the-thickness-of-the-edge-contour-of-an-image
        """
        bg = np.zeros(contour_image.shape)
        contours, _ = cv2.findContours(contour_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            biggest_contour, area = get_largest_contour(contours)
            # parameters of drawContours are image, contours, contourIdx, color, thickness...
            return cv2.drawContours(bg, [biggest_contour], 0, (255, 255, 255), width).astype(bool)
        return bg

    assert tag_name in Tags, f"{tag_name} not found in Tags"
    peri = np.zeros_like(image)
    peri[image == Tags[tag_name]] = Tags[tag_name]
    new_peri = get_bordered(peri, additional_width)
    peri_combined = peri.astype(np.bool) | new_peri.astype(np.bool)
    image[peri_combined] = Tags[tag_name]
    return image
