import os
import random
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes
from skimage.transform import resize
from torchvision import transforms
from torchvision.transforms import functional as F

from mesh_utils import Tags, masking
from mesh_utils.image_operations import random_pad, pad, crop, brightening, shadowing, show_img, generate_heatmap, \
    gaussian_blur_img

DEBUG_WITH_IMAGES = False


class TransformBase(ABC):
    """ Helper class used to define the format of the __call__ method so that it always includes a state variable """
    @abstractmethod
    def __call__(self, img, state: dict):
        pass


class CenterAndRotateLV(TransformBase):
    """ Center an attribute and then rotate around it.
    Particularly useful for the LV which should be centered in the field of view, but the location of the apex will
    depend on the amount of rotation. This automatically accomplishes that.
    if shift_apex_to_top is set then this function will not shift X to the center, but Y such that the apex is near the
    cone origin.
    """

    def __init__(self, degrees: tuple, apex_pos: tuple, shift_apex_to_top=False):
        """
        :param degrees: tuple of (min, max) degrees of rotation counter-clockwise, will be sampled uniformly
        :param apex_pos: tuple of (min, max) apex positions in terms of percentage of image size.
        :param shift_apex_to_top: (bool) if true will shift Apex to the top of the image
        """
        assert len(degrees) == len(apex_pos) == 2, "degrees and apex_pos must be 2 tuples of (min, max)"
        self.degrees = degrees
        self.shift_apex_to_top = shift_apex_to_top
        self.attribute_name = "lv_myocardium"  # This transform only works for LV
        self.apex_pos = apex_pos

    @staticmethod
    def get_params(attribute_name, degrees, img):
        image_locs = np.where(np.array(img) == Tags[attribute_name])
        com = np.mean(image_locs, axis=1)
        apex = (np.min(image_locs[0]), image_locs[1][np.argmin(image_locs[0])])  # rough apex estimate is highest pt
        # translation = (np.array(img.size)/2 - com).astype(int)
        angle = np.random.uniform(degrees[0], degrees[1])
        return tuple(com.astype(int)), apex, angle

    def get_apex_loc(self, img, apex):
        # based on observation apex should be in this range as a percentage of the img
        apex_dist_from_origin = np.random.uniform(int(self.apex_pos[0] * img.height),
                                                  int(self.apex_pos[1] * img.height))
        origin = np.array([0, img.width / 2])
        current_dist = np.linalg.norm(apex - origin)
        apex_movement = (current_dist - apex_dist_from_origin) / current_dist
        apex_shift = apex_movement * (apex - origin)
        return apex_shift

    @staticmethod
    def rotate(p, origin=(0, 0), degrees=0):
        """ rotate a point to check new apex """
        angle = np.deg2rad(degrees)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        o = np.atleast_2d(origin)
        p = np.atleast_2d(p)
        return np.squeeze((R @ (p.T - o.T) + o.T).T)

    @staticmethod
    def plot_img(img, rotation=None, pts=None):
        title = "rotating around attribute"
        if rotation is not None:
            title += f"- rotation={rotation}"
        plt.imshow(img, cmap="viridis")
        colors = ["red", "blue", "green", "yellow", "cyan"]
        if pts is not None:
            for k, pt in pts.items():
                plt.plot(pt[1], pt[0], 'o', color=colors.pop(), label=k)
            plt.legend()
        plt.title(title)
        plt.show()
        plt.close("all")

    def __call__(self, img, state: dict):
        """
        :param img: (PIL Image) to be transformed
        :return:
        """
        an = self.attribute_name
        assert (np.array(img) == Tags[an]).sum() > 0, f"no points for {an} found in img"
        if type(img) == np.ndarray:
            img = Image.fromarray(img, mode='L')
        center, apex, angle = self.get_params(an, self.degrees, img)
        origin = np.array([img.height / 2, img.width / 2]).astype(int)
        shift = np.array([0, center[1] - origin[1]])  # dont shift rows to not cut image features

        # shift to center: affine: (x, y) = (ax + by + c, dx + ey + f) so we just set c and f to shift amount
        img = img.transform(img.size, Image.AFFINE, (1, 0, shift[1], 0, 1, shift[0]), fillcolor=0)
        apex, center = np.array(apex) - shift, np.array(center) - shift

        # rotate desired amount
        # BUG with torchvision 0.5 and PIL https://github.com/pytorch/vision/issues/1759
        img = F.rotate(img, angle, center=tuple(center))
        apex = self.rotate(np.array(apex), origin, angle)
        if DEBUG_WITH_IMAGES:
            cone_img = (np.array(img) + 50) * next(iter(masking.generate_US_cones(1, img.height)))[0]  # cone for debug
            self.plot_img(img, rotation=angle, pts=dict(center=center, apex=apex))

        # Now shift apex to within the range we want
        apex_shift = self.get_apex_loc(img, apex)
        img = img.transform(img.size, Image.AFFINE, (1, 0, apex_shift[1], 0, 1, apex_shift[0]), fillcolor=0)

        if DEBUG_WITH_IMAGES:
            apex, center = apex - apex_shift, center - apex_shift
            cone_img = (np.array(img) + 50) * next(iter(masking.generate_US_cones(1, img.height)))[0]  # cone for debug
            self.plot_img(img, rotation=angle, pts=dict(center=center, apex=apex))
        return img, state


class MoveAttributeWithinCone(TransformBase):
    """Class to move a given attribute within the ultrasound sector cone.
    For now this class does not check against an actual ultrasound cone, but based on
    an assumption of approximately where an ultrasound cone will be. This leaves room for some
    natural variability in the actual placement as an augmentation.
    """

    def __init__(self, ultrasound_cone_angle=np.pi / 5, min_loc=0.8, max_loc=0.9):
        self.ultrasound_cone_angle = ultrasound_cone_angle
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.right_to_left_ratio = 1.25  # angle on left side is multiplied by this to be more lenient on the left pt

    @staticmethod
    def calc_max_pts_on_mask(img, mask_value, pt=(0, 0)):
        """
        This function calculates the maximum angle between a segmentation mask and a point.
        It does this for both the left and right sides of the segmentation.
        Note that this function currently makes a few assumptions:
            - the point is above the segmentation in the image
            - the point of maximum angle will also be the point that is farthest from the bottom middle of the
              segmentation mask + top_middle of segmentation mask and is in the top 25% if the image
        :param img: image containing the segmentation mask
        :param mask_value: value of the desired segmentation within the image
        :param pt: (row, col) tuple location of the point (default 0,0)
        :return: (max_angle_left, max_angle_right) tuple describing the range of angle values
        """

        # helper functions
        def distance_from_pt(arr, pt):
            return np.sqrt((arr[0] - pt[0]) ** 2 + (arr[1] - pt[1]) ** 2)

        def angle_from_pt(arr, pt):
            return np.arctan2(arr[0] - pt[0], arr[1] - pt[1])

        def norm(arr):
            return (arr - arr.min()) / arr.ptp()

        def norm_inv(arr):  # for when the min should become the max
            return abs((arr - arr.max()) / arr.ptp())

        # left in for easy debugging
        def plot_found_pts():
            plt.imshow(img, cmap="viridis")
            plt.plot(com[1], max_vals[0], 'ro', label="bot center")
            plt.plot(com[1], com[0], 'purple', marker='o', label="com")
            # plt.plot(max_right_pt_angle[1], max_right_pt_angle[0], 'g', marker='o')
            plt.plot(max_right_pt[1], max_right_pt[0], 'ko', label="max right")
            plt.plot(max_left_pt[1], max_left_pt[0], 'bo', label="max left")
            # plt.plot(max_left_pt_angle[1], max_left_pt_angle[0], 'lightblue', marker='o')
            plt.plot(pt[1], pt[0], 'cyan', marker='o', label="origin")
            plt.plot(segmentation_coords_c[np.argmin(segmentation_coords_r)], min_vals[0], marker='o', color="salmon",
                     label="apex?")
            plt.legend(loc=3)
            plt.title("found points for moving attribute in cone")
            plt.show()
            plt.close("all")

        img = np.array(img)
        segmentation_coords_r, segmentation_coords_c = np.where(img == mask_value)
        segmentation_coords = np.stack([segmentation_coords_r, segmentation_coords_c])
        # find some useful parts of the segmentation mask: center of mass, min, max
        com = np.mean(segmentation_coords, axis=1)
        min_vals = np.min(segmentation_coords, axis=1)
        max_vals = np.max(segmentation_coords, axis=1)
        row_cutoff = min_vals[0] + .25 * (max_vals[0] - min_vals[0])
        # separate out the coords in the top right/left of the image
        top_right_coords = segmentation_coords[:,
                           (segmentation_coords[0] < row_cutoff) & (segmentation_coords[1] > com[1])]
        top_left_coords = segmentation_coords[:,
                          (segmentation_coords[0] < row_cutoff) & (segmentation_coords[1] < com[1])]
        # get distance from every point in mask to the bottom middle of the mask
        right_dist_from_top = distance_from_pt(top_right_coords, (max_vals[0], com[1]))
        right_dist_from_bottom = distance_from_pt(top_right_coords, (min_vals[0], com[1]))
        left_dist_from_top = distance_from_pt(top_left_coords, (max_vals[0], com[1]))
        left_dist_from_bottom = distance_from_pt(top_left_coords, (min_vals[0], com[1]))

        angles_right = angle_from_pt(top_right_coords, pt)
        angles_left = angle_from_pt(top_left_coords, pt)

        distance_right = norm(right_dist_from_top) + norm(right_dist_from_bottom) + norm_inv(angles_right)
        distance_left = norm(left_dist_from_top) + norm(left_dist_from_bottom) + norm(angles_left)

        max_right_pt = top_right_coords[:, np.argmax(distance_right)]
        max_left_pt = top_left_coords[:, np.argmax(distance_left)]

        # max_right_pt_angle = top_right_coords[:, np.argmin(angles_right)]
        # max_left_pt_angle = top_left_coords[:, np.argmax(angles_left)]
        if DEBUG_WITH_IMAGES:
            plot_found_pts()
        return (max_left_pt, max_right_pt), max_vals[0]

    def find_shift_amount(self, img, desired_angle_range, max_points, origin):
        """find amount to shift the segmentation mask to meet the angle requirements"""

        def closest_pt_on_line(a, b, p):
            """ a and b define the line, p is the point"""
            ap = p - a
            ab = b - a
            return a + np.dot(ap, ab) / np.dot(ab, ab) * ab

        def check_right_val_in_cone(difference, cone_angle):
            """ check if right val has passed the right vector"""
            angle = np.arctan2(*difference)
            if np.pi / 2 - cone_angle > angle > -np.pi / 2 - cone_angle:
                return True
            return False

        def check_left_val_in_cone(difference, cone_angle):
            """ check if left val has passed the right vector"""
            # return True  # disabling for now
            angle = np.arctan2(*difference)
            if angle > np.pi / 2 + cone_angle or angle < -np.pi / 2 + cone_angle:
                return True
            return False

        def plot_found():
            plt.imshow(img, cmap="viridis")
            plt.plot([origin[1], origin[1] + vector_right_side_of_sector[1]],
                     [origin[0], origin[0] + vector_right_side_of_sector[0]], 'r-')
            plt.plot(closest_pt_on_right_vector[1], closest_pt_on_right_vector[0], 'ro')
            plt.plot([origin[1], origin[1] + vector_left_side_of_sector[1]],
                     [origin[0], origin[0] + vector_left_side_of_sector[0]], 'r-')
            plt.plot(closest_pt_on_left_vector[1], closest_pt_on_left_vector[0], 'ro')
            plt.plot(max_points[1][1], max_points[1][0], 'go')
            plt.plot(max_points[0][1], max_points[0][0], 'bo')
            plt.title("found points in MoveAttributeWithinCone")
            plt.show()
            plt.close("all")

        max_points = np.array(max_points).astype(float)
        # row, col ordering for all coords
        vector_right_side_of_sector = img.width / 2 * np.array(
            (np.cos(desired_angle_range), np.sin(desired_angle_range)))
        closest_pt_on_right_vector = closest_pt_on_line(origin, origin + vector_right_side_of_sector, max_points[1])
        right_difference = closest_pt_on_right_vector - max_points[1]

        # now do the same for left side
        vector_left_side_of_sector = img.width / 2 * np.array(
            (np.cos(desired_angle_range * self.right_to_left_ratio),
             -np.sin(desired_angle_range * self.right_to_left_ratio)))
        closest_pt_on_left_vector = closest_pt_on_line(origin, origin + vector_left_side_of_sector, max_points[0])
        left_difference = closest_pt_on_left_vector - max_points[0]
        if DEBUG_WITH_IMAGES:
            plot_found()

        # decide on movements
        right_inside = check_right_val_in_cone(right_difference, desired_angle_range)
        left_inside = check_left_val_in_cone(left_difference, desired_angle_range)
        total_difference = np.zeros(2)
        try_ct = 0
        while not right_inside or not left_inside:
            center_col = np.mean(max_points[:, 1])
            if try_ct < 1:
                diff = origin[1] - center_col
                max_points[:, 1] += diff
                total_difference[1] += diff
            max_points[:, 0] += max([left_difference[0], right_difference[0], img.height * .01])
            total_difference[0] += max([left_difference[0], right_difference[0], img.height * .01])
            if not right_inside and left_inside:
                max_points[:, 1] += right_difference[1]
                total_difference[1] += right_difference[1]
            if not left_inside and right_inside:
                max_points[:, 1] += left_difference[1] * .5
                total_difference[1] += left_difference[1] * .5
            # if DEBUG_WITH_IMAGES:
            #     plot_found()
            closest_pt_on_right_vector = closest_pt_on_line(origin, origin + vector_right_side_of_sector, max_points[1])
            closest_pt_on_left_vector = closest_pt_on_line(origin, origin + vector_left_side_of_sector, max_points[0])
            right_difference = closest_pt_on_right_vector - max_points[1]
            left_difference = closest_pt_on_left_vector - max_points[0]
            right_inside = check_right_val_in_cone(right_difference, desired_angle_range)
            left_inside = check_left_val_in_cone(left_difference, desired_angle_range)
            try_ct += 1
        return total_difference

    def __call__(self, img, state: dict):
        assert "attribute_name" in state, "RandomChoiceAttribute must be called first"
        an = state["attribute_name"]
        av = Tags[an]
        if type(img) == np.ndarray:
            img = Image.fromarray(img)
        origin = (0, img.width / 2)  # r, c
        max_points, max_row = self.calc_max_pts_on_mask(img, av, pt=origin)
        movement = self.find_shift_amount(img, self.ultrasound_cone_angle, max_points, origin)
        # avoid cropping out features - TODO: do computations with bounding box instead, which will simplify!
        if max_row + movement[0] > self.max_loc * img.height:
            pad_high = self.max_loc - self.min_loc
            pad_amount = int(
                np.random.uniform(0, pad_high) * img.height + (max_row + movement[0]) - self.max_loc * img.height)
            original_rows = img.height
            img = pad(img, pad_amount, locs=("bottom",))
            if "bottom_rows" in state:
                num_rows_to_include = min([pad_amount, state["bottom_rows"].shape[0]])
                img[original_rows:original_rows + num_rows_to_include] = state["bottom_rows"][:num_rows_to_include]
            img = Image.fromarray(pad(img, int(pad_amount / 2), locs=("left", "right")))  # to not change aspect ratio
        if max_row + movement[0] < self.min_loc * img.height:
            final_pos = np.random.uniform(self.min_loc, self.max_loc)
            crop_amount = final_pos * img.height - (max_row + movement[0])
            img = Image.fromarray(crop(img, int(crop_amount), locs="bottom", ))
            img = Image.fromarray(crop(img, int(crop_amount / 2), locs=("left", "right")))  # to not change aspect ratio
        # perform translation
        if DEBUG_WITH_IMAGES:
            plt.imshow(img, cmap="viridis")
            plt.title("before shifting")
            plt.show()
        img = img.transform(img.size, Image.AFFINE, (1, 0, -movement[1], 0, 1, -movement[0]), fillcolor=0)
        if DEBUG_WITH_IMAGES:
            plt.imshow(img, cmap="viridis")
            plt.title("after shifting")
            plt.show()
            plt.close("all")
        if "bottom_rows" in state:
            # additional logic needed here if bottom rows is needed by a future transform
            state.pop("bottom_rows")
        return img, state


class CropToAttribute(TransformBase):
    """Crop to an attribute. """

    def __init__(self):
        pass

    @staticmethod
    def get_params(img, name):
        locs_r, locs_c = np.where(np.array(img) == Tags[name])
        assert locs_r.shape[0] > 0, f"attribute {name} not found in image"
        min_r, max_r = locs_r.min(), locs_r.max()
        com_c = np.mean(locs_c)

        # choose cropping region based on rows
        diff_r = locs_r.ptp()
        top = max(0, min_r - .1 * diff_r)
        bottom = max_r + .1 * diff_r
        bottom += 50 * (name == "lv_myocardium")  # add some extra for myo
        diff_r = bottom - top
        left = max(0, com_c - int(diff_r / 2))
        height = min([diff_r, img.height - top, img.width - left])
        width = height
        # logging.info(f"crop_params are {top, left, height, width}")
        return int(top), int(left), int(height), int(width)

    @staticmethod
    def show_img(img):
        plt.imshow(img, cmap="viridis")
        plt.title("image in crop to attribute")
        plt.show()
        plt.close("all")

    def __call__(self, img, state: dict):
        assert "attribute_name" in state, "RandomChoiceAttribute must be called first"
        an = state["attribute_name"]
        if type(img) == np.ndarray:
            img = Image.fromarray(img)
        top, left, height, width = self.get_params(img, an)
        # cropping out the bottom rows here is a problem, becuase they might be "uncropped" later
        if top + height < img.height:
            state["bottom_rows"] = np.array(img)[top + height:, left:left + width]
            pass
        img = F.crop(img, top, left, height, width)
        if DEBUG_WITH_IMAGES:
            self.show_img(img)
        return img, state


class RandomChoiceAttribute(TransformBase):
    """randomly choose an attribute from a provided list. This transform only updates the state dict and not the image
    """

    def __init__(self, names, probs):
        """
        names and probs should be iterables of the same length where each entry in probs matches the probability of
        selecting the corresponding attribute. probs should also add to 1 (+/- .1)
        """
        assert len(names) == len(probs), "names and probs must be the same length"
        assert .99 < np.sum(np.array(probs)) < 1.01, "probs must sum to 1 +/- .1"
        assert all([a in Tags for a in names]), "all names must be a Tag"
        self.names = names
        self.probs = probs

    def __call__(self, img, state: dict):
        an = np.random.choice(np.array(self.names), p=np.array(self.probs))
        assert (np.array(img) == Tags[an]).sum() > 0, f"no points for {an} found in img"
        state["attribute_name"] = an
        return img, state


class IterateTransforms(TransformBase):
    """ iterate through a list of transforms a given number of times"""

    def __init__(self, input_transforms, n_iters):
        """
        :param input_transforms: list of transforms to iterate
        :param n_iters: number of times ot iterate
        """
        self.transform = Compose(input_transforms)
        self.niters = n_iters

    def __call__(self, img, state: dict):
        for n in range(self.niters):
            img, state = self.transform(img, state)
        return img, state


class RandomSpotting(TransformBase):
    """This class will apply a gaussian to make pieces of the image darker/brigther than others"""

    def __init__(self, loc_min: tuple, loc_mean: tuple, loc_std: tuple, brightness_mean: float,
                 brightness_std: float, size_mean: float, size_std: float, row_col_ratio: float = 1.0,
                 brightness_prob=0.5):
        """
        :param brightness_prob: (float) probability (between 0 and 1) of being brightening (else shadowing)
        :param loc_min: (row, col) min center for gaussian, tuple of length 2
        :param loc_mean: (row, col) average center for gaussian, tuple of length 2
        :param loc_std: (row, col) std for gaussian location, tuple of length 2
        :param brightness_mean: (float) mean brightness for gaussian
        :param brightness_std: (float) std of brightness for gaussian
        :param size_mean: (float) mean size for gaussian
        :param size_std: (float) std of size for gaussian
        :param row_col_ratio: (float) ratio of widths between row and col. 1 means same, >1 = rows > cols
        """
        TransformBase.__init__(self)
        self.brightness_prob = brightness_prob
        assert (0. <= self.brightness_prob and brightness_prob <= 1.), "brightness probability must be between 0 and 1"
        self.loc_min = loc_min
        self.loc_mean = loc_mean
        self.loc_std = loc_std
        self.brightness_mean = brightness_mean
        self.brightness_std = brightness_std
        self.size_mean = size_mean
        self.size_std = size_std
        self.row_col_ratio = row_col_ratio

    @staticmethod
    def show_img(img, pt, op=''):
        plt.imshow(img, cmap="viridis")
        plt.plot(pt[0], pt[1], 'ro')
        plt.title(f"image after random spotting with op = {op}")
        plt.show()
        plt.close("all")

    def __call__(self, img, state: dict):
        if type(img) == Image.Image:
            img = np.array(img)
        operation = np.random.choice([brightening, shadowing], p=[self.brightness_prob, 1 - self.brightness_prob])
        loc_row = min(img.shape[0], max(self.loc_min[0], np.random.normal(self.loc_mean[0], self.loc_std[0])))
        loc_col = min(img.shape[1], max(self.loc_min[1], np.random.normal(self.loc_mean[1], self.loc_std[1])))
        brightness = min(1, max(0, np.random.normal(self.brightness_mean, self.brightness_std)))
        sigma = max(1, 4 * img.shape[0] + np.random.normal(self.size_mean, self.size_std))
        img = operation(img, (loc_row, loc_col), brightness, sigma, self.row_col_ratio)
        if DEBUG_WITH_IMAGES:
            self.show_img(img, (loc_col, loc_row), operation.__name__)
        return Image.fromarray(img), state


class ShadowConeOrigin(TransformBase):
    def __init__(self, size_mean, size_std):
        self.size_mean = size_mean
        self.size_std = size_std

    @staticmethod
    def show_img(img, title=None):
        plt.imshow(img, cmap="viridis")
        title = "Image after shadowing origin" if title is None else title
        plt.title(title)
        plt.show()
        plt.close("all")

    def __call__(self, img, state: dict):
        if type(img) == Image.Image:
            img = np.array(img)
        sigma = np.random.normal(self.size_mean, self.size_std)
        hmap = generate_heatmap(img.shape, (0, int(img.shape[1] / 2)), sigma=100000)
        hmap /= hmap.max()
        r, c = img.shape
        noise = resize(abs(np.random.normal(25, 20, size=(int(r / 8), int(c / 8)))), (r, c))
        noise2 = resize(abs(np.random.normal(25, 20, size=(int(r / 8), int(c / 8)))), (r, c))
        img -= img * hmap
        img += noise * hmap
        img -= noise2 * hmap  # dont increase brightness at origin too much
        if DEBUG_WITH_IMAGES:
            self.show_img(hmap, "hmap for origin blurring")
            self.show_img(img)
        return Image.fromarray(img), state


class HorizontalSin(TransformBase):
    """Selectively reduces horizontal rows of image using a sin wave"""

    def __init__(self, freq_range: tuple = (.5, 2), min_range: tuple = (0, .5), max_range: tuple = (1, 1.5)):
        self.freq_range = freq_range
        self.min_range = min_range
        self.max_range = max_range

    def __call__(self, img, state: dict):
        img = np.array(img)
        rows = img.shape[0]
        freq = np.random.uniform(self.freq_range[0], self.freq_range[1])
        max_val = np.random.uniform(self.max_range[0], self.max_range[1])
        min_val = np.random.uniform(self.min_range[0], self.min_range[1])
        r_vals = np.arange(rows)
        wave = 1 + np.sin(2 * np.pi / (freq * rows) * r_vals) * (max_val - min_val) / 2 + min_val
        return Image.fromarray(img), state


class HistogramMatcher(TransformBase):
    """ Matches the histogram of the image to another source image"""

    def __init__(self, source_img_dir, mean=.1937, std=0.22):
        """ default mean and std were set from camus. Must set a reference dir containing images to match with """
        self.files = [os.path.join(source_img_dir, f) for f in os.listdir(source_img_dir) if ".png" in f]
        assert len(self.files) > 0, f"no files found in {source_img_dir}"
        self.image_mean = mean
        self.image_std = std

    @staticmethod
    def cdf(im):
        ''' computes the CDF of an image im as 2D numpy ndarray '''
        from skimage.exposure import cumulative_distribution
        c, b = cumulative_distribution(im)
        # pad the beginning and ending pixels and their CDF values
        c = np.insert(c, 0, [0] * int(b[0]))
        c = np.append(c, [1] * int(255 - b[-1]))
        return c

    @staticmethod
    def hist_matching(c, c_t, im):
        '''
        c: CDF of input image computed with the function cdf()
        c_t: CDF of template image computed with the function cdf()
        im: input image as 2D numpy ndarray
        returns the modified pixel values
        '''
        pixels = np.arange(256)
        # find closest pixel-matches corresponding to the CDF of the input image, given the value of the CDF H of
        # the template image at the corresponding pixels, s.t. c_t = H(pixels) <=> pixels = H-1(c_t)
        new_pixels = np.interp(c, c_t, pixels)
        im = (np.reshape(new_pixels[im.ravel()], im.shape)).astype(np.uint8)
        return im

    @staticmethod
    def get_cone_mask(im):
        mask = im > 0
        mask = binary_fill_holes(mask)
        return mask

    def __call__(self, img, state: dict):
        img = np.array(img).astype(np.int)

        cutoff = self.image_mean + .2 * self.image_std  # super bright images ruin transformation so exclude
        if "attribute_name" in state and state["attribute_name"] == "pericardium":
            cutoff -= .2 * self.image_std
        im_mean = cutoff
        while im_mean >= cutoff:
            f = np.random.choice(self.files)
            im_template = np.array(Image.open(f)).astype(np.int)
            im_mean = im_template.mean() / 255.
        if True:  # DEBUG_WITH_IMAGES:
            show_img(img, "histogram matcher original")
            show_img(im_template, "histogram matcher template")
        im_mask = self.get_cone_mask(img)
        template_mask = self.get_cone_mask(im_template)
        cdf_img = self.cdf(img[im_mask])
        cdf_match = self.cdf(im_template[template_mask])
        img[img > 0] = self.hist_matching(cdf_img, cdf_match, img[im_mask])
        # hist matching tends to create patches of pixel with 0 intensity... smooth these
        img = gaussian_blur_img(img.astype(np.uint8), blur_kernel_size=1)
        img[~im_mask] = 0
        if True:  # DEBUG_WITH_IMAGES:
            show_img(im_mask, "image mask")
            show_img(template_mask, "template mask")
            show_img(img, "histogram matcher final")
        return Image.fromarray(img.astype(np.uint8)), state


class EraseWrapper(TransformBase):
    """ Wrapper around random eraser """
    def __init__(self):
        self.to_tensor = transforms.ToTensor()
        self.random_eraser = transforms.RandomErasing()
        self.to_pil = transforms.ToPILImage()

    def __call__(self, img, state: dict):
        img = self.to_tensor(img)
        img = self.random_eraser(img)
        img = self.to_pil(img)
        return img, state


class SqueezeHorizontalByAttribute(TransformBase):
    """ squeezes more for full 4ch then for lv focused
    real_mean and std across the entire dataset were mean = 1.237, std=.103
    I just guessed based on those values for the real value
    """

    def __init__(self, wr):
        self.means = dict(
            lv_myocardium=1.2,
            pericardium=wr, )
        self.stds = dict(
            lv_myocardium=.1,
            pericardium=.1, )

    @staticmethod
    def plot_img(img, param):
        plt.imshow(img, cmap="viridis")
        plt.title(f"after squeezing horizontal with factor {param}")
        plt.show()

    def __call__(self, img, state: dict):
        assert "attribute_name" in state, "must call RandomChoiceAttributeFirst"
        an = state["attribute_name"]
        mean, std = self.means[an], self.stds[an]
        param = np.random.normal(mean, std)
        pad_pix = int((img.width * param - img.width) / 2)
        img = random_pad(np.array(img), pad_pix, locs=("left", "right")).convert("L")
        if "bottom_rows" in state:
            pass
            state["bottom_rows"] = np.array(random_pad(state["bottom_rows"], pad_pix, locs=("left", "right")))
        if DEBUG_WITH_IMAGES:
            self.plot_img(img, param)
        if type(img) == np.ndarray:
            img = Image.fromarray(img)
        return img, state


class Compose(TransformBase):
    """ copy of compose for segmentation dataset"""

    def __init__(self, input_transforms: list):
        self.input_transforms = input_transforms

    def __call__(self, img, state: dict):
        for t in self.input_transforms:
            img, state = t(img, state)
        return img, state


class RandomApply(TransformBase):
    """Apply randomly a list of transformations with a given probability

    Args:
        transforms (list or tuple): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        self.transform = Compose(transforms)
        self.p = p

    def __call__(self, img, state: dict):
        if self.p < random.random():
            return img, state
        else:
            return self.transform(img, state)


class GrayScale(TransformBase, transforms.Grayscale):
    """ Wrapper around Grayscale transform"""
    def __init__(self, num_output_channels=1):
        transforms.Grayscale.__init__(self, num_output_channels)

    def __call__(self, img, state: dict):
        img = transforms.Grayscale.__call__(self, img)
        return img, state
