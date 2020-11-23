import logging

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import mesh_utils.image_operations as image_operations
import mesh_utils.masking as masking
from mesh_utils import Tags, TissueTagNames, ValveTagNames
from mesh_utils import image_transforms as it

"""This class takes in an image output from vtk and transforms it to a psuedo-image. This includes:
 - filling in areas of the image to add necessary labels
 - adjusting the tissue values to math their ultrasound properties (most tissue is the same, pericardium is brighter).
 - adding artificial reflectors if desired.
 
 """

DEBUG_WITH_IMAGES = False


class SliceHandler(object):
    def __init__(self, img, opts):
        """
        A class to handle images that get directly output by the vtk processing (export_vtk_slices).
        :param img: image from VTK slice file.
            img can either be directly a PIL image/numpy array or a filename pointing to a png image
        :param opts: params from config file
        """
        # Load Image
        self.label = self._load_img(img)
        self.opts = opts

        # Define params
        self.pseudo_images = list()
        self.tissue_value = opts.tissue_value  # how much to raise tissue values by
        self.pericardium_value = opts.pericardium_value  # how much to raise pericardium values by
        self.inside_val = opts.inside_val  # used to set non tissue value intensity
        self.outside_val = opts.outside_val
        self.image_mode = opts.image_mode
        self.verbose = opts.verbose
        self.show_plots = opts.show_plots

        # initialize transforms
        self.alignment_transforms = opts.alignment_transforms
        self.modifiers = opts.modifiers
        self.custom_transform = it.Compose([it.GrayScale(num_output_channels=1)] + opts.movement_transforms)
        self.pseudo_transform = it.Compose(opts.pseudo_transforms)
        self.post_cone_pseudo = it.Compose(opts.post_cone_pseudo)
        # color_params = opts.color_transforms
        # color_transform = transforms.ColorJitter(**color_params)
        # TODO: apply color transforms to image

        # apply all transforms
        self._apply_transforms()

        if self.show_plots:
            self.show_lbl()
        pass

    @staticmethod
    def _load_img(input_image):
        """ load VTK image from either file or a PIL image or a numpy array."""
        if type(input_image) is str:
            img = np.array(Image.open(input_image))
        elif type(input_image) is Image.Image:
            img = input_image
        elif type(input_image) is np.array:
            img = Image.fromarray(input_image)
        else:
            raise TypeError(
                f"img must either be a filename (str), a PIL Image, or a numpy array but was {type(input_image)}")
        return img

    def _apply_transforms(self):
        # Align image to default setting
        for func, params in self.alignment_transforms:
            self.label = func(self.label, **params)
        # Add blood pools and modifiers
        self._add_blood_pools(self.opts.ignored_blood_pools, debug=False)  # add blood pools to the label images
        self._apply_modifiers(self.modifiers)

        if type(self.label) != Image.Image:
            self.label = Image.fromarray(self.label, mode="L")
            # perform rotation augmentations
        self.label, self.transform_params = self.custom_transform(self.label, {})
        lbl_array = np.array(self.label)
        self.transform_params["lv_area"] = (lbl_array == Tags["lv_blood_pool"]).sum() / lbl_array.size
        self.transform_params["la_area"] = (lbl_array == Tags["la_blood_pool"]).sum() / lbl_array.size

    def _apply_modifiers(self, modifier_fcns):
        """ apply modifier functions to the image specific for each view"""
        for fcn in modifier_fcns:
            self.label = fcn(self.label)

    def _merge_tissue_and_change_values(self, img, include_valves=True):
        """
        First this function merges all tissue tags into a single value. Only elements with
        TissueTags (as definited in mesh_utils.__init__.py) and the pericardium are included in the resulting final
        image. if include_valves is set, ValveTagNames will also be included.

        Second, in original image the values are all right next to each other. This function helps differentiate the
        tissue from the background clearly. It also differentiates between the tissue and the pericardium.

        It makes sense to use a single function here since the same masks are created.

        Will also return a mask for the area inside the heart so that area can be treated differently from the
        area outside later if desired

        """
        tissue_mask = np.zeros_like(img).astype(np.bool)
        for tissue_tag in TissueTagNames:
            tissue_mask += img == Tags[tissue_tag]
        if include_valves:
            for valve_tag in ValveTagNames:
                tissue_mask += img == Tags[valve_tag]
        pericardium_mask = img == Tags["pericardium"]
        outside_mask = img == 0  # all other values will have a tag
        inside_mask = outside_mask + pericardium_mask + tissue_mask == 0
        # make a new image to automatically remove all other tissue types
        img = np.zeros_like(img) + self.inside_val
        img[tissue_mask] = self.tissue_value
        img[pericardium_mask] = self.pericardium_value
        return img, inside_mask

    def generate_pseudo_from_label(self):
        image = np.array(self.label.copy())  # maintain a copy that will serve as the label
        include_valves = np.random.uniform(0, 1.0) < self.opts.include_valves_prob
        image, inside_mask = self._merge_tissue_and_change_values(image, include_valves=include_valves)
        pseudo = image.copy()
        if self.image_mode == "noisy":
            pseudo = image_operations.add_multiplicative_noise(pseudo)
        else:
            raise NotImplementedError("only noisy image mode currently implemented")
        pseudo, self.transform_params = self.pseudo_transform(Image.fromarray(pseudo), self.transform_params)
        pseudo = np.array(pseudo)
        pseudo = image_operations.add_additive_noise(pseudo, downsize_factor=4)
        pseudo = image_operations.add_additive_noise(pseudo, downsize_factor=1)
        pseudo[inside_mask] *= 0.90
        pseudo = image_operations.gaussian_blur_img(pseudo, blur_kernel_size=7)
        pseudo = image_operations.gaussian_blur_img(pseudo, blur_kernel_size=3)
        pseudo = image_operations.resize(pseudo, (self.opts.image_size, self.opts.image_size), Image.BILINEAR)
        pseudo = image_operations.add_additive_noise(pseudo, downsize_factor=4, noise_type="normal")
        pseudo = image_operations.add_additive_noise(pseudo, downsize_factor=2, noise_type="normal")
        pseudo = image_operations.gaussian_blur_img(pseudo, blur_kernel_size=5)
        pseudo = image_operations.set_random_max(pseudo)
        pseudo = image_operations.set_random_min(pseudo)
        return image, pseudo

    def _add_blood_pools(self, ignored_blood_pools, debug=False):
        """ function to add blood pools to the label image """
        for tissue_tag in TissueTagNames:
            chamber = tissue_tag.split('_')[0]
            new_tag_name = chamber + '_blood_pool'
            # ignore some chambers for some views and also only define those for which a value has been defined.
            if chamber.lower() not in ignored_blood_pools and new_tag_name in Tags:
                if self.verbose:
                    logging.info('adding {} to label'.format(new_tag_name))
                self.label = masking.add_blood_pool_mask(self.label, Tags[tissue_tag], Tags[new_tag_name], self.verbose)
                if debug:
                    self.show_lbl()

    # Public functions beneath here
    def show_lbl(self):
        """ Debug function for viewing label """
        plt.imshow(self.label, cmap='viridis')
        plt.show()

    def show_img(self, img, title=None):
        """ Debug function for viewing any image """
        plt.imshow(img)
        if title is None:
            plt.title("image in Slice handler")
        else:
            plt.title(title)
        plt.show()
        plt.close("all")

    def post_adjust_pseudo(self, pseudo, cone):
        # have to remove the padded pix to get back to original
        pseudo = image_operations.resize(pseudo, (cone.shape[0], cone.shape[0]), Image.BILINEAR)
        pseudo *= cone.astype(np.uint8)  # add the cone
        pseudo, self.transform_params = self.post_cone_pseudo(pseudo, self.transform_params)
        if DEBUG_WITH_IMAGES:
            self.show_img(pseudo, title="after post adjustment")
        return pseudo

    def create_pseudo_images(self, opts):
        """ creates images_per_slice pseudo images from this VTK slice"""
        cone_size = opts.image_size
        image, pseudo = self.generate_pseudo_from_label()
        label = np.array(self.label).copy()  # add background to label to accentuate it against the cone
        label[label == 0] = Tags["label_background"]
        cone, cone_params = next(iter(masking.generate_US_cones(1, cone_size)))
        pseudo = self.post_adjust_pseudo(pseudo, cone)
        output = dict(cone_params=cone_params, transform_params=self.transform_params)
        for name, img in dict(normal_img=image, tissue_img=pseudo, label_img=label, cone_img=cone).items():
            if self.verbose:
                shape = img.shape if type(img) == np.ndarray else img.size
                logging.info(f"shape of {name} is {shape[0]}, resizing to {cone.shape[0]} pix")
            resample = Image.BILINEAR if name == "tissue_img" else Image.NEAREST
            # have to remove the padded pix to get back to original
            img = image_operations.resize(img, (cone.shape[0], cone.shape[0]), resample)
            if not opts.remove_cone and name != "label_img":
                img *= cone.astype(np.uint8)
            img = image_operations.crop_to_mask(img, cone)  # strip off any extra pixels
            img = image_operations.resize(img, (opts.image_size, opts.image_size), resample)
            if name != "tissue_img":
                img = img.convert("L")  # more native format
            output[name] = img
        if self.show_plots:
            self.show_img(output["label_img"])
        return output
