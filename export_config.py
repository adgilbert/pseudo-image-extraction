import argparse
import os

import numpy as np
import torchvision.transforms.functional as f
from box import Box

import mesh_utils.masking as masking
from mesh_utils import image_transforms as it
from mesh_utils.utilities import mkdirs


class BaseOptions:
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and
    models class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initialized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument("--exclude_vtk", default=False, action="store_true",
                            help="run vtk part of population. If false, loads vtk from dataframe.")
        parser.add_argument("--exclude_image", default=False, action="store_true",
                            help="run image creation part of population. If false, loads vtk from dataframe.")
        parser.add_argument("--exclude_export", default=False, action="store_true",
                            help="run png export part of pipeline")
        parser.add_argument("--vtk_data_dir", required=True, help="location of data.")
        parser.add_argument("--output_dir", required=True, help="where to save dataframe files and pngs")
        parser.add_argument("--name", required=True, help="name for this run")

        # vtk parameters
        parser.add_argument("-v", "--view_name", required=True,
                            choices=["plax", "a4ch", "a2ch", "aplax", "apical", "a4ch_viewing"],
                            help="type of view to extract from vtk models. For more on each choice see vtk_iterators.py")
        parser.add_argument("--max_models", type=int, default=np.inf,
                            help="only use first X models, negative uses all.")
        parser.add_argument("--num_slices", type=int, default=3,
                            help="number of slices to extract from each vtk models.")
        parser.add_argument("--save_vtk_slices", default=False, action="store_true",
                            help="use this flag to save each vtk slice as it's own file for later viewing in paraview.")
        parser.add_argument("--align_vtk_slices", default=True, action="store_false",
                            help="use this flag to disable alignment of slices to view them relative to each other.")
        parser.add_argument("--rotation_type", type=str, default="random", choices=["iterate", "random"],
                            help="rotation type. random samples randomly, iterate samples linearly")
        parser.add_argument("--x_axis_rotation_param", type=float, default=0.05,
                            help="x axis rotation parameter in radians (will be multiplied by pi). Standard deviation if rotation type is random or limit if rotation type is iterate. ")
        parser.add_argument("--y_axis_rotation_param", type=float, default=0.05,
                            help="y axis rotation parameter in radians (will be multiplied by pi). Standard deviation if rotation type is random or limit if rotation type is iterate. ")

        # image parameters
        parser.add_argument("--images_per_slice", type=int, default=1, help="images created per vtk slice")
        parser.add_argument("--image_size", type=int, default=256, help="final size of image")
        parser.add_argument("--image_mode", type=str, default="tissue", choices=["tissue", "noisy", "none"],
                            help="type of image to output. tissue adds reflectors, noisy adds noise, and none does nothing")
        parser.add_argument("--include_inverse_images", default=False, action="store_true",
                            help="set to include saving of inverse images")
        parser.add_argument("--hist_matching_reference_dir", default=None, type=str,
                            help="If histogram matching filter is used this option must be set to specify where the matching images should be drawn from")

        # export parameters
        parser.add_argument("--remove_cone", default=False, action="store_true",
                            help="set to disable applying an ultrasound cone")
        parser.add_argument("--original_sized_images", default=False, action="store_true",
                            help="set to resize images according to original model size")

        # other parameters
        parser.add_argument("--verbose", default=False, action="store_true", help="additional output information")
        parser.add_argument("--show_plots", default=False, action="store_true", help="additional output information")

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional models-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in models and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            self.dataset_specific_params = DatasetSpecificParams(parser)
            parser = self.initialize(parser)
            self.parser = parser

        # save and return the parser
        opts = self.parser.parse_args()
        return opts, self.dataset_specific_params

    @staticmethod
    def get_base_params():
        # These options are not included in the argparser because they shouldn't really change between runs.
        base_params = dict(
            tissue_value=175,  # value of tissue before speckle processing
            pericardium_value=250,  # value of pericardium before speckle processing
            inside_val=50,  # value of non tissue before processing
            outside_val=100,
        )
        return base_params

    @staticmethod
    def get_view_specific_params(opts, dataset_specific_params):
        """ Define parameters that change based on the desired view here. """
        # TODO: this implementation forces user to define these params - rethink this
        lv_p, peri_p = dataset_specific_params.lv_focus_percentage, 1 - dataset_specific_params.lv_focus_percentage
        wr = dataset_specific_params.width_ratio

        # settings specific to each view
        view_specific_params = dict(
            # for now all of these functions take only the image as a parameter and return an image.
            # if this changes than need to change this structure
            # modifiers modify the vtk slice as needed
            # TODO: define these as classes with a __call__ function like movement transforms
            modifiers=dict(
                default=(),
                plax=(masking.increase_attribute_thickness, masking.filter_plax_pericardium,),
                apical=(),
                a4ch=(masking.increase_attribute_thickness,),
                a2ch=(masking.increase_attribute_thickness,),
            ),

            # alignment will align the slice to the default alignment first
            # values should be a list of tuples of (functions, function parameters)
            # TODO: define these as classes with a __call__ function like movement transforms
            alignment_transforms=dict(
                plax=[],
                apical=[],
                a4ch=[(f.pad, dict(padding=60)),
                      (f.affine, dict(angle=140, translate=(-20, 50), scale=1.0, shear=0)), ],
                a4ch_viewing=[(f.pad, dict(padding=160)),
                              (f.affine, dict(angle=140, translate=(-20, 50), scale=1.0, shear=0)), ],
                a2ch=[(f.pad, dict(padding=60)),
                      (f.affine, dict(angle=50, translate=(-10, 30), scale=1.0, shear=0)), ],
            ),
            # transforms apply a list of custom augmentation transforms
            # style is list of tuples with (class, class parameters) as above)
            # all classes should have the type of torchvision augmentation classes
            # classes are applied before label/image split (i.e. blood pools not available).
            movement_transforms=dict(
                default=[],
                a4ch=[
                    it.RandomChoiceAttribute(names=("lv_myocardium", "pericardium"), probs=(lv_p, peri_p)),
                    it.CenterAndRotateLV(degrees=(-15, 15), apex_pos=(.05, .15), shift_apex_to_top=True),
                    it.CropToAttribute(),
                    it.SqueezeHorizontalByAttribute(wr),
                    it.MoveAttributeWithinCone(), ],
                a2ch=[
                    it.RandomChoiceAttribute(names=("lv_myocardium", "pericardium"), probs=(lv_p, peri_p)),
                    it.CenterAndRotateLV(degrees=(-15, 15), apex_pos=(.05, .15), shift_apex_to_top=True),
                    it.CropToAttribute(),
                    it.SqueezeHorizontalByAttribute(wr),
                    it.MoveAttributeWithinCone(), ],
            ),
            # These are appearance transforms that are applied to only the pseudo image (so no movement should be
            # included). All transforms should follow the torchvision transforms style
            pseudo_transforms=dict(
                default=(),
                plax=(),
                apical=(),
                a4ch=[it.ShadowConeOrigin(100000, 10000), it.IterateTransforms([it.RandomApply([
                    it.RandomSpotting(
                        loc_min=(100, 0), loc_mean=(400, 600), loc_std=(100, 100), brightness_mean=.3,
                        brightness_std=.05, size_mean=4000, size_std=500, row_col_ratio=1.)])], n_iters=3)],
                a2ch=[it.ShadowConeOrigin(100000, 10000), it.IterateTransforms([it.RandomApply([
                    it.RandomSpotting(
                        loc_min=(100, 0), loc_mean=(400, 600), loc_std=(100, 100), brightness_mean=.3,
                        brightness_std=.05, size_mean=4000, size_std=500, row_col_ratio=1.)])], n_iters=3)],
            ),
            post_cone_pseudo=dict(
                default=(),
                a4ch=(),
            ),
            # probability of including valves as tissue in pseudo images
            include_valves_prob=dict(
                default=1,
                a4ch=.4,
                a2ch=.4,
            ),
            # regions that can be ignored when adding blood pools
            ignored_blood_pools=dict(
                default=[],
                a2ch=["ra", "rv"], # RA and RV are not in A2CH images
            )
            # define others as needed here. format should be param name as dict with entry for each view (and "default")
        )
        return view_specific_params

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opts, dataset_specific_params = self.gather_options()
        print_options(self.parser, opts)
        opts = dict(**vars(opts))
        if not opts["exclude_image"]:
            base_params = self.get_base_params()
            view_specific_params = self.get_view_specific_params(opts, dataset_specific_params)
            opts.update(**base_params)
            # add view specific params
            for k, v in view_specific_params.items():
                assert (opts[
                            "view_name"] in v or "default" in v), f"Define modifiers for view {opts['view_name']} or default values in export_config.py"
                if opts["view_name"] in v:
                    opts.update({k: v[opts["view_name"]]})
                else:
                    opts.update({k: v["default"]})
        self.opts = opts
        return Box(self.opts)


class DatasetSpecificParams:
    """ Define parameters that should change based on the dataset """
    def __init__(self, parser):
        # Set defaults:
        self.lv_focus_percentage = 0.5  # what percentage of images should be lv focused)
        self.width_ratio = 1.0  # how much to shrink width of images (1 = leave the same)

        # adjust as necessary
        parser.add_argument("--dataset", default=None, type=str, help="set dataset specific parameters")
        args, _ = parser.parse_known_args()
        if args.dataset is not None:
            eval(f"self.set_{args.dataset}_params()")

    def set_pd_params(self):
        self.width_ratio = 1.3

    def set_camus_params(self):
        self.width_ratio = 1.2
        self.lv_focus_percentage = 0.6

    def set_echonet_params(self):
        self.min_feature_loc = 0.6
        self.lv_focus_percentage = 0.6
        self.width_ratio = 1.1

    def set_perionly_params(self):
        """ dataset to only focus on the entire heart"""
        self.lv_focus_percentage = 0.0

    def set_lvonly_params(self):
        """ dataset to only focus on the lv"""
        self.lv_focus_percentage = 1.0

def print_options(parser, opt):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    if "name" in vars(opt).keys():
        expr_dir = os.path.join(opt.output_dir, opt.name)
        mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')