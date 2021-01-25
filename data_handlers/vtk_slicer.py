import logging
from pathlib import Path

import numpy as np

import mesh_utils.vtk_functions as vtk_model_api
from data_handlers.heart import Heart
from mesh_utils.utilities import mkdirs
from mesh_utils.vtk_iterators import IteratorSelector
from mesh_utils.vtk_landmarks import LandmarkSelector

"""
A class to handle the slicing of a vtk models.
"""


class VTKSlicer(object):
    def __init__(self, opts):
        input_dir = Path(opts.vtk_data_dir)
        vtk_files = sorted([f for f in input_dir.iterdir() if Path(f).suffix == ".vtk"])
        if opts.max_models > 0:
            max_models = min([len(vtk_files), opts.max_models])
            vtk_files = vtk_files[:max_models]
        print(f"using {len(vtk_files)} vtk files")
        if opts.verbose:
            for f in vtk_files:
                print(f"\t{f}")
        self.vtk_files = vtk_files
        self.opts = opts
        self.iterator_fcn = IteratorSelector(opts.view_name)
        self.landmark_fcn = LandmarkSelector(opts.view_name)
        if opts.save_vtk_slices:
            self.vtk_slice_dir = Path(opts.output_dir) / Path(opts.name) / Path("vtk")
            mkdirs(self.vtk_slice_dir)

    def create_slices(self, vtk_slice_df):
        for c, case in enumerate(self.vtk_files):
            try:
                if self.opts.verbose:
                    print(f"creating slices from {c}: {case}")
                if 0 < self.opts.max_models < c:  # 0 because max_models is set to -1 to avoid stopping
                    logging.info("c > max_models, breaking loop")
                    break
                single_model = Heart(str(case), opts=self.opts)
                origin, normal, landmarks = self.landmark_fcn(single_model)
                # iterator through rotation angles specified in params
                iterator = self.iterator_fcn(origin, normal, landmarks, opts=self.opts)

                # actually do function
                pvs_norm = np.array([0, 1, 0])  # initialize for consistency between runs
                for i, (rotation, angle_dict) in enumerate(iterator):
                    if i != 0:
                        single_model = Heart(str(case), opts=self.opts)  # reinitialize every time to avoid bugs
                    single_model, new_norm = self._create_slice(single_model, rotation, origin, landmarks, pvs_norm)
                    if self.opts.rotation_type == "iterate":
                        pvs_norm = new_norm  # only update for iterate, for random just leave as initialized value

                    if self.opts.save_vtk_slices:
                        single_model.write_vtk(f'_{i:03d}', outname=str(self.vtk_slice_dir / Path(case).stem))
                        dest_file = self.vtk_slice_dir / Path(str(Path(case).stem) + f'_{i:03d}.vtk')
                    else:
                        dest_file = "n/a"
                    vtk_slice_df.add_record(
                        dict(source_file=str(case), rotation=rotation, origin=origin, landmarks=landmarks,
                             aligned=(not self.opts.disable_align_vtk_slices), dest_file=str(dest_file), model_id=c, norm=pvs_norm,
                             rotation_id=i, vtk_img=single_model.get_PIL(), **angle_dict))
            except Exception as e:
                logging.warning(f"models {case} FAILED because {e}. Skipping")
                continue
        vtk_slice_df.save_df()

    def _create_slice(self, _model, rotation, origin, landmarks, preferred_direction=None):
        """
        :param _model: Heart models
        :return: either a single slice models if mode is "normal" or a list of num slice models if mode is "random"
        """
        # print('rotation = {}'.format(rotation))
        # rotation is dict with origin and dir_x and dir_y
        # get normal from cross product of dir_x and dir_y
        normal_r = np.cross(rotation['dir_y'], rotation['dir_x'])
        normal_r /= np.linalg.norm(normal_r)
        # project origin to new plane
        origin_r = vtk_model_api.project_point_to_plane(rotation, origin)
        # extract slice
        print('slice = origin: {}, normal {}'.format(origin_r, normal_r))
        _model.slice_extraction(origin_r, normal_r)
        # project landmarks onto new plane
        landmarks_r = [vtk_model_api.project_point_to_plane(rotation, l) for l in landmarks]
        if not self.opts.disable_align_vtk_slices:
            normal_r = _model.align_slice(landmarks_r[2], landmarks_r[1], landmarks_r[0], preferred_direction)
            _model.rotate(gamma=-90)
        return _model, normal_r


