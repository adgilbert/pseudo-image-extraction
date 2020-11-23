import numpy as np

from data_handlers.heart import calculate_plane_normal
from mesh_utils.image_operations import apply_rotations
from mesh_utils.vtk_functions import get_centers, get_apex


class LandmarkSelector:
    def __init__(self, view_name):
        if not "_" + view_name in dir(self):
            raise KeyError(f"must define iterator function with name {'_' + view_name} in utils/vtk_landmarks.py")
        self.landmark_fcn = eval(f"self._{view_name}")

    def __call__(self, _model):
        return self.landmark_fcn(_model)

    @staticmethod
    def _a4ch(_model):
        """
        Get the landmarks for a apical 4ch slice from a models. S
        Landmarks are:
            - Apex
            - mitral valve center
            - tricuspid valve center
        """
        valve_centers = get_centers(_model, (7, 8))
        apex = get_apex(_model)

        landmarks = (*valve_centers, apex)
        normal = calculate_plane_normal(*landmarks)
        origin = np.mean(np.array(landmarks), axis=0)

        return origin, normal, landmarks

    def _a4ch_viewing(self, _model):
        """ same as a4ch """
        return self._a4ch(_model)

    def _a2ch(self, _model):
        """
        Get the landmarks for a apical 2ch slice from a models. S
        Landmarks are:
            - Apex
            - mitral valve center
            - tricuspid valve center
        """
        origin, _, (mitral_valve, tricuspid_valve, apex) = self._a4ch(_model)
        long_axis = mitral_valve - apex  # long axis (y-axis)
        short_axis = mitral_valve - tricuspid_valve  # short axis (x-axis)

        # set up to use existing infrastructure
        rotations = [dict(name="get_a2ch", axis=long_axis, center=mitral_valve, angles=[-1 * np.deg2rad(70)])]
        base_plane = dict(origin=origin, dir_x=short_axis, dir_y=long_axis)
        a2ch_axis = next(apply_rotations(base_plane, rotations, gen_type='zip'))

        # now the new short axis (after rotations) should point in the direction of a2ch
        new_pt = mitral_valve + a2ch_axis[0]["dir_x"] / 2
        landmarks = (mitral_valve, new_pt, apex)
        normal = calculate_plane_normal(*landmarks)
        origin = np.mean(np.array(landmarks), axis=0)
        return origin, normal, landmarks

    @staticmethod
    def _apical(_model):
        valve_centers = get_centers(_model, (7, 8))
        mid_valve = valve_centers[0] + (valve_centers[0] - valve_centers[1]) / 2
        apex = get_apex(_model)
        landmarks = [mid_valve, valve_centers[0], apex]
        normal = calculate_plane_normal(*landmarks)
        origin = np.mean(np.array(landmarks), axis=0)

        return origin, normal, landmarks


    @staticmethod
    def _plax(_model):
        """
        Get the landmarks for a PLAX slice from a models. Should be the same as a APLAX slice as well.
        Landmakrs are:
            - apex
            - mitral valve center
            - aortic valve center

        """

        valve_centers = get_centers(_model, (7, 9))
        apex = get_apex(_model)
        # TODO: we defined normal and origin using opposite origins here. Probably not good practice.
        normal = calculate_plane_normal(*valve_centers, apex)
        origin = np.mean(np.array((apex, *valve_centers)), axis=0)

        return origin, normal, (*valve_centers, apex)
