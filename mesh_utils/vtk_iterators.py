import numpy as np

from mesh_utils.image_operations import apply_rotations
from mesh_utils.vtk_functions import get_angles_from_params

"""This file contains code for creating iterators to extract slices from vtk models """


class IteratorSelector:
    def __init__(self, view_name):
        if not "_" + view_name in dir(self):
            raise KeyError(f"must define iterator function with name {'_' + view_name} in utils/vtk_iterators.py")
        self.iterator_fcn = eval(f"self._{view_name}")

    def __call__(self, *args, **kwargs):
        return self.iterator_fcn(*args, **kwargs)

    @staticmethod
    def _plax(origin, normal, landmarks, params):
        """ Generate the rotation iterator for a PLAX slice
        One axis is the long axis from the apex to the center of the valves.
        The other axis is the short axis that is orthogonal to the long axis and the axis of the plane.
        Params defines whether to use random sampling or iterate as well as the limits.
        """
        if params is None:
            raise ValueError("create_slices requires params object")
        mitral_valve, aortic_valve, apex = landmarks
        mid_valve = mitral_valve + .5 * (aortic_valve - mitral_valve)
        axis0 = mid_valve - apex
        axis1 = np.cross(axis0, normal)
        angles0, angles1 = get_angles_from_params(params)

        rotations = [
            {'name': 'axis1',
             'axis': axis1,
             'center': mid_valve,
             'angles': angles1
             },
            {'name': 'axis0',
             'axis': axis0,
             'center': apex,
             'angles': angles0
             }
        ]

        # TODO: need to see where origin is and make sure it is what we want - but does this matter?
        base_plane = dict(origin=origin,
                          dir_x=axis0,
                          dir_y=axis1)

        return apply_rotations(base_plane, rotations, gen_type='zip')

    @staticmethod
    def _a4ch(origin, normal, landmarks, opts):
        """ Generate the rotation iterator for a A4CH slice
            One axis is the long axis from the apex to the center of the valves.
            The other axis is the short axis that is orthogonal to the long axis and the axis of the plane.

            Params defines whether to use random sampling or iterate as well as the limits.
            """
        mitral_valve, tricuspid_valve, apex = landmarks
        mid_valve = mitral_valve + .5 * (tricuspid_valve - mitral_valve)
        long_axis = mid_valve - apex  # long axis (y-axis)
        short_axis = mitral_valve - tricuspid_valve  # short axis (x-axis)
        # axis1 = np.cross(axis0, normal)  # short axis
        short_axis_angles, long_axis_angles = get_angles_from_params(opts)

        # HACK, FIXME: this is just to make the apical datagen work quickly. Need to update params struct
        # angles1 = np.random.normal(0, 0.01*np.pi, params.num_slices)

        rotations = [
            {'name': 'short_axis',
             'axis': short_axis,
             'center': mid_valve,
             'angles': short_axis_angles
             },
            {'name': 'long_axis',
             'axis': long_axis,
             'center': mid_valve,
             'angles': long_axis_angles
             }
        ]

        # TODO: need to see where origin is and make sure it is what we want
        base_plane = dict(origin=origin,
                          dir_x=short_axis,
                          dir_y=long_axis)

        return apply_rotations(base_plane, rotations, gen_type='zip')

    def _a2ch(self, *args, **kwargs):
        return self._a4ch(*args, **kwargs)

    def _a4ch_viewing(self, *args, **kwargs):
        return self._a4ch(*args, **kwargs)
