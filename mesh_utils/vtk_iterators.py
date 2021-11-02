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
        
        ## DEBUG
        #print("My Prints:", base_plane, rotations, sep='\n\n')

        return apply_rotations(base_plane, rotations, gen_type='zip')

    def _a2ch(self, *args, **kwargs):
        return self._a4ch(*args, **kwargs)

    def _a4ch_viewing(self, *args, **kwargs):
        return self._a4ch(*args, **kwargs)

    # ME
    ########################################################
    #              Midesophageal Views (1-15)              #
    ########################################################

    def get_plane_rotations(opts, origin, axis0, axis1):
        angles0, angles1 = get_angles_from_params(opts)

        rotations = [
            {'name': 'axis1',
             'axis': axis1,
             'center': origin,
             'angles': angles1
             },
            {'name': 'axis0',
             'axis': axis0,
             'center': origin,
             'angles': angles0
             }
        ]

        base_plane = dict(origin=origin,
                          dir_x=axis0,
                          dir_y=axis1)

        return base_plane, rotations


    ########################################################
    #              Midesophageal Views (1-15)              #
    ########################################################

    @staticmethod
    def _v1(origin, normal, landmarks, opts):

        mv, tv, av = landmarks
        axis0 = tv - mv
        axis1 = np.cross(axis0, normal)
        base_plane, rotations, = IteratorSelector.get_plane_rotations(opts, origin, axis0, axis1)

        return apply_rotations(base_plane, rotations, gen_type='zip')

    @staticmethod
    def _v2(origin, normal, landmarks, opts):

        lv, mv, tv = landmarks
        axis0 = tv - mv
        axis1 = np.cross(axis0, normal)

        base_plane, rotations, = IteratorSelector.get_plane_rotations(opts, origin, axis0, axis1)

        return apply_rotations(base_plane, rotations, gen_type='zip')


    @staticmethod
    def _v3(origin, normal, landmarks, opts):

        lv, la, mv = landmarks
        axis1 = la - lv
        axis0 = np.cross(axis1, normal)

        base_plane, rotations, = IteratorSelector.get_plane_rotations(opts, origin, axis0, axis1)

        return apply_rotations(base_plane, rotations, gen_type='zip')

    @staticmethod
    def _v4(origin, normal, landmarks, opts):

        lv, mv, appendage = landmarks
        axis1 = appendage - lv
        axis0 = np.cross(axis1, normal)

        base_plane, rotations, = IteratorSelector.get_plane_rotations(opts, origin, axis0, axis1)

        return apply_rotations(base_plane, rotations, gen_type='zip')

    @staticmethod
    def _v5(origin, normal, landmarks, opts):

        lv, mv, av = landmarks
        mid_valve = mv + 0.5*(av - mv)
        axis1 = mid_valve - lv
        axis0 = np.cross(axis1, normal)

        base_plane, rotations, = IteratorSelector.get_plane_rotations(opts, origin, axis0, axis1)

        return apply_rotations(base_plane, rotations, gen_type='zip')

    @staticmethod
    def _v6(origin, normal, landmarks, opts):

        la, mv, av = landmarks
        mid_valve = mv + 0.5*(av - mv)
        axis0 = mv - av
        axis1 = np.cross(axis0, normal)

        base_plane, rotations, = IteratorSelector.get_plane_rotations(opts, origin, axis0, axis1)

        return apply_rotations(base_plane, rotations, gen_type='zip')

    @staticmethod
    def _v10(origin, normal, landmarks, opts):

        ra, av, pv = landmarks
        axis0 = ra - av
        axis1 = np.cross(axis0, normal)

        base_plane, rotations, = IteratorSelector.get_plane_rotations(opts, origin, axis0, axis1)
                          
        return apply_rotations(base_plane, rotations, gen_type='zip')


    @staticmethod
    def _v11(origin, normal, landmarks, opts):

        tv, av, pv = landmarks
        axis0 = tv - av 
        axis1 = np.cross(normal, axis0)

        base_plane, rotations, = IteratorSelector.get_plane_rotations(opts, origin, axis0, axis1)

        return apply_rotations(base_plane, rotations, gen_type='zip')


    @staticmethod
    def _v12(origin, normal, landmarks, opts):

        tv, svc, ivc = landmarks
        mid_vc = ivc + 0.5*(svc - ivc)
        axis1 = mid_vc - tv 
        axis0 = np.cross(normal, axis1)

        base_plane, rotations, = IteratorSelector.get_plane_rotations(opts, origin, axis0, axis1)

        return apply_rotations(base_plane, rotations, gen_type='zip')

    @staticmethod
    def _v13(origin, normal, landmarks, opts):

        la, app, svc = landmarks

        axis0 = app - la 
        axis1 = np.cross(normal, axis0)

        base_plane, rotations, = IteratorSelector.get_plane_rotations(opts, origin, axis0, axis1)

        return apply_rotations(base_plane, rotations, gen_type='zip')


    @staticmethod
    def _v14(origin, normal, landmarks, opts):

        pa, lspv, ripv = landmarks
        mid_pve = lspv + 0.5*(ripv - lspv)
        axis0 = mid_pve - pa
        axis1 = np.cross(normal, axis0)

        base_plane, rotations, = IteratorSelector.get_plane_rotations(opts, origin, axis0, axis1)

        return apply_rotations(base_plane, rotations, gen_type='zip')


    @staticmethod
    def _v15(origin, normal, landmarks, opts):

        la, app, lspv = landmarks
        axis0 = la - app
        axis1 = np.cross(normal, axis0)

        base_plane, rotations, = IteratorSelector.get_plane_rotations(opts, origin, axis0, axis1)

        return apply_rotations(base_plane, rotations, gen_type='zip')


    ########################################################
    #              Transgastric Views (16-24)              #
    ########################################################

    @staticmethod
    def _v16(origin, normal, landmarks, opts):

        mv, tv = landmarks
        mid_valve = tv + 0.5*(tv - tv)
        axis0 = tv - tv
        #axis1 = np.cross(axis0, (mid_apex - mv) )
        axis1 = np.cross(axis0, normal)


        angles0, angles1 = get_angles_from_params(opts)

        rotations = [
            {'name': 'axis1',
             'axis': axis1,
             'center': mid_valve,
             'angles': angles1
             },
            {'name': 'axis0',
             'axis': axis0,
             'center': mid_valve,
             'angles': angles0
             }
        ]

        base_plane = dict(origin=origin,
                          dir_x=axis0,
                          dir_y=axis1)

        return apply_rotations(base_plane, rotations, gen_type='zip')

    @staticmethod
    def _v17(origin, normal, landmarks, opts):

        lv, rv = landmarks
        mid_vent = rv + 0.5*(lv - rv)
        axis0 = lv - rv
        #axis1 = np.cross(axis0, (mid_apex - mv) )
        axis1 = np.cross(axis0, normal)


        base_plane, rotations, = IteratorSelector.get_plane_rotations(opts, origin, axis0, axis1)

        return apply_rotations(base_plane, rotations, gen_type='zip')

    @staticmethod
    def _v18(origin, normal, landmarks, opts):

        apex, rv_apex, tv = landmarks
        mid_apex = rv_apex + 0.5*(apex - rv_apex)
        axis0 = apex - rv_apex
        axis1 = np.cross(axis0, (mid_apex - tv) )
        #axis1 = np.cross(axis0, normal)


        angles0, angles1 = get_angles_from_params(opts)

        rotations = [
            {'name': 'axis1',
             'axis': axis1,
             'center': mid_apex,
             'angles': angles1
             },
            {'name': 'axis0',
             'axis': axis0,
             'center': mid_apex,
             'angles': angles0
             }
        ]

        base_plane = dict(origin=origin,
                          dir_x=axis0,
                          dir_y=axis1)

        return apply_rotations(base_plane, rotations, gen_type='zip')


    @staticmethod
    def _v19(origin, normal, landmarks, opts):

        tv, pv, mid_vent = landmarks
        axis1 = pv - mid_vent #why not mid_valve?
        axis0 = np.cross(normal, axis1)


        base_plane, rotations, = IteratorSelector.get_plane_rotations(opts, origin, axis0, axis1)

        return apply_rotations(base_plane, rotations, gen_type='zip')



    @staticmethod
    def _v20(origin, normal, landmarks, opts):

        ra, tv, pv = landmarks
        axis1 = tv - pv
        axis0 = np.cross(normal, axis1)

        base_plane, rotations, = IteratorSelector.get_plane_rotations(opts, origin, axis0, axis1)

        return apply_rotations(base_plane, rotations, gen_type='zip')


    @staticmethod
    def _v21(origin, normal, landmarks, opts):

        lv, aorta, av = landmarks
        axis0 = aorta - av
        axis1 = np.cross(normal, axis0)

        base_plane, rotations, = IteratorSelector.get_plane_rotations(opts, origin, axis0, axis1)

        return apply_rotations(base_plane, rotations, gen_type='zip')


    @staticmethod
    def _v22(origin, normal, landmarks, opts):

        lv, la, mv = landmarks
        axis1 = lv - la #why not mid_valve?
        axis0 = np.cross(normal, axis1)

        base_plane, rotations, = IteratorSelector.get_plane_rotations(opts, origin, axis0, axis1)

        return apply_rotations(base_plane, rotations, gen_type='zip')


    @staticmethod
    def _v23(origin, normal, landmarks, opts):

        rv, ra, tv = landmarks
        axis1 = rv - ra #why not mid_valve?
        axis0 = np.cross(normal, axis1)

        base_plane, rotations, = IteratorSelector.get_plane_rotations(opts, origin, axis0, axis1)

        return apply_rotations(base_plane, rotations, gen_type='zip')


    @staticmethod
    def _v24(origin, normal, landmarks, opts):

        lv, mv, av = landmarks
        axis1 = av - lv #why not mid_valve?
        axis0 = np.cross(normal, axis1)

        base_plane, rotations, = IteratorSelector.get_plane_rotations(opts, origin, axis0, axis1)

        return apply_rotations(base_plane, rotations, gen_type='zip')

    ########################################################
    #                   Aortic Views (25-28)               #
    ########################################################
    
    # No aorta present in the models