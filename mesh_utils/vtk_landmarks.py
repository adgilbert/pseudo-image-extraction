from os import stat
import numpy as np

from data_handlers.heart import calculate_plane_normal
from mesh_utils.image_operations import apply_rotations
from mesh_utils.vtk_functions import get_centers, get_apex, get_rv_apex


class LandmarkSelector:
    def __init__(self, view_name):
        if not "_" + view_name in dir(self):
            raise KeyError(
                f"must define iterator function with name {'_' + view_name} in utils/vtk_landmarks.py")
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
        rotations = [dict(name="get_a2ch", axis=long_axis,
                          center=mitral_valve, angles=[-1 * np.deg2rad(70)])]
        base_plane = dict(origin=origin, dir_x=short_axis, dir_y=long_axis)
        a2ch_axis = next(apply_rotations(
            base_plane, rotations, gen_type='zip'))

        # now the new short axis (after rotations) should point in the direction of a2ch
        new_pt = mitral_valve + a2ch_axis[0]["dir_x"] / 2
        landmarks = (mitral_valve, new_pt, apex)
        normal = calculate_plane_normal(*landmarks)
        origin = np.mean(np.array(landmarks), axis=0)
        return origin, normal, landmarks

    @staticmethod
    def _apical(_model):
        valve_centers = get_centers(_model, (7, 8))
        mid_valve = valve_centers[0] + \
            (valve_centers[0] - valve_centers[1]) / 2
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

    # ME
    """
        TEE
        Views can be found at: https://www.asecho.org/wp-content/uploads/2014/05/2013_Performing-Comprehensive-TEE.pdf
    """
    ########################################################
    #              Midesophageal Views (1-15)              #
    ########################################################

    def _v1(self, _model):
        """
        Get the landmarks required for the 1st view 
        Landmarks are:
            MV, TV, AV
        """
        landmarks = get_centers(_model, (7, 8, 9))
        mv, tv, av = landmarks

        normal = -calculate_plane_normal(*landmarks)
        origin = np.mean(np.array(landmarks), axis=0)

        return origin, normal, landmarks

    def _v2(self, _model):
        """
        Get the landmarks required for the 2nd view 
        Landmarks are:
            LV, MV, TV
        """
        landmarks = get_centers(_model, (1, 7, 8))
        la, mv, tv = landmarks

        normal = calculate_plane_normal(*landmarks)
        origin = np.mean(np.array(landmarks), axis=0)

        return origin, normal, landmarks

    def _v3(self, _model):
        """
        Get the landmarks required for the 3rd view 
        Landmarks are:
            LV, LA, MV 
        """
        landmarks = get_centers(_model, (1, 3, 7))
        lv, la, mv = landmarks

        normal = calculate_plane_normal(*landmarks)
        origin = np.mean(np.array(landmarks), axis=0)

        return origin, normal, landmarks

    def _v4(self, _model):
        """
        Get the landmarks required for the 4th view 
        Landmarks are:
            LV, MV, LA appendage
        """
        landmarks = get_centers(_model, (1, 7, 11))
        lv, mv, appendage = landmarks

        normal = -calculate_plane_normal(*landmarks)
        origin = np.mean(np.array(landmarks), axis=0)

        return origin, normal, landmarks

    def _v5(self, _model):
        """
        Get the landmarks required for the 5th view 
        Landmarks are:
            LV, MV,  AV 
        """
        landmarks = get_centers(_model, (1, 7, 9))
        lv, mv, av = landmarks
        
        normal = calculate_plane_normal(*landmarks)
        origin = np.mean(np.array(landmarks), axis=0)

        return origin, normal, landmarks

    def _v6(self, _model):
        """
        Get the landmarks required for the 6th view 
        Landmarks are:
            LA, MV, AV 
        """
        landmarks = get_centers(_model, (3, 7, 9))
        la, mv, av = landmarks

        normal = calculate_plane_normal(*landmarks)
        origin = np.mean(np.array(landmarks), axis=0)

        return origin, normal, landmarks

    def _v10(self, _model):
        """
        Get the landmarks required for the 10th view 
        Landmarks are:
            RA, AV, PV
        """
        landmarks = get_centers(_model, (4, 9, 10))
        ra, av, pv = landmarks

        normal = calculate_plane_normal(*landmarks)
        origin = np.mean(np.array(landmarks), axis=0)

        return origin, normal, landmarks

    def _v11(self, _model):
        """
        Get the landmarks required for the 11th view 
        Landmarks are:
            TV, AV, PV 
        """
        landmarks = get_centers(_model, (8, 9, 10))
        tv, av, pv = landmarks

        normal = calculate_plane_normal(*landmarks)
        origin = np.mean(np.array(landmarks), axis=0)

        return origin, normal, landmarks

    def _v12(self, _model):
        """
        Get the landmarks required for the 12th view 
        Landmarks are:
            TV, SVC, IVC 
        """
        landmarks = get_centers(_model, (8, 16, 17))
        tv, svc, ivc = landmarks

        normal = calculate_plane_normal(*landmarks)
        origin = np.mean(np.array(landmarks), axis=0)

        return origin, normal, landmarks

    def _v13(self, _model):
        """
        Get the landmarks required for the 13th view 
        Landmarks are:
            LA, APP, SVC 
        """
        landmarks = get_centers(_model, (3, 11, 16))
        la, app, svc = landmarks

        normal = calculate_plane_normal(*landmarks)
        origin = np.mean(np.array(landmarks), axis=0)

        return origin, normal, landmarks

    def _v14(self, _model):
        """
        Get the landmarks required for the 14th view 
        Landmarks are:
            PA, LSPV, RIPV
        """
        landmarks = get_centers(_model, (6, 12, 14))
        pa, lspv, ripv = landmarks

        normal = calculate_plane_normal(*landmarks)
        origin = np.mean(np.array(landmarks), axis=0)

        return origin, normal, landmarks

    def _v15(self, _model):
        """
        Get the landmarks required for the 15th view 
        Landmarks are:
            LA, APP, LSPV
        """

        landmarks = get_centers(_model, (3, 11, 12))
        la, app, lspv = landmarks

        normal = calculate_plane_normal(*landmarks)
        origin = np.mean(np.array(landmarks), axis=0)

        return origin, normal, landmarks

    ########################################################
    #              Transgastric Views (16-24)              #
    ########################################################

    def _v16(self, _model):  # TGBSAX
        """
        Get the landmarks required for the 16th view 
        Landmarks are:
            MV, TV
        """
        _, plax_normal, (mv, av, apex) = self._plax(_model)

        # PSAX views are normal to PLAX views
        tv = get_centers(_model, (8,))[0]

        landmarks = [mv, tv]
        normal = np.cross(plax_normal, (mv - tv))
        origin = np.mean(np.array(landmarks), axis=0)

        return origin, normal, landmarks

    def _v17(self, _model):  # TGBMAX
        """
        Get the landmarks required for the 17th view 
        Landmarks are:
            LV, RV 
        """
        _, plax_normal, _ = self._plax(_model)

        # PSAX views are normal to PLAX views
        lv, rv = get_centers(_model, (1, 2))

        landmarks = [lv, rv, ]
        normal = -np.cross(plax_normal, (lv - rv))
        origin = np.mean(np.array(landmarks), axis=0)

        return origin, normal, landmarks

    def _v18(self, _model):
        """
        Get the landmarks required for the 18th view in 
        Landmarks are:
            - LV apex
            - RV apex
        """

        _, plax_normal, (mv, _, apex) = self._plax(_model)

        rv_apex = get_rv_apex(_model)
        landmarks = [apex, rv_apex, mv]  # TEMP mitral
        #normal = np.cross(plax_normal, (apex - rv_apex))
        normal = mv - apex
        origin = np.mean(np.array(landmarks), axis=0)
        print("plax: ", plax_normal/np.linalg.norm(plax_normal), "normal: ",normal/np.linalg.norm(normal))
        return origin, normal, landmarks

    def _v19(self, _model):
        """
        Get the landmarks required for the 19th view in 
        Landmarks are:
            PV, TV, mid_ventricles
        """
        lv, rv, tv, pv = get_centers(
            _model, (1, 2, 8, 10))
        mid_vent = rv + 0.5*(lv - rv)
        landmarks = [tv, pv, mid_vent]

        normal = calculate_plane_normal(*landmarks)
        origin = np.mean(np.array(landmarks), axis=0)

        return origin, normal, landmarks

    def _v20(self, _model):  
        """
        Get the landmarks required for the 20th view in 
        Landmarks are:
            RA, TV, PV
        """
        ra, tv, pv = get_centers(
            _model, (4, 8, 10))
        landmarks = [ra, tv, pv]

        normal = calculate_plane_normal(*landmarks)
        origin = np.mean(np.array(landmarks), axis=0)

        return origin, normal, landmarks

    def _v21(self, _model):  
        """
        Get the landmarks required for the 21st view in 
        Landmarks are:
            LV, aorta, AV
        """
        lv, aorta, av = get_centers(
            _model, (1, 5, 9))
        landmarks = [lv, aorta, av]

        normal = calculate_plane_normal(*landmarks)
        origin = np.mean(np.array(landmarks), axis=0)

        return origin, normal, landmarks

    def _v22(self, _model):  # unverified
        """
        Get the landmarks required for the 22nd view 
        Landmarks are:
            LV, LA, MV
        """
        # PSAX views are normal to PLAX views
        landmarks = get_centers(_model, (1, 3, 7))
        lv, la, mv = landmarks

        normal = -calculate_plane_normal(*landmarks)
        origin = np.mean(np.array(landmarks), axis=0)

        return origin, normal, landmarks

    def _v23(self, _model):  # unverified
        """
        Get the landmarks required for the 23rd view 
        Landmarks are:
            RV, RA, TV
        """
        # PSAX views are normal to PLAX views
        landmarks = get_centers(_model, (2, 4, 8))
        rv, ra, tv = landmarks

        normal = -calculate_plane_normal(*landmarks)
        origin = np.mean(np.array(landmarks), axis=0)

        return origin, normal, landmarks

    def _v24(self, _model):  # unverified v5 and v24 duplication
        """
        Get the landmarks required for the 24th view 
        Landmarks are:
            LV, MV, AV
        """

        # PSAX views are normal to PLAX views
        landmarks = get_centers(_model, (1, 7, 9))
        lv, mv, av = landmarks

        normal = -calculate_plane_normal(*landmarks)
        origin = np.mean(np.array(landmarks), axis=0)

        return origin, normal, landmarks

    ########################################################
    #                   Aortic Views (25-28)               #
    ########################################################

    # No aorta present in the models
