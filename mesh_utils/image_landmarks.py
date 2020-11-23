""" Mimics the vtk_landmarks for images """
import numpy as np

from mesh_utils import Tags


def a4ch_image_landmarks(image):
    """ find the landmarks for a4ch in an image: mitral valve, tricuspid valve, and apex """

    def center_of_mass(locs):
        return [np.mean(l) for l in locs]

    mitral_locs = np.where(np.array(image) == Tags["mitral_valve"])
    tricuspid_locs = np.where(np.array(image) == Tags["tricuspid_valve"])
    mitral_valve = center_of_mass(mitral_locs)
    tricuspid_valve = center_of_mass(tricuspid_locs)
    lv = np.vstack(np.where(np.array(image) == Tags["lv_myocardium"])).T
    apex_pt = np.argmax([np.linalg.norm(mitral_valve - x) for x in lv])
    apex = lv[apex_pt]
    return np.array(mitral_valve), np.array(tricuspid_valve), apex
