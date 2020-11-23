import numpy as np
from vtk.util.numpy_support import vtk_to_numpy


# 01. LV myocardium (endo + epi)
# 02. RV myocardium (endo + epi)
# 03. LA myocardium (endo + epi)
# 04. RA myocardium (endo + epi)
#
# 05. Aorta
# 06. Pulmonary artery
#
# 07. Mitral valve
# 08. Triscupid valve
#
# 09. Aortic valve
# 10. Pulmonary valve

# 11. Appendage
# 12. Left superior pulmonary vein
# 13. Left inferior pulmonary vein
# 14. Right inferior pulmonary vein
# 15. Right superior pulmonary vein
#
# 16. Superior vena cava
# 17. Inferior vena cava

# 18. Appendage border
# 19. Right inferior pulmonary vein border
# 20. Left inferior pulmonary vein border
# 21. Left superior pulmonary vein border
# 22. Right superior pulmonary vein border
# 23. Superior vena cava border
# 24. Inferior vena cava border


def project_point_to_plane(bbox, point):
    """
    Following https://stackoverflow.com/questions/9605556/how-to-project-a-point-onto-a-plane-in-3d (valdo answer)
    Project a point onto a plane
    :param bbox: defines a plane, should contain "origin", "dir_x" and "dir_y"
    :param point: the point to project onto the plane
    :return:
    """
    # normalize axes
    x_normed = bbox["dir_x"] / np.linalg.norm(bbox["dir_x"])
    y_normed = bbox["dir_y"] / np.linalg.norm(bbox["dir_y"])
    x_proj = (point - bbox["origin"]).dot(x_normed)
    y_proj = (point - bbox["origin"]).dot(y_normed)
    proj_point = bbox["origin"] + x_proj * x_normed + y_proj * y_normed
    return proj_point


def get_angles_from_params(opts):
    """ extract the angles to use based on the defined params"""
    num = opts.num_slices  # number of angles is defined based on number of desired slices.
    if opts.rotation_type == "random":
        angles_x = np.random.normal(0, np.pi * opts.x_axis_rotation_param, num)
        angles_y = np.random.normal(0, np.pi * opts.y_axis_rotation_param, num)
        # we dont want to go more than +/- 2std
        angles_x[angles_x > 2*np.pi * opts.x_axis_rotation_param] = 2*np.pi * opts.x_axis_rotation_param
        angles_x[angles_x < -2 * np.pi * opts.x_axis_rotation_param] = -2 * np.pi * opts.x_axis_rotation_param
        angles_y[angles_x > 2 * np.pi * opts.y_axis_rotation_param] = 2 * np.pi * opts.y_axis_rotation_param
        angles_y[angles_x < -2 * np.pi * opts.y_axis_rotation_param] = -2 * np.pi * opts.y_axis_rotation_param

    elif opts.rotation_type == "iterate":
        angles_x = np.linspace(-np.pi * opts.x_axis_rotation_param, np.pi * opts.x_axis_rotation_param, num)
        angles_y = np.linspace(-np.pi * opts.y_axis_rotation_param, np.pi * opts.y_axis_rotation_param, num)
    else:
        angles_x = np.zeros(num)
        angles_y = np.zeros(num)
    return angles_x, angles_y


def get_centers(_model, _labels):
    centers = []
    for lab in _labels:
        centers.append(_model.get_center(_model.threshold(lab, lab)))
    print('centers: {}'.format(centers))
    return centers


def get_apex(_model):
    """ Get the apex of the LV (index 1) from the models.
    The apex is defined as the point farthest away from the center of the mitral_valve
    """
    lv = _model.threshold(1, 1)
    lv_points = vtk_to_numpy(lv.GetOutput().GetPoints().GetData())
    mv_center = get_centers(_model, (7,))[0]  # output is a list that matches input labels so take first index.
    apex_id = np.argmax([np.linalg.norm(mv_center - x) for x in lv_points])
    apex = lv_points[apex_id]
    return apex
