import numpy as np
from src.utils.colmap.read_write_model import read_model
from src.utils.box_utils import compute_oriented_bbox

def compute_3dbbox_from_sfm(sfm_ws_dir, data_root):

    cameras, images, points3D = read_model(sfm_ws_dir)
    for id, key in enumerate(points3D):
        if id == 0:
            points = points3D[key][1]
        else:
            points = np.vstack((points, points3D[key][1]))
    corners = compute_oriented_bbox(points)
    corners = np.vstack((corners[:,0], corners[:,1], corners[:,2])).T * 10**-1
    np.savetxt(data_root + '/box3d_corners.txt', corners, delimiter=' ')
    return corners