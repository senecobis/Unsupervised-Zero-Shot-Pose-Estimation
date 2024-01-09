import numpy as np
from matplotlib import pyplot as plt
import itertools 
from .utils import compute_estimates, dual_quadric_to_ellipsoid_parameters

class Detector3D():
    def __init__(self, K) -> None:
        self.K = K
        self.bboxes = None
        self.poses = None


    def add_view(self, bbox_t: np.ndarray, pose_t: np.ndarray):
        if self.bboxes is None:
            self.bboxes = bbox_t
        else:
            self.bboxes = np.vstack((self.bboxes, bbox_t))
        
        if self.poses is None:
            self.poses = pose_t
        else:
            self.poses = np.vstack((self.poses, pose_t))
        

    def detect_3D_box(self):
        object_idx = 0
        selected_frames = self.bboxes.shape[0]
        self.visibility = np.ones((selected_frames,1))
        estQs = compute_estimates(self.bboxes, self.K, self.poses, self.visibility)
        centre, axes, R = dual_quadric_to_ellipsoid_parameters(estQs[object_idx])

        # Possible coordinates
        mins = [-ax for (ax) in axes]
        maxs = [ax for (ax) in axes]

        # Coordinates of the points mins and maxs
        points = np.array(list(itertools.product(*zip(mins, maxs))))

        # Points in the camera frame
        points = np.dot(points, R.T)

        # Shift correctly the parralelepiped
        points[:, 0:3] = np.add(centre[None, :], points[:, :3],)
        
        self.points = points

    def save_3D_box(self, data_root):
        np.savetxt(data_root + '/box3d_corners.txt', self.points, delimiter=' ')


    
