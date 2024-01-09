import numpy as np
from matplotlib import pyplot as plt
import itertools
import glob
import os
from utils import (
    compute_estimates,
    dual_quadric_to_ellipsoid_parameters,
    read_list_box,
    read_list_poses,
)
from plotting import plot_3D_scene


###########################################
# 1. Set the parameters for the algorithm #
#    and load the input data.             #
###########################################
# Select the dataset to be used.
# The name of the dataset defines the names of input and output directories.
dataset = "tiger-2"

# Select whether to save output images to files.
save_output_images = True

# Randomly use less images (messo se ci sono video troppo lunghi)
random_downsample = False

if dataset != "Aldoma":
    PATH = os.getcwd() + "/data/onepose_datasets/val_data/0606-tiger-others"
    box_list = sorted(
        glob.glob(os.path.join(os.getcwd(), f"{PATH}/{dataset}/bboxes", "*.txt"))
    )
    poses_list = sorted(
        glob.glob(os.path.join(os.getcwd(), f"{PATH}/{dataset}/poses_ba", "*.txt"))
    )
    intrinsics = f"{PATH}/{dataset}/intrinsics.txt"
    bbs = read_list_box(box_list)
    Ms_t = read_list_poses(poses_list)
    GT_bb = np.loadtxt(f"{PATH}/box3d_corners_original.txt")
    visibility = np.ones((bbs.shape[0], 1))
    with open(intrinsics) as f:
        intr = f.readlines()
        K = np.array(
            [
                [float(intr[0]), 0, float(intr[2])],
                [0, float(intr[1]), float(intr[3])],
                [0, 0, 1],
            ]
        )
else:
    bbs = np.load("data/{:s}/bounding_boxes.npy".format(dataset))
    K = np.load("data/{:s}/intrinsics.npy".format(dataset))
    Ms_t = np.load("data/{:s}/camera_poses.npy".format(dataset))
    visibility = np.load("data/{:s}/visibility.npy".format(dataset))

if random_downsample:
    randomline = np.random.choice(bbs.shape[0], 100)
    visibility[randomline, :] = 1

# Compute the number of frames and the number of objects
# for the current dataset from the size of the visibility matrix.

n_frames = visibility.shape[0]
n_objects = visibility.shape[1]

######################################
# 2. Run the algorithm: estimate the #
#    object ellipsoids for all the   #
#    objects in the scene.           #
######################################

estQs = compute_estimates(bbs, K, Ms_t, visibility)

##################################
# 3. Get the points of the bbox  #
# of the object with object_idx  #
##################################
object_idx = 0
while object_idx >= estQs.shape[0] or object_idx < 0:
    print(
        "Insert a valid object idx, possible values are: "
        + str(np.arange(0, estQs.shape[0]))
    )
    object_idx = int(input("Enter your value: "))
centre, axes, R = dual_quadric_to_ellipsoid_parameters(estQs[object_idx])

# Possible coordinates
mins = [-ax for (ax) in axes]
maxs = [ax for (ax) in axes]

# Coordinates of the points mins and maxs
points = np.array(list(itertools.product(*zip(mins, maxs))))

# Points in the camera frame
points = np.dot(points, R.T)

# Shift correctly the parralelepiped
points[:, 0:3] = np.add(
    centre[None, :],
    points[:, :3],
)

# print(points)

# Plot ellipsoids and camera poses in 3D.
plot = True
if plot:
    plot_3D_scene(
        estQs=estQs,
        gtQs=estQs,
        Ms_t=Ms_t,
        dataset=dataset,
        save_output_images=save_output_images,
        points=points,
        GT_points=GT_bb,
    )
    plt.show()
