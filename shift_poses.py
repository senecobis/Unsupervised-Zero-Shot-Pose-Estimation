import os
import glob
import numpy as np
from src.bbox_3D_estimation.utils import read_list_poses_orig
import re

regex = re.compile('[^0-9]')

DIR = "data/onepose_datasets/val_data/0606-tiger-others/tiger-2/"

poses = sorted(glob.glob(f"{DIR}backup/poses/*.txt"))
names = []
for pose in poses:
    name = regex.sub("", pose)[5:]
    names.append(name)
poses = read_list_poses_orig(poses)



shifted_poses = []
for pose in poses:
    inverted = np.linalg.inv(pose)
    inverted[0:3,3] += np.array([+5, -10, +10])
    original = np.linalg.inv(inverted)

    shifted_poses.append(original)


shift_pose_dir = f"{DIR}poses/"
os.makedirs(shift_pose_dir, exist_ok=True)
for pose, name in zip(shifted_poses, names):
    np.savetxt(f"{shift_pose_dir}{name}.txt", pose, delimiter=" ")
