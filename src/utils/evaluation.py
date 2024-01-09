"""
The following script are used to compute the distance between the predicted poses
and the ground truth poses
"""

import math
import numpy as np
import glob
import os.path as osp
import natsort
import matplotlib.pyplot as plt

def matrix_to_euler(R):
  # Extract the yaw angle
  yaw = np.arctan2(R[1, 0], R[0, 0])
  
  # Extract the pitch angle
  pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
  
  # Extract the roll angle
  roll = np.arctan2(R[2, 1], R[2, 2])
  
  return roll, pitch, yaw

def get_pose_format(pose):
    R = pose[0:3,0:3]
    x, y, z = pose[0:3,-1]
    roll, pitch, yaw = matrix_to_euler(R)

    return  x, y, z, roll, pitch, yaw

def pose_distance(pose1, pose2):
  # Unpack the poses
  x1, y1, z1, roll1, pitch1, yaw1 = get_pose_format(pose1)
  x2, y2, z2, roll2, pitch2, yaw2 = get_pose_format(pose2)
  
  # Compute the position distance
  pos_dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
  
  # Compute the orientation distance
  R = pose1[0:3,0:3] @ pose2[0:3,0:3].T
  orient_dist_rpy = math.sqrt((roll1 - roll2)**2 + (pitch1 - pitch2)**2 + (yaw1 - yaw2)**2)
  orient_dist = np.arccos((np.trace(R)-1)/2)
  
  # Return the total distance
  return pos_dist, orient_dist

def load_gt_poses(data_dir):

    poses_dir = osp.join(data_dir, "poses")
    pose_list = glob.glob(poses_dir + "/*.txt", recursive=True)
    pose_list = natsort.natsorted(pose_list)
    return pose_list


def load_current_pose(pose_list, ind):
    return np.loadtxt(pose_list[ind])

def prepare_data(error_1, error_2, error_3):
    data = []
    for ind, error in enumerate((error_1, error_2, error_3)):
        mean = np.mean(error)
        mean_plot = [mean]*len(error)
        flier_high = [mean + 3 * np.std(error)]*len(error)
        flier_low = [mean - 3 * np.std(error)]*len(error)
        if ind == 0:
            flier_high = [0.59]*len(error)
            flier_low = [-0.59]*len(error)
        if ind == 2:
            flier_high = [0.15]*len(error)
            flier_low = [-0.11]*len(error)
        data.append(np.concatenate((error, mean_plot, flier_high, flier_low)))

        print(f"error num {ind} = {mean} ({np.std(error)})")
    return data


def plot_results(error_1, error_2, error_3, rot_error_1, rot_error_2, rot_error_3):
    # produce a plot of pos error and orientation error
    fig1, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    data_pos = prepare_data(error_1, error_2, error_3) 
    data_rot = prepare_data(rot_error_1, rot_error_2, rot_error_3)

    labels = ["easy", "medium", "hard"]
    colors = ['pink', 'lightblue', 'lightgreen']

    ax1.set_title('Position error')
    ax2.set_title('Orientation error')
    ax1.set_ylabel("[m] Error in meters")
    ax2.set_ylabel("[rad] Error in radiants")

    bplot = ax1.boxplot(data_pos,                     
                    vert=True,  # vertical box alignment
                    patch_artist=True,  # fill with color
                    labels=labels)  # will be used to label x-ticks)
    bplot1 = ax2.boxplot(data_rot,                    
                    vert=True,  # vertical box alignment
                    patch_artist=True,  # fill with color
                    labels=labels)  # will be used to label x-ticks))

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)

    for ax in [ax1, ax2]:
        ax.yaxis.grid(True)
    
    ax1.set_ylim([-0.5, 1])

    plt.savefig("myplot")

if __name__ == "__main__":
    tiger_translation_eval = np.loadtxt("calendar_translation_eval.txt")
    tiger_orientation_eval = np.loadtxt("calendar_orientation_eval.txt")

    adidas_translation_eval = np.loadtxt("adidas_translation_eval.txt")
    adidas_orientation_eval = np.loadtxt("adidas_orientation_eval.txt")

    dinosaurcup_translation_eval = np.loadtxt("milk_translation_eval.txt")
    dinosaurcup_orientation_eval = np.loadtxt("milk_orientation_eval.txt")

    plot_results(
        tiger_translation_eval, 
        adidas_translation_eval, 
        dinosaurcup_translation_eval, 
        tiger_orientation_eval, 
        adidas_orientation_eval, 
        dinosaurcup_orientation_eval
        )