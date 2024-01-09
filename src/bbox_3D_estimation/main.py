import glob
from tqdm import tqdm
import os
import collections
from pathlib import Path
import time
import sys
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.utils import data_utils
from deep_spectral_method.detection_2D_utils import UnsupBbox
from bbox_3D_estimation.utils import sort_path_list, predict_3D_bboxes_wrapper


if __name__ == "__main__":
    data_root = os.getcwd() + "/data/onepose_datasets/val_data/0606-tiger-others"
    segment_dir = data_root + "/tiger-2"
    intriscs_path = segment_dir + "/intrinsics.txt"

    BboxPredictor = UnsupBbox(on_GPU=True)
    K, _ = data_utils.get_K(intriscs_path)

    poses_list = glob.glob(os.path.join(os.getcwd(), f"{segment_dir}/poses", "*.txt"))
    poses_list = sort_path_list(poses_list)
    img_lists = glob.glob(
        os.path.join(os.getcwd(), f"{segment_dir}/color_full", "*.png")
    )
    img_lists = sort_path_list(img_lists)

    predict_3D_bboxes_wrapper(
        BboxPredictor=BboxPredictor,
        full_res_img_paths=img_lists,
        poses_paths=poses_list,
        intrinsics_path=K,
        data_root=data_root,
    )
