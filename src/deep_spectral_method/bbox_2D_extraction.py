import glob
import os
import collections
from pathlib import Path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import data_utils
from deep_spectral_method.detection_2D_utils import UnsupBbox

data_root = "/Users/PELLERITO/Desktop/mixed_reality_code/OnePose/data/onepose_datasets/test_moccona"
feature_dir = data_root + "/DSM_features"
segment_dir = data_root + "/test_moccona-test"
intriscs_path = segment_dir + "/intrinsics.txt"

def sort_path_list(path_list):
    files = {int(Path(file).stem) : file for file in path_list}
    ordered_dict = collections.OrderedDict(sorted(files.items()))
    return list(ordered_dict.values())

K, _ = data_utils.get_K(intriscs_path)

img_lists = glob.glob(os.path.join(os.getcwd(), f"{segment_dir}/color_full", "*.png"))
img_lists = sort_path_list(img_lists)

BboxPredictor = UnsupBbox(feature_dir=feature_dir)

bbox_orig_res = BboxPredictor.infer_2d_bbox(image_path=img_lists[0], K=K)

print(bbox_orig_res)