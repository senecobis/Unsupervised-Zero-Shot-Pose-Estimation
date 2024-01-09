# ZeroShotPoseEstimation: OnePose and Deep Spectral Method

> ZeroShotPoseEstimation: OnePose and Deep Spectral Method 
> [Roberto Pellerito](https://github.com/senecobis)<sup>\*</sup>, [Alessandro Burzio](https://github.com/Ale-Burzio)<sup>\*</sup>, [Lorenzo Piglia]()<sup>\*</sup>, [Diego Machain]()

(to run DEMO with custom dataset see below)

## Installation

```shell
conda env create -f environment.yaml
conda activate onepose
```
if you have a m1 chip run the following commands:
```shell
CONDA_SUBDIR=osx-64 conda env create -f environment.yaml
conda activate onepose
```
Download DINO

We use [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork) and [SuperGlue](https://github.com/magicleap/SuperPointPretrainedNetwork) 
for 2D feature detection and matching in this project.
We can't provide the code directly due its LICENSE requirements, please download the inference code and pretrained models using the following script：
```shell
REPO_ROOT=/path/to/OnePose
cd $REPO_ROOT
sh ./scripts/prepare_2D_matching_resources.sh
```

[COLMAP](https://colmap.github.io/) is used in this project for Structure-from-Motion. 
Please refer to the official [instructions](https://colmap.github.io/install.html) for the installation.

[Optional, WIP] You may optionally try out our web-based 3D visualization tool [Wis3D](https://github.com/zju3dv/Wis3D) for convenient and interactive visualizations of feature matches. We also provide many other cool visualization features in Wis3D, welcome to try it out.


# Running on Euler

```shell
source $HOME/onepose/bin/activate
```

## Training and Evaluation on OnePose dataset
### Dataset setup 
1. Download OnePose dataset from [onedrive storage](https://zjueducn-my.sharepoint.com/:f:/g/personal/zihaowang_zju_edu_cn/ElfzHE0sTXxNndx6uDLWlbYB-2zWuLfjNr56WxF11_DwSg?e=GKI0Df) and extract them into `$/your/path/to/onepose_datasets`. 
The directory should be organized in the following structure:
    ```
    |--- /your/path/to/onepose_datasets
    |       |--- train_data
    |       |--- val_data
    |       |--- test_data
    |       |--- sample_data
    ```

2. Build the dataset symlinks
    ```shell
    REPO_ROOT=/path/to/OnePose
    ln -s /your/path/to/onepose_datasets $REPO_ROOT/data/onepose_datasets
    ```

3. Run Structure-from-Motion for the data sequences

    Reconstructed the object point cloud and 2D-3D correspondences are needed for both training and test objects (if you haven't run the sfm befor remember to set the  "redo" fields to true in config files in hydra and configs):
    ```python
    python run.py +preprocess=sfm_spp_spg_train.yaml # for training data
    python run.py +preprocess=sfm_spp_spg_test.yaml # for testing data
    python run.py +preprocess=sfm_spp_spg_val.yaml # for val data
    python run.py +preprocess=sfm_spp_spg_sample.yaml # an example, if you don't want to test the full dataset
    python run.py +preprocess=sfm_spp_spg_test_experiment.yaml
    ```


### Inference on OnePose dataset
1. Download the pretrain weights [pretrained model](https://drive.google.com/drive/folders/1VjLLjJ9oxjKV5Xy3Aty0uQUVwyEhgtIE?usp=sharing) and move it to `${REPO_ROOT}/data/model/checkpoints/onepose/GATsSPG.ckpt`.

2. Inference with category-agnostic 2D object detection.

    When deploying OnePose to a real world system, 
    an off-the-shelf category-level 2D object detector like [YOLOv5](https://github.com/ultralytics/yolov5) can be used.
    However, this could defeat the category-agnostic nature of OnePose.
    We can instead use a feature-matching-based pipeline for 2D object detection, which locates the scanned object on the query image through 2D feature matching.
    Note that the 2D object detection is only necessary during the initialization.
    After the initialization, the 2D bounding box can be obtained from projecting the previously detected 3D bounding box to the current camera frame.
    Please refer to the [supplementary material](https://zju3dv.github.io/onepose/files/onepose_supp.pdf) for more details. 

    ```python
    # Obtaining category-agnostic 2D object detection results first.
    # Increasing the `n_ref_view` will improve the detection robustness but with the cost of slowing down the initialization speed.
    python feature_matching_object_detector.py +experiment=object_detector.yaml n_ref_view=15
    #This command takes less time and only uses the sample dataset
    python feature_matching_object_detector.py +experiment=object_detector_2.yaml n_ref_view=2

    # Running pose estimation with `object_detect_mode` set to `feature_matching`.
    # Note that enabling visualization will slow down the inference.
    #This inference only takes into consideration the sample data, if we want to apply it to the complete dataset, we need to set +experiment=test_GATsSPG_2.yaml
    python inference.py +experiment=test_GATsSPG_2.yaml object_detect_mode=feature_matching save_wis3d=True
    ```

3. Running inference with ground-truth 2D bounding boxes

    The following command should reproduce results in the paper, which use 2D boxes projected from 3D boxes as object detection results.

    ```python
    # Note that enabling visualization will slow down the inference.
    python inference.py +experiment=test_GATsSPG_2.yaml object_detect_mode=GT_box save_wis3d=True # for testing data
    ```
    
4. [Optional] Visualize matching and estimated poses with Wis3D. Make sure the flag `save_wis3d` is set as True in testing 
and the full images are extracted from `Frames.m4v` by script `scripts/parse_full_img.sh`. 
The visualization file will be saved under `cfg.output.vis_dir` directory which is set as `GATsSPG` by default. 
Run the following commands for visualization:
    ```shell
    sh ./scripts/parse_full_img.sh path_to_Frames_m4v # parse full image from m4v file

    /Users/diego/Desktop/Escritorio_MacBook_Pro_de_Diego/ETH/Third_Semester/Mixed_Reality/OnePose/runs/vis/GATsSPG/0501-matchafranzzi-box_matchafranzzi-4

    cd runs/vis/GATsSPG
    wis3d --vis_dir ./ --host localhost --port 11020

    wis3d --vis_dir /Users/diego/Desktop/Escritorio_MacBook_Pro_de_Diego/ETH/Third_Semester/Mixed_Reality/OnePose/runs/vis/GATsSPG --host localhost --port 11020
    ```
    This would launch a web service for visualization at port 11020.


### Training the GATs Network
1. Prepare ground-truth annotations. Merge annotations of training/val data:
    ```python
    python run.py +preprocess=merge_anno task_name=onepose split=train
    python run.py +preprocess=merge_anno task_name=onepose split=val
    ```
   
2. Begin training
    ```python
    python train.py +experiment=train_GATsSPG task_name=onepose exp_name=training_onepose
    ```
   
All model weights will be saved under `${REPO_ROOT}/data/models/checkpoints/${exp_name}` and logs will be saved under `${REPO_ROOT}/data/logs/${exp_name}`.
<!-- You can visualize the training process by tensorboard:
```shell
tensorboard xx
``` -->

# Additionals
    1. Use video2img.py
    python video2img.py --input=/Users/diego/Desktop/Escritorio_MacBook_Pro_de_Diego/ETH/Third_Semester/Mixed_Reality/OnePose/data/onepose_datasets/test_experiment/test_frames

    2. Downsample the images probably with parse_scanned_data.py

    Use for the box detection
    path_utils uses intrin_ba folder, which contains txt files with the bounding boxes, maaybeee

# Select the kind of object recognition method DSM vs Feature matching
    1. In inference_demo change from features to detection to use DSM


# Run DEMO in default onepose: OnePose Demo on Custom Data

In this tutorial we introduce the demo of OnePose running with data captured
with our **OnePose Cap** application available for iOS device. 
The app is still under preparing for release.
However, you can try it with the [sample data]() and skip the first step.  

### Step 1: Capture the mapping sequence and the test sequence with OnePose Cap. 
#### The app is under brewing🍺 coming soon.

### Step 2: Organize the file structure of collected sequences
1. Export the collected mapping sequence and the test sequence to the PC.
2. Rename the **annotate** and **test** sequences directories to ``your_obj_name-annotate`` and `your_obj_name-test` respectively and organize the data as the follow structure:
    ```
    |--- /your/path/to/scanned_data
    |       |--- your_obj_name
    |       |       |---your_obj_name-annotate
    |       |       |---your_obj_name-test
    ```
   Refer to the [sample data]() as an example.
3. Link the collected data to the project directory
    ```shell
    REPO_ROOT=/path/to/OnePose
    ln -s /path/to/scanned_data $REPO_ROOT/data/demo
    ```
   
Now the data is prepared!

### Step 3: Run OnePose with collected data
Download the [pretrained OnePose model](https://drive.google.com/drive/folders/1VjLLjJ9oxjKV5Xy3Aty0uQUVwyEhgtIE?usp=sharing) and move it to `${REPO_ROOT}/data/model/checkpoints/onepose/GATsSPG.ckpt`.

[Optional] To run OnePose with tracking modeule, pelase install [DeepLM](https://github.com/hjwdzh/DeepLM.git).
Please make sure the sample program in `DeepLM` can be correctly executed to ensure successful installation.


Execute the following commands, and a demo video naming `demo_video.mp4` will be saved in the folder of the test sequence.
```shell
REPO_ROOT=/path/to/OnePose
OBJ_NAME=your_obj_name

cd $REPO_ROOT
conda activate OnePose

bash scripts/demo_pipeline.sh $OBJ_NAME

# [Optional] running OnePose with tracking
export PYTHONPATH=$PYTHONPATH:/path/to/DeepLM/build
export TORCH_USE_RTLD_GLOBAL=YES

bash scripts/demo_pipeline.sh $OBJ_NAME --WITH_TRACKING 

```
