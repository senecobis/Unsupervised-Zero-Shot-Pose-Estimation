import images_extracting
from extract import extract
from pathlib import Path
from extract import extract_utils as utils
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle
import os, cv2, torch
from accelerate import Accelerator

from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    extract_video = True
    plot = True
    on_GPU = True if device == "cuda" else False
    PATH = (
        os.getcwd()
        + "/data/onepose_datasets/val_data/0620-dinosaurcup-bottle/dinosaurcup-4"
    )
    video_path = f"{PATH}/Frames.m4v"
    images_root = f"{PATH}/extracted_images"
    imlist_root = f"{PATH}/lists"
    file_txt = "/images.txt"
    model_name = "dino_vits16"
    images_list = imlist_root + file_txt

    # Try with first example
    # Before running this example, you should already have the list of images and the images per frame from the video
    # You can get this by running the normal DSM applied to the video recorded by Roberto

    if extract_video:
        images_extracting.extract_images(video_path, images_root)
        images_extracting.write_images_txt(imlist_root, file_txt, images_root)

    # list files in img directory, this is the txt file containing the name of all images
    filenames = Path(images_list).read_text().splitlines()

    # Second step, all the functions without creating folders
    # Load the model
    model, val_transform, patch_size, num_heads = utils.get_model(model_name)
    model = model.to(device)

    dataset = utils.ImagesDataset(
        filenames=filenames, images_root=images_root, transform=val_transform
    )

    dataloader = DataLoader(dataset, batch_size=1)

    # if on_GPU:
    #     accelerator = Accelerator(mixed_precision="no", cpu=False)
    # else:
    #     accelerator = Accelerator(mixed_precision="no", cpu=True)
    # model, dataloader = accelerator.prepare(model, dataloader)
    # model = model.to(accelerator.device)

    # here we are creating sub plots
    for k, (images, _, _) in enumerate(tqdm(dataloader)):

        bbox = extract_bbox(
            model=model,
            patch_size=patch_size,
            num_heads=num_heads,
            images=images,
            on_GPU=on_GPU,
        )

        if plot:
            # Bounding boxes
            limits = bbox["bboxes_original_resolution"][0]
            image_PIL = Image.open(images_root + "/" + filenames[k])
            fig = plt.figure(num=42)
            plt.clf()
            plt.imshow(image_PIL, alpha=0.9)
            plt.gca().add_patch(
                Rectangle(
                    (limits[0], limits[1]),
                    limits[2] - limits[0],
                    limits[3] - limits[1],
                    edgecolor="red",
                    facecolor="none",
                    lw=4,
                )
            )
            plt.show(block=False)
            plt.pause(0.0001)


def extract_bbox(model, patch_size, num_heads, images, on_GPU):
    feature_dict = extract.extract_features(
        model=model,
        patch_size=patch_size,
        num_heads=num_heads,
        images=images,
        on_GPU=on_GPU,
    )

    eigs_dict = extract._extract_eig(
        K=8, data_dict=feature_dict, on_gpu=on_GPU, viz=True
    )

    # Segmentation
    segmap = extract.extract_single_region_segmentations(
        feature_dict=feature_dict,
        eigs_dict=eigs_dict,
    )

    return extract.extract_bboxes(
        feature_dict=feature_dict,
        segmap=segmap,
    )


if __name__ == "__main__":
    main()
