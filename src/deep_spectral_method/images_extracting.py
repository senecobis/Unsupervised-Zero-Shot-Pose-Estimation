#
# Script to extract and write txt of images from
# a video by @senecobis
#
import cv2, os
from os import listdir
from os.path import isfile, join

# TODO: use pathlib.Path for path setting
name_len = 6
downscale_factor = 0.3


def extract_images(video_path: str, images_root):
    cam = cv2.VideoCapture(video_path)
    try:
        if not os.path.exists(images_root):
            os.makedirs(images_root)
    except OSError:
        print("Error: Creating directory of" + images_root)

    currentframe = 0
    while True:
        remains, frame = cam.read()  # reading from frame
        name_str = str(currentframe)
        curr_name = "/" + "0" * (name_len - len(name_str)) + name_str

        if remains:
            name = images_root + curr_name + ".jpg"
            print("Creating..." + name)
            img_half = cv2.resize(
                frame, (0, 0), fx=downscale_factor, fy=downscale_factor
            )
            cv2.imwrite(name, img_half)
            currentframe += 1
        else:
            break

    cam.release()
    print("\n Saving all images in ", images_root)


def write_images_txt(imlist_root, file_txt, images_path):
    try:
        if not os.path.exists(imlist_root):
            os.makedirs(imlist_root)
    except OSError:
        print("Error: Creating directory of" + imlist_root)

    sorted_imgs = [
        f for f in sorted(listdir(images_path)) if isfile(join(images_path, f))
    ]
    txt_path = imlist_root + file_txt
    with open(txt_path, "w+") as f:
        f.write("\n".join(sorted_imgs))
