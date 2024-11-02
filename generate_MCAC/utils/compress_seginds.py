# %%
# import matplotlib.pyplot as plt
import argparse
import json
import os

import cv2
import matplotlib.image as mpimg
import yaml
from PIL import ImageFile
from tqdm import tqdm

from configs.ConfigClass import ConfigClass

parser = argparse.ArgumentParser(description="Process some integers.")

parser.add_argument("--config", type=str, default="003", help="which dataset config")
parser.add_argument(
    "--redo_already_done",
    action="store_true",
    help="dont skip over ones that are already done, redo them",
)

args = parser.parse_args()


ImageFile.LOAD_TRUNCATED_IMAGES = True
size = [224, 224]

blender_dataset_configs = args.config
base_address = "BASE_PATH/"

basepath = base_address + "ims/"


specific_stream = open(f"{base_address}configs/{args.config}.yml", "r")
specific_dictionary = yaml.load(specific_stream, Loader=yaml.Loader)
defualt_stream = open(f"{base_address}configs/__DEFAULTS__.yml", "r")
configs = yaml.load(defualt_stream, Loader=yaml.Loader)
CFG = ConfigClass(configs, specific_dictionary)
frame_end = f"{CFG.simulation.end_frame - 1:04}"
blender_dataset_configs = CFG.dataset.savename

train = True

save = True
plot = False

size = [224, 224]
gaussian_type = "constant_over_dataset"
gaus_constant = 8

if gaussian_type == "image_mean_minimum_dimension":
    gs_file = ""
elif gaussian_type == "instance_minimum_dimension":
    gs_file = "_gs_indv"
elif gaussian_type == "constant_over_dataset":
    gs_file = f"_c_{gaus_constant}"


print("COMPRESSING SEGS")
for tag in ["test", "train", "val"]:
    print(tag)
    if tag == "train":
        im_dir = basepath + f"{blender_dataset_configs}_train"
    elif tag == "test":
        im_dir = basepath + f"{blender_dataset_configs}_test"
    elif tag == "val":
        im_dir = basepath + f"{blender_dataset_configs}_val"
    else:
        print("not_valid_tag")
        exit()

    im_ids = [f for f in os.listdir(im_dir) if os.path.isdir(im_dir + "/" + f)]
    for i, id in tqdm(enumerate(im_ids), total=len(im_ids)):
        if not os.path.exists(f"{im_dir}/{id}/img.png"):
            print(id, f"img doesnt exist")
        elif not os.path.exists(f"{im_dir}/{id}/info.json"):
            print(id, f"info doesnt exist")
        else:
            if args.redo_already_done or not os.path.exists(
                f"{im_dir}/{id}/segindsbin"
            ):
                with open(f"{im_dir}/{id}/info.json", "r") as f:
                    try:
                        img_info = json.load(f)
                    except json.decoder.JSONDecodeError as e:
                        print(id)
                        print(e)
                        continue

                indv_seg_folder = f"{im_dir}/{id}/seginds"
                indv_seg_folder_BIN = f"{im_dir}/{id}/segindsbin"
                if not os.path.exists(indv_seg_folder_BIN):
                    os.mkdir(indv_seg_folder_BIN)

                for cls_i, countable in enumerate(img_info["countables"]):
                    countable_occ = []
                    area = []
                    indv_area = []
                    for ind in countable["inds"]:
                        indv_seg_alone = mpimg.imread(f"{indv_seg_folder}/{ind}.png")
                        indv_seg_alone[indv_seg_alone > 0.1] = 1
                        indv_seg_alone[indv_seg_alone <= 0.1] = 0
                        indv_seg_alone_cv = indv_seg_alone.copy()
                        indv_seg_alone_cv = indv_seg_alone.astype("uint8")
                        indv_seg_alone_cv *= 255

                        cv2.imwrite(
                            f"{indv_seg_folder_BIN}/{ind}.PNG",
                            indv_seg_alone_cv,
                        )

                # if len(os.listdir(indv_seg_folder)) == len(
                #     os.listdir(indv_seg_folder_BIN)
                # ):
                #     shutil.rmtree(indv_seg_folder)
