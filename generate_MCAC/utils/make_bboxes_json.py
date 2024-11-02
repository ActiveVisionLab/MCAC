import argparse
import json
import os

import matplotlib.image as mpimg
import numpy as np
import yaml
from tqdm import tqdm

from configs.ConfigClass import ConfigClass

print("MAKING BBOXES JSON")
parser = argparse.ArgumentParser(description="Process some integers.")

parser.add_argument("--config", type=str, default="003", help="which dataset config")
parser.add_argument(
    "--redo_already_done",
    action="store_true",
    help="dont skip over ones that are already done, redo them",
)
parser.add_argument(
    "--crop_size",
    type=int,
    default=-1,
    help="if not -1 then this willl be the size of the final image from the center",
)

args = parser.parse_args()


# blender_dataset_configs = args.config
basepath = "BASE_PATH/ims/"


base_address = "BASE_PATH/"
basepath = base_address + "ims/"


specific_stream = open(f"{base_address}configs/{args.config}.yml", "r")
specific_dictionary = yaml.load(specific_stream, Loader=yaml.Loader)
defualt_stream = open(f"{base_address}configs/__DEFAULTS__.yml", "r")
configs = yaml.load(defualt_stream, Loader=yaml.Loader)
CFG = ConfigClass(configs, specific_dictionary)
frame_end = f"{CFG.simulation.end_frame - 1:04}"
blender_dataset_configs = CFG.dataset.savename

save = False
save_json = True

tags = ["test", "train", "val"]

for tag in tags:
    print(tag)
    parent_dir = f"ims/{blender_dataset_configs}_{tag}"
    seeds = os.listdir(parent_dir)
    for seed in tqdm(seeds):
        save_path = f"{parent_dir}/{seed}/info_with_occ_bbox.json"
        if not os.path.exists(seed) and save:
            os.mkdir(f"{seed}")
        if args.redo_already_done or not os.path.exists(save_path):
            seg_path = f"{parent_dir}/{seed}/seg.png"
            json_path = f"{parent_dir}/{seed}/info_with_occ.json"
            if not os.path.exists(json_path):
                print("ERROR IN BBOX JSON DOESNT EXIST", json_path)
            else:
                with open(json_path) as f:
                    dic = json.load(f)
                dic = dict(dic)
                countables = dic["countables"]
                cols = ["r", "b", "g", "c", "y"]
                all_blacks = []
                for c_i, countable in enumerate(countables):
                    countable_bounds = []
                    for (ind,) in zip(countable["inds"]):
                        indvseg_path = f"{parent_dir}/{seed}/segindsbin/{ind}.PNG"
                        indvseg = mpimg.imread(indvseg_path)
                        orig_im_shape = indvseg.shape
                        itemindex_0 = np.where(np.sum(indvseg, axis=0) != 0)
                        itemindex_1 = np.where(np.sum(indvseg, axis=1) != 0)

                        if np.sum(indvseg) == 0:
                            all_blacks.append(ind)
                            countable_bounds.append([[0, 0], [0, 0]])

                        else:
                            bounds_0 = [np.min(itemindex_1), np.max(itemindex_1)]
                            bounds_1 = [np.min(itemindex_0), np.max(itemindex_0)]

                            bounds_0[0] = int(np.max((0, bounds_0[0] - 1)))
                            bounds_0[1] = int(np.min((1080, bounds_0[1])))
                            bounds_1[0] = int(np.max((0, bounds_1[0] - 1)))
                            bounds_1[1] = int(np.min((1080, bounds_1[1])))
                            countable_bounds.append([bounds_0, bounds_1])
                    countable["bboxes"] = countable_bounds

                    if args.crop_size != -1:
                        countable_bounds_cropped = []

                        for (ind,) in zip(countable["inds"]):

                            indvseg_path = f"{parent_dir}/{seed}/segindsbin/{ind}.PNG"
                            indvseg = mpimg.imread(indvseg_path)
                            crop_voundary_size_0 = int(
                                (indvseg.shape[0] - args.crop_size) / 2
                            )
                            crop_voundary_size_1 = int(
                                (indvseg.shape[1] - args.crop_size) / 2
                            )
                            indvseg_cropped = indvseg[
                                crop_voundary_size_0:-crop_voundary_size_0,
                                crop_voundary_size_0:-crop_voundary_size_0,
                            ]
                            itemindex_0 = np.where(np.sum(indvseg_cropped, axis=0) != 0)
                            itemindex_1 = np.where(np.sum(indvseg_cropped, axis=1) != 0)

                            if np.sum(indvseg_cropped) == 0:
                                print(
                                    f"    Cropped: All Black instance map {seed} {ind}"
                                )
                                countable_bounds_cropped.append([[0, 0], [0, 0]])

                            else:
                                bounds_0 = [np.min(itemindex_1), np.max(itemindex_1)]
                                bounds_1 = [np.min(itemindex_0), np.max(itemindex_0)]

                                bounds_0[0] = int(np.max((0, bounds_0[0] - 1)))
                                bounds_0[1] = int(np.min((1080, bounds_0[1])))
                                bounds_1[0] = int(np.max((0, bounds_1[0] - 1)))
                                bounds_1[1] = int(np.min((1080, bounds_1[1])))
                                countable_bounds_cropped.append([bounds_0, bounds_1])

                        countable[f"bboxes_crop{args.crop_size}"] = (
                            countable_bounds_cropped
                        )

                    countables[c_i] = countable

                if len(all_blacks) > 1:
                    print(
                        f"   {len(all_blacks)}/{len(countables)} All Black instance maps {seed}, {all_blacks}"
                    )

                if save_json:
                    dic["countables"] = countables
                    with open(save_path, "w") as f:
                        json.dump(dic, f)

