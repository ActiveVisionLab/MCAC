import os
import shutil
import argparse

from configs.ConfigClass import ConfigClass
import yaml

parser = argparse.ArgumentParser(description="Process some integers.")

parser.add_argument("--config", type=str, default="003", help="which dataset config")
args = parser.parse_args()


base_address = "BASE_PATH/"
basepath = base_address + "ims/"


specific_stream = open(f"{base_address}configs/{args.config}.yml", "r")
specific_dictionary = yaml.load(specific_stream, Loader=yaml.Loader)
defualt_stream = open(f"{base_address}configs/__DEFAULTS__.yml", "r")
configs = yaml.load(defualt_stream, Loader=yaml.Loader)
CFG = ConfigClass(configs, specific_dictionary)
frame_end = f"{CFG.simulation.end_frame - 1:04}"
blender_dataset_configs = CFG.dataset.savename

root = "BASE_PATH/ims/shapenet_001_"
root = f"BASE_PATH/ims/{blender_dataset_configs}_"

# subset_check = ["info0499.json", "seg0499.png", "img0499.png"]
one_of_checks = [
    ["seginds", "segindsbin"],
    [f"info{frame_end}.json", "info0499.json", "info2499.json", "info.json"],
    [f"seg{frame_end}.png", "seg0499.png", "seg2499.png", "seg.png"],
    [f"img{frame_end}.png", "img0499.png", "img2499.png", "img.png"],
]
tags = ["train", "test", "val"]

print("DELETING UNCOMPLETE SAMPLES")
for tag in tags:
    total = 0
    deleted = 0
    folders = os.listdir(root + tag)
    for folder in folders:
        sub_files = os.listdir(os.path.join(root + tag, folder))
        all_one_ofs = True
        for one_of_check in one_of_checks:
            one_of = False
            for sf in sub_files:
                if sf in one_of_check:
                    one_of = True
            if not one_of:
                all_one_ofs = False
                
        if all_one_ofs:
            # it contains all the necessary things and one of the list
            a = None
        else:
            print(f"Deleting {os.path.join(root + tag, folder)} {len(sub_files)} sfs")
            deleted += 1
            shutil.rmtree(os.path.join(root + tag, folder))
        total += 1
    print(f"{tag}: deleted {deleted} of {total}")
