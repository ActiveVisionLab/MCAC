import argparse
import os
import re

import yaml

from configs.ConfigClass import ConfigClass

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
blender_dataset_configs = CFG.dataset.savename


directory_parent = f"BASE_PATH/ims/{blender_dataset_configs}_"
tags = ["train", "test", "val"]

possible_end_frame = ["0499", "2499", "0074"]


def check_indv_format_bin(string):
    pattern = r"^\d+_\d+.PNG$"
    match = re.match(pattern, string)
    return bool(match)


def check_indv_format(string):
    pattern = r"^\d+_\d+.png$"
    match = re.match(pattern, string)
    return bool(match)


def check_new_indv_format_bin(string):
    pattern = r"^\d+.PNG$"
    match = re.match(pattern, string)
    return bool(match)


def check_new_indv_format(string):
    pattern = r"^\d+.png$"
    match = re.match(pattern, string)
    return bool(match)


print("renaming files")

for tag in tags:
    print(tag)
    directory = directory_parent + tag

    for subdir, dirs, files in os.walk(directory):
        for file in files:
            # get the full path of the file
            file_path = os.path.join(subdir, file)
            # print(file_path)

            for fr in possible_end_frame:
                if file == f"info{fr}.json":
                    new_file_path = os.path.join(subdir, "info.json")
                    os.rename(file_path, new_file_path)
                    break
                elif file == f"info{fr}_with_occ.json":
                    new_file_path = os.path.join(subdir, "info_with_occ.json")
                    os.rename(file_path, new_file_path)
                    break
                elif file == f"info{fr}_with_occ_bbox.json":
                    new_file_path = os.path.join(subdir, "info_with_occ_bbox.json")
                    os.rename(file_path, new_file_path)
                    break
                elif file == f"img{fr}.png":
                    new_file_path = os.path.join(subdir, "img.png")
                    os.rename(file_path, new_file_path)
                    break
                elif file == f"seg{fr}.png":
                    new_file_path = os.path.join(subdir, "seg.png")
                    os.rename(file_path, new_file_path)
                    break
                elif "segindsbin" in subdir:
                    if check_indv_format_bin(file):
                        new_file_path = os.path.join(
                            subdir, file.split(".")[0].split("_")[0] + ".PNG"
                        )
                        os.rename(file_path, new_file_path)
                        break
                    elif check_new_indv_format_bin(file):
                        pass
                    else:
                        print("ERROR, bad individual seg bin", file, subdir)

                elif "seginds" in subdir:
                    if check_indv_format(file):
                        new_file_path = os.path.join(
                            subdir, file.split(".")[0].split("_")[0] + ".png"
                        )
                        os.rename(file_path, new_file_path)
                        break
                    elif check_new_indv_format(file):
                        pass
                    else:
                        print("ERROR, bad individual seg", file, subdir)

                    # for i in range(100):
                    #     print(i)
                    #     if file == "f{i}_{fr}.PNG":
                    #         print(f"change {file}")
                    #         break

                    # print(file)
                # else:
                #     print("  ", file)

                # if file.split(".")[-1] == "PNG":
                #     if file.split(".")[0].split("_")[-1] == "0499" or file.split(".")[0].split("_")[-1] == "2499":
                #         new_file_path = os.path.join(subdir, file.split(".")[0].split("_")[0] + ".PNG")
                #         os.rename(file_path, new_file_path)
                # else:
                #     print(file)
