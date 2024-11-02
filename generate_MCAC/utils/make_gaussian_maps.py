import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import ImageFile
from tqdm import tqdm

from configs.ConfigClass import ConfigClass


def gaussian_2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    mx = np.array((mx))
    my = np.array((my))
    sx = np.array((sx))
    sy = np.array((sy))
    return (
        1
        / (2 * math.pi * sx * sy)
        * np.exp(-((x - mx) ** 2 / (2 * sx**2) + (y - my) ** 2 / (2 * sy**2)))
    )


parser = argparse.ArgumentParser(description="Process some integers.")

parser.add_argument("--config", type=str, default="003", help="which dataset config")
parser.add_argument(
    "--gauss_type",
    type=str,
    default="constant_over_dataset",
    help="if use constant gauss curve on all images",
)

parser.add_argument(
    "--gauss_constant", type=int, default=8, help="std of constant gauss curve"
)

parser.add_argument(
    "--occulsion_limit", type=int, default=-1, help="limit of occlusion allowed"
)
parser.add_argument(
    "--vis_area_limit", type=int, default=-1, help="limit of visible area allowed"
)
parser.add_argument(
    "--non_int_count",
    action="store_true",
    help="when true an instance on the border of the image will have a gaussian sum less than 1",
)
parser.add_argument(
    "--redo_already_done",
    action="store_true",
    help="dont skip over ones that are already done, redo them",
)
parser.add_argument("--dont_save", action="store_true", help="")
parser.add_argument("--plot", action="store_true", help="")

parser.add_argument(
    "--crop_size",
    type=int,
    default=-1,
    help="if not -1 then this will be the size of the final image,",
)
parser.add_argument(
    "--img_size",
    type=int,
    default=224,
    help="square size",
)
args = parser.parse_args()

print("making gaussians")
ImageFile.LOAD_TRUNCATED_IMAGES = True
size = [args.img_size, args.img_size]
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


save = not args.dont_save
plot = args.plot

gaussian_type = args.gauss_type
gaus_constant = args.gauss_constant

if gaussian_type == "image_mean_minimum_dimension":
    gs_file = ""
elif gaussian_type == "instance_minimum_dimension":
    gs_file = "_gs_indv"
elif gaussian_type == "constant_over_dataset":
    gs_file = f"_c_{gaus_constant}"

if args.occulsion_limit != -1:
    gs_file += "_occ_" + str(int(args.occulsion_limit))
elif args.vis_area_limit != -1:
    gs_file += "_area_" + str(int(args.vis_area_limit))


if args.non_int_count:
    gs_file += "_non_int"

if args.crop_size != -1:
    gs_file += f"_crop{args.crop_size}"

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
        sv_pth = f"{im_dir}/{id}/mah_gtdensity_{size[0]}{gs_file}_np.npy"
        if args.redo_already_done or not os.path.exists(sv_pth):
            if args.occulsion_limit != -1 or args.vis_area_limit != -1:
                img_info_path = f"{im_dir}/{id}/info_with_occ.json"
            else:
                img_info_path = f"{im_dir}/{id}/info.json"

            if not os.path.exists(img_info_path):
                print("ERROR GAUSSINAN INFOR PATH MISSING", img_info_path)
            else:
                with open(img_info_path, "r") as f:
                    img_info = json.load(f)
                h, w = size[0], size[1]
                x = np.linspace(0, h, h)
                y = np.linspace(0, w, w)
                x, y = np.meshgrid(x, y)
                cols_0 = ["r", "g", "b", "c", "k"]

                z_all = np.zeros([size[0], size[1], 1])

                if args.non_int_count:
                    if gaussian_type == "constant_over_dataset":
                        sd = gaus_constant
                    else:
                        print("XXX not a valid config for gaussian type")

                    gs = gaussian_2d(x, y, mx=w / 2, my=h / 2, sx=sd, sy=sd)
                    sum_scale = np.sum(gs)

                for cls_i, countable in enumerate(img_info["countables"]):

                    if args.crop_size == -1:
                        centers_str = "centers"
                        occlusions_str = "occlusions"
                        area_str = "area"
                    else:
                        centers_str = f"centers_crop{args.crop_size}"
                        occlusions_str = f"occlusions_crop{args.crop_size}"
                        area_str = "area"

                    z = np.zeros((h, w))
                    for center_2, occ, ar in zip(
                        countable[centers_str],
                        countable[occlusions_str],
                        countable[area_str],
                    ):

                        if (
                            args.occulsion_limit != -1 and occ < args.occulsion_limit
                        ) or (args.vis_area_limit != -1 and ar > args.vis_area_limit):
                            my, mx = (
                                size[0] * center_2[0],
                                size[1] - (size[1] * center_2[1]),
                            )
                            if gaussian_type == "constant_over_dataset":
                                sd = gaus_constant
                            else:
                                print("not a valid config for gaussian type")
                            gs = gaussian_2d(
                                x,
                                y,
                                mx=mx,
                                my=my,
                                sx=sd,
                                sy=sd,
                            )
                            if not args.non_int_count:
                                # instances half off the image in this cases will have a gaussain that sums to 1 even though mostly occluded
                                sum_scale = np.sum(gs)
                            gs /= sum_scale

                            z += gs


                    if z.max() > 0:
                        z_scaled = z / z.max()
                    else:
                        z_scaled = z

                    if cls_i == 0:
                        z_all = np.expand_dims(z, 0)
                    else:
                        z_all = np.concatenate((z_all, np.expand_dims(z, 0)), axis=0)

                z_all = z_all.transpose(2, 1, 0)

                if plot:
                    z_t = z_all.copy()
                    z_t /= z_t.max()
                    print(z_t.max())
                    plt.imshow(z_t[:3])
                    plt.show()
                if save:
                    np.save(sv_pth, z_all)
                else:
                    print("NOT SAVING")
