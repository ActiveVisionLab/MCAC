import argparse
import json
import os

import matplotlib.image as mpimg
import numpy as np
import yaml
from matplotlib import cm
from PIL import Image, ImageFile
from tqdm import tqdm

from configs.ConfigClass import ConfigClass

parser = argparse.ArgumentParser(description="Process some integers.")

parser.add_argument("--config", type=str, default="003", help="which dataset config")
parser.add_argument(
    "--crop_size",
    type=int,
    default=-1,
    help="if not -1 then this willl be the size of the final image, count pixels that are in the wider shot of the unocclueded, i.e. if most of an object is off screen then its occluded",
)
parser.add_argument(
    "--redo_already_done",
    action="store_true",
    help="dont skip over ones that are already done, redo them",
)

args = parser.parse_args()


ImageFile.LOAD_TRUNCATED_IMAGES = True
size = [224, 224]
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

# transform = T.Compose([T.ToTensor()])
use_compressed = False
use_compressed = True

if use_compressed:
    seg_inds_folder = "segindsbin"
    filetype = "PNG"
else:
    seg_inds_folder = "seginds"
    filetype = "png"


print("OCCLUSION FINDER")
for tag in ["test", "train", "val"]:
    # for tag in ["train"]:
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
        sv_name = f"{im_dir}/{id}/info_with_occ.json"
        if not os.path.exists(f"{im_dir}/{id}/img.png"):
            print(id, f"img doesnt exist")
        elif not os.path.exists(f"{im_dir}/{id}/info.json"):
            print(id, f"info doesnt exist")
        else:
            if args.redo_already_done or not os.path.exists(sv_name):
                image = Image.open(f"{im_dir}/{id}/img.png")
                image.load()
                if image.mode != "RGB":
                    if image.mode != "RGBA":
                        print("IMAGE NOT RGB or RGBA", id, image.mode)
                    image = image.convert("RGB")

                with open(f"{im_dir}/{id}/info.json", "r") as f:
                    try:
                        img_info = json.load(f)
                    except json.decoder.JSONDecodeError as e:
                        print(id)
                        print(e)
                        continue

                indv_seg_folder = f"{im_dir}/{id}/{seg_inds_folder}"
                indv_seg_files = os.listdir(indv_seg_folder)

                seg = mpimg.imread(f"{im_dir}/{id}/seg.png")
                if CFG.scene.colour_depth == 16:
                    seg = (seg * 65535).astype(int)
                else:
                    if (
                        CFG.objects.max_number_per_type * CFG.objects.max_num_countables
                        > 255
                    ):
                        print("ERROR SHOULD BE USING COLOUR DEPTH 16")
                        seg = (seg * 65535).astype(int)
                    else:
                        seg = (seg * 255).astype(int)
                seg_scaled = seg / np.max(seg)
                seg -= 1  # background is 0 and first object is 1 in segmentation image

                seg_mask = seg > -1

                seg_scaled_colour = np.uint8(cm.plasma(seg_scaled) * 255)
                seg_scaled_colour = seg_scaled_colour * np.stack(
                    (seg_mask, seg_mask, seg_mask, seg_mask), axis=-1
                )
                seg_mask = Image.fromarray(seg_mask)

                seg_scaled_colour = Image.fromarray(seg_scaled_colour)
                seg_mask.save(f"{im_dir}/{id}/seg_bin.png")
                seg_scaled_colour.save(f"{im_dir}/{id}/seg_vis.png")

                if args.crop_size != -1:
                    crop_voundary_size_0 = int((seg.shape[0] - args.crop_size) / 2)
                    crop_voundary_size_1 = int((seg.shape[1] - args.crop_size) / 2)
                    seg_cropped = seg[
                        crop_voundary_size_0:-crop_voundary_size_0,
                        crop_voundary_size_0:-crop_voundary_size_0,
                    ]

                h, w = size[0], size[1]

                for cls_i, countable in enumerate(img_info["countables"]):
                    all_blacks = []
                    countable_occ = []
                    area = []
                    indv_area = []
                    occ_error_inds = []
                    for ind in countable["inds"]:
                        image_path = f"{indv_seg_folder}/{ind}.{filetype}"

                        if not os.path.exists(image_path):
                            print(id, f"indv seg doesnt exist", image_path)
                            exit()
                        else:
                            indv_seg_alone = mpimg.imread(image_path)

                        indv_seg_alone[indv_seg_alone > 0.1] = 1
                        indv_seg_alone[indv_seg_alone <= 0.1] = 0
                        ind_seg_in_image = np.zeros(indv_seg_alone.shape)
                        ind_seg_in_image[seg == ind] = 1
                        overlapping = ind_seg_in_image * indv_seg_alone

                        if np.sum(indv_seg_alone) < 1:
                            countable_occ.append(100)
                            all_blacks.append(ind)
                            occluded_section = indv_seg_alone
                            visible_percentage = 0
                            area.append(int(np.sum(ind_seg_in_image)))
                            indv_area.append(int(0))

                        else:
                            visible_percentage = np.sum(overlapping) / np.sum(
                                indv_seg_alone
                            )
                            occluded_section = indv_seg_alone - ind_seg_in_image
                            error_area = np.sum(occluded_section == -1)
                            if error_area / np.sum(indv_seg_alone) > 0.1:
                                occ_error_inds.append(ind)
                                print(
                                    f"{id=} {ind=}  perc:{100*error_area/np.sum(indv_seg_alone):.1f}% {error_area=}  area in image: {np.sum(ind_seg_in_image)}, unoccluded_area: {np.sum(indv_seg_alone)}"
                                )
                            countable_occ.append(100 - int(visible_percentage * 100))
                            area.append(int(np.sum(ind_seg_in_image)))
                            indv_area.append(int(np.sum(indv_seg_alone)))
                            if visible_percentage > 1.01:
                                print(
                                    f"   ERROR: bigger {100*np.sum(indv_seg_alone)/np.sum(ind_seg_in_image):.2f} ind:{ind} id:{id}, {np.sum(indv_seg_alone):.0f}, {np.sum(ind_seg_in_image):.0f}"
                                )

                    if len(occ_error_inds) > 0:
                        print(
                            f"{len(occ_error_inds)} errors in {id}, of final image seg being bigger than final"
                        )
                    if all(v == 100 for v in countable_occ):
                        print(f"ALL INSTANCES ARE OCCLUDED {id} {len(countable_occ)}")
                    countable["occlusions"] = countable_occ
                    countable["area"] = area
                    countable["indv_area"] = indv_area

                    if len(all_blacks) > 1:
                        print(
                            f"    {len(all_blacks)}/{len(countable['inds'])} indv_seg_alone less that 1 {id=}, {all_blacks}"
                        )

                if args.crop_size != -1:

                    for cls_i, countable in enumerate(img_info["countables"]):

                        countable_occ = []
                        area = []
                        indv_area = []
                        for ind in countable["inds"]:
                            indv_seg_alone = mpimg.imread(
                                f"{indv_seg_folder}/{ind}.{filetype}"
                            )

                            indv_seg_alone[indv_seg_alone > 0.1] = 1
                            indv_seg_alone[indv_seg_alone <= 0.1] = 0
                            ind_seg_in_image = np.zeros(seg_cropped.shape)
                            ind_seg_in_image[seg_cropped == ind] = 1
                            # padding out with zeros as this area isnt visible in the final cropped frame
                            ind_seg_in_image = np.pad(
                                ind_seg_in_image, crop_voundary_size_0
                            )
                            overlapping = ind_seg_in_image * indv_seg_alone

                            if np.sum(indv_seg_alone) < 1:
                                countable_occ.append(100)
                                print(
                                    np.sum(indv_seg_alone),
                                    "Cropped: indv_seg_alone less that 1",
                                    id,
                                    ind,
                                )
                                occluded_section = indv_seg_alone
                                visible_percentage = 0

                                area.append(int(0))
                                indv_area.append(int(0))
                            else:
                                visible_percentage = np.sum(overlapping) / np.sum(
                                    indv_seg_alone
                                )
                                occluded_section = indv_seg_alone - ind_seg_in_image
                                error_area = np.sum(occluded_section == -1)
                                if error_area / np.sum(indv_seg_alone) > 0.1:
                                    occ_error_inds.append(ind)
                                    print(
                                        f"{id=} {ind=}  perc:{100*error_area/np.sum(indv_seg_alone):.1f}% {error_area=}  area in image: {np.sum(ind_seg_in_image)}, unoccluded_area: {np.sum(indv_seg_alone)}"
                                    )
                                countable_occ.append(
                                    100 - int(visible_percentage * 100)
                                )
                                area.append(int(np.sum(ind_seg_in_image)))
                                indv_area.append(int(np.sum(indv_seg_alone)))
                                if visible_percentage > 1.01:
                                    print(
                                        f"   ERROR: bigger {100*np.sum(indv_seg_alone)/np.sum(ind_seg_in_image):.2f} ind:{ind} id:{id}, {np.sum(indv_seg_alone):.0f}, {np.sum(ind_seg_in_image):.0f}"
                                    )

                                if visible_percentage > 1.01:
                                    print(
                                        f"ERROR: cropped: bigger {100*np.sum(indv_seg_alone)/np.sum(ind_seg_in_image):.2f} ind:{ind} id:{id}, {np.sum(indv_seg_alone):.0f}, {np.sum(ind_seg_in_image):.0f}"
                                    )
                        new_centers = []
                        for center in countable["centers"]:
                            # scale to pixel size from propotion
                            center_pxlscl = center.copy()
                            center_pxlscl[0] *= seg.shape[0]
                            center_pxlscl[1] *= seg.shape[1]

                            # shift by th e crop amount
                            center_pxlscl_shifted = center_pxlscl
                            center_pxlscl_shifted[0] -= (
                                seg.shape[0] - args.crop_size
                            ) / 2
                            center_pxlscl_shifted[1] -= (
                                seg.shape[1] - args.crop_size
                            ) / 2

                            # scale back down to proportion of new image
                            center_shifted_rescaled = center_pxlscl_shifted
                            center_shifted_rescaled[0] /= args.crop_size
                            center_shifted_rescaled[1] /= args.crop_size

                            new_centers.append(center_shifted_rescaled)

                        countable[f"centers_crop{args.crop_size}"] = new_centers
                        countable[f"occlusions_crop{args.crop_size}"] = countable_occ
                        countable[f"area_crop{args.crop_size}"] = area

                with open(sv_name, "w") as f:
                    json.dump(img_info, f)
