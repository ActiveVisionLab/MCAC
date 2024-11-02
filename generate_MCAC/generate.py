import argparse
import json
import math
import os
import random
import shutil
import sys
from datetime import datetime

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from tqdm import tqdm

sys.path.append("BASE_PATH/")

import platform
import time

import yaml

from configs.ConfigClass import ConfigClass
from utils.blender_utils import *

if platform.platform().startswith("macOS"):
    base_address = "/Users/mahobley/blender/toy_dataset/"

elif platform.platform().startswith("Linux"):

    base_address = "BASE_PATH/"

start = time.perf_counter()

print("\n\n")
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--config", type=str, default="003", help="which dataset config")
parser.add_argument(
    "--split",
    type=str,
    default="config_spec",
    help="train, test or val, if config_spec then dont add an extra bit onto the config filename",
)
parser.add_argument("--seed", type=int, default=-1, help="if specific seed is wanted")
args = parser.parse_args()
print(f"{args=}")
if os.path.basename(sys.executable) == "python3.10":
    print("BEING RUN IN BLENDER")
    being_run_in_blender = True
else:
    print("Being run in terminal")
    being_run_in_blender = False

if being_run_in_blender:
    args.config = "shapenetsem_007_high_density"
    args.split = "test"
    args.seed = 7311808234164844

defualt_stream = open(f"{base_address}configs/__DEFAULTS__.yml", "r")
configs = yaml.load(defualt_stream, Loader=yaml.Loader)
if args.seed == -1:
    random_seed = random.randint(0, 10000000000000000)
    print(f"seed {random_seed}")
else:
    print(f"USING STATIC SEED: {args.seed}")
    random_seed = args.seed
random.seed(random_seed)

specific_stream = open(f"{base_address}configs/{args.config}.yml", "r")
specific_dictionary = yaml.load(specific_stream, Loader=yaml.Loader)

CFG = ConfigClass(configs, specific_dictionary)

scene = bpy.context.scene
bpy.data.scenes.new("Scene")
bpy.data.scenes.remove(scene, do_unlink=True)

if CFG.dataset.split == "arg_spec":
    CFG.dataset.split = args.split
    CFG.dataset.savename = CFG.dataset.savename + "_" + args.split

# in order to tell them apart you should be using colour depth 16
if CFG.scene.colour_depth != 16:
    if CFG.objects.max_number_per_type * CFG.objects.max_num_countables > 255:
        print("ERROR SHOULD BE USING COLOUR DEPTH 16")

countables = []
info_dict = {"seed": random_seed}
all_colours = []

floor_mat, floor_col = new_colour_noisey_colour_noisey_roughness(
    col_mins=CFG.scene.floor_min_colours,
    col_maxs=CFG.scene.floor_max_colours,
)
info_dict["floor_col"] = floor_col
all_colours.extend(floor_col)

filepath = base_address + "ims/"
if CFG.dataset.savename != "":
    filepath = base_address + "ims/" + CFG.dataset.savename + "/"
if not os.path.exists(filepath):
    os.makedirs(filepath)

now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")

if not os.path.exists(f"{filepath}{random_seed}"):
    os.makedirs(f"{filepath}{random_seed}")
    print("made", f"{filepath}{random_seed}")
else:
    print("Exists already", f"{filepath}{random_seed}")


scene = bpy.context.scene
scene.frame_start = CFG.simulation.start_frame
scene.frame_end = CFG.simulation.end_frame
# create camera
camera_data = bpy.data.cameras.new(name="Camera")
camera_object = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera_object)
camera_object.location[2] = CFG.scene.camera.location.z
scene.render.resolution_x = CFG.rendering.resolution.x
scene.render.resolution_y = CFG.rendering.resolution.y
scene.camera = camera_object

min_location_x = (
    -(CFG.scene.floor_size / 2 - CFG.objects.max_size)
    if CFG.objects.min_location_x == "None"
    else CFG.objects.min_location_x
)
max_location_x = (
    (CFG.scene.floor_size / 2 - CFG.objects.max_size)
    if CFG.objects.max_location_x == "None"
    else CFG.objects.max_location_x
)
min_location_y = (
    -(CFG.scene.floor_size / 2 - CFG.objects.max_size)
    if CFG.objects.min_location_y == "None"
    else CFG.objects.min_location_y
)
max_location_y = (
    (CFG.scene.floor_size / 2 - CFG.objects.max_size)
    if CFG.objects.max_location_y == "None"
    else CFG.objects.max_location_y
)


# Create new object, pass the light data

lights_collection = bpy.data.collections.new("lights")
lights = []
bpy.context.scene.collection.children.link(lights_collection)

lights_brightnesses = []
for i in range(
    random.randint(CFG.scene.lights.min_number, CFG.scene.lights.max_number)
):
    if CFG.rendering.image_renderer == "cycles":
        lights_brightnesses.append(
            random.uniform(
                CFG.scene.lights.cycles_min_energy, CFG.scene.lights.cycles_max_energy
            )
        )
    else:
        lights_brightnesses.append(
            random.uniform(
                CFG.scene.lights.eevee_min_energy, CFG.scene.lights.eevee_max_energy
            )
        )

lights_brightnesses = np.array(lights_brightnesses)

if CFG.rendering.image_renderer == "cycles":
    global_min_total = CFG.scene.lights.cycles_global_min_total

else:
    global_min_total = CFG.scene.lights.eevee_global_min_total

if np.sum(lights_brightnesses) < global_min_total:
    lights_brightnesses = lights_brightnesses * (
        global_min_total / np.sum(lights_brightnesses)
    )

for i, brightness in enumerate(lights_brightnesses):
    # Create light datablock
    light_data = bpy.data.lights.new(name="my-light-data", type="POINT")
    light_data.energy = brightness
    light_object = bpy.data.objects.new(name="my-light", object_data=light_data)
    # Link object to collection in context
    lights_collection.objects.link(light_object)
    # Change light position
    x_loc = random.uniform(min_location_x, max_location_x)
    y_loc = random.uniform(min_location_y, max_location_y)
    light_object.location = (x_loc, y_loc, CFG.scene.lights.height)
    # print(light_data.energy)
    lights.append([x_loc, y_loc, CFG.scene.lights.height, light_data.energy])

box_collection = bpy.data.collections.new("box")
bpy.context.scene.collection.children.link(box_collection)
make_planes(CFG.scene.floor_size, box_collection, floor_mat=floor_mat)

for plane_i in box_collection.objects:
    bpy.context.view_layer.objects.active = plane_i
    bpy.ops.rigidbody.object_add()

    bpy.context.object.rigid_body.type = "PASSIVE"
    plane_i.rigid_body.restitution = 0.0

if CFG.scene.randomly_move_floor_vertexs:
    #    add subsurf, apply then move the vertices
    floor = box_collection.objects[0]
    floor.modifiers.new("My SubDiv", "SUBSURF")
    floor.modifiers["My SubDiv"].levels = 4
    floor.modifiers["My SubDiv"].subdivision_type = "SIMPLE"
    # set the object to active
    bpy.context.view_layer.objects.active = floor
    bpy.ops.object.modifier_apply(modifier="My SubDiv")
    for vert in floor.data.vertices:
        vert.co[2] += random.uniform(0, CFG.scene.randomly_move_floor_vertexs_amount)

bpy.context.scene.frame_set(0)
object_collections = []
j = 0
num_countables = random.randint(
    CFG.objects.min_num_countables, CFG.objects.max_num_countables
)

if CFG.dataset.dataset == "primitives":
    with open(os.path.join(CFG.dataset.types_split_path), "r") as file:
        type_split = json.load(file)
    type_split = dict(type_split)
    object_types = type_split[CFG.dataset.split]

    if CFG.objects.ensure_different_geometries:
        if num_countables > len(object_types):
            print(
                f"Cannot ensure different geometries as there are {len(object_types)} types to draw {num_countables} from"
            )
            exit()
    available_types = set(object_types)

    if CFG.objects.in_image_colour_range != -1:
        object_min_colours = [
            random.uniform(
                CFG.objects.object_min_colours[0],
                CFG.objects.object_max_colours[0] - CFG.objects.in_image_colour_range,
            ),
            random.uniform(
                CFG.objects.object_min_colours[1],
                CFG.objects.object_max_colours[1] - CFG.objects.in_image_colour_range,
            ),
            random.uniform(
                CFG.objects.object_min_colours[2],
                CFG.objects.object_max_colours[2] - CFG.objects.in_image_colour_range,
            ),
        ]
        object_max_colours = [
            object_min_colours[0] + CFG.objects.in_image_colour_range,
            object_min_colours[1] + CFG.objects.in_image_colour_range,
            object_min_colours[2] + CFG.objects.in_image_colour_range,
        ]
    else:
        object_min_colours = CFG.objects.object_min_colours
        object_max_colours = CFG.objects.object_max_colours

elif CFG.dataset.dataset == "shapenet":
    with open(
        os.path.join(CFG.dataset.dataset_path, "cat_folder_pairs.json"), "r"
    ) as file:
        superclasses_mapping = json.load(file)
    superclasses_mapping = dict(superclasses_mapping)

    with open(os.path.join(CFG.dataset.types_split_path), "r") as file:
        type_split = json.load(file)
    type_split = dict(type_split)
    object_types = type_split[CFG.dataset.split]

elif CFG.dataset.dataset == "shapenetsem":
    print(CFG.dataset.dataset)

    with open(os.path.join(base_address, CFG.dataset.types_split_path), "r") as file:
        type_split = json.load(file)
    with open(os.path.join(base_address, CFG.dataset.types_id_class), "r") as file:
        class_id = json.load(file)
    type_split = dict(type_split)
    class_id = dict(class_id)
    object_types = type_split[CFG.dataset.split]


start_making_countables = time.perf_counter()
j_n = 0
for i in range(num_countables):

    start_making_countable_i = time.perf_counter()
    dict_i = {}

    if CFG.dataset.dataset == "primitives":
        dict_i["mesh_path"] = ""
        dict_i["mesh"] = None
        class_type = random.choice(list(available_types))
        if CFG.objects.ensure_different_geometries:
            available_types.remove(class_type)
        dict_i["type"] = class_type
    elif CFG.dataset.dataset == "shapenet":
        superclass_id = random.choice(object_types)
        print(f"{superclass_id=}")
        superclass = superclasses_mapping[superclass_id]
        subclasses = os.listdir(os.path.join(CFG.dataset.dataset_path, superclass))
        subclass = random.choice(subclasses)
        print(f"  super: {superclass},{superclass_id} sub:{subclass}")
        dict_i["superclass"] = superclass
        dict_i["superclass_id"] = superclass_id
        dict_i["subclass"] = subclass
    elif CFG.dataset.dataset == "shapenetsem":
        obj_id = random.choice(object_types)
        print(f"  obj_id: {obj_id}")
        dict_i["obj_id"] = obj_id
        dict_i["obj_class"] = class_id[obj_id]

    nominal_size = random.uniform(CFG.objects.min_size, CFG.objects.max_size)
    dict_i["sizing"] = [
        nominal_size * (1 - CFG.objects.size_variation),
        nominal_size * (1 + CFG.objects.size_variation),
    ]
    dict_i["instances"] = []
    dict_i["centers"] = []
    dict_i["verticies"] = []
    dict_i["inds"] = []
    if CFG.dataset.dataset == "primitives":
        mat, col = new_colour_noisey_colour_noisey_roughness(
            noise_scale=5,
            all_colours=all_colours,
            maximally_different_colours=CFG.scene.maximally_different_colours,
            col_mins=object_min_colours,
            col_maxs=object_max_colours,
        )
        all_colours.extend(col)
        dict_i["colour"] = col
        dict_i["material"] = mat
    countables.append(dict_i)

    num_of_type_x = random.randint(
        CFG.objects.min_number_per_type, CFG.objects.max_number_per_type
    )
    print(f"  num_of_type_x {num_of_type_x}")
    objs_class_x = []
    for i in range(num_of_type_x):
        obj_i_collection = bpy.data.collections.new(f"obj{j}")

        object_collections.append(obj_i_collection)
        bpy.context.scene.collection.children.link(obj_i_collection)

        if CFG.dataset.dataset == "primitives":
            obj_i = load_primitive(dict_i)
        elif CFG.dataset.dataset == "shapenet":
            if i == 0:
                objs_pre_import = set(bpy.context.scene.objects)
                shapenet_filepath = os.path.join(
                    CFG.dataset.dataset_path,
                    superclass,
                    subclass,
                    "models",
                    f"model_normalized{CFG.dataset.file_type}",
                )
                # bpy.ops.import_scene.obj(filepath=shapenet_filepath)

                # todo put back
                print("PUTBACK")
                pth_g = "BASE_PATH/7f6bd9a933f6cbd33585ebacb5c964c2/models/model_normalized.gltf"
                bpy.ops.import_scene.gltf(filepath=pth_g)

                imported_objs = set(bpy.context.scene.objects) - objs_pre_import
                imported_obj = list(imported_objs)[0]
                objs_class_x.append(imported_obj)
                obj_i = imported_obj
            else:
                obj_i = imported_obj.copy()
                obj_i.data = imported_obj.data.copy()
                bpy.context.collection.objects.link(obj_i)
                objs_class_x.append(obj_i)
        elif CFG.dataset.dataset == "shapenetsem":
            if i == 0:
                objs_pre_import = set(bpy.context.scene.objects)
                shapenet_filepath = os.path.join(
                    CFG.dataset.dataset_path,
                    f"{obj_id}{CFG.dataset.file_type}",
                )
                with stdout_redirected():
                    bpy.ops.import_scene.gltf(filepath=shapenet_filepath)
                imported_objs = set(bpy.context.scene.objects) - objs_pre_import
                imported_obj = list(imported_objs)[0]
                print(
                    f"  vertices:{len(imported_obj.data.vertices)} edges:{len(imported_obj.data.edges)} polygons:{len(imported_obj.data.polygons)}"
                )

                if len(imported_obj.data.vertices) > 100_000:
                    error_str = f"{len(imported_obj.data.vertices)}"
                    with open(
                        f"BASE_PATH/too_many_indicies/{args.config}__{random_seed}.txt",
                        "w",
                    ) as f:
                        f.write(error_str)
                    print("KILLING AS TOO MANY VERTICIES")
                    print(f"Deleting {filepath}{random_seed}")
                    os.rmdir(f"{filepath}{random_seed}")
                    exit()
                imported_obj.data.use_auto_smooth = False
                # deselect all of the objects
                bpy.ops.object.select_all(action="DESELECT")

                # loop all scene objects

                imported_obj.select_set(True)
                bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_MASS")
                imported_obj.select_set(False)

                for mat in bpy.data.materials:
                    # If the material has a node tree
                    if mat.node_tree:
                        #     # Run through all nodes
                        for node in mat.node_tree.nodes:
                            #         # If the node type is Ambient Occlusion
                            if node.type == "BSDF_PRINCIPLED":
                                # set alpha to 1
                                node.inputs[21].default_value = 1.0

                    mat.use_backface_culling = False

                objs_class_x.append(imported_obj)
                obj_i = imported_obj
            else:
                obj_i = imported_obj.copy()
                obj_i.data = imported_obj.data.copy()
                imported_obj.data.use_auto_smooth = False

                bpy.context.collection.objects.link(obj_i)
                objs_class_x.append(obj_i)

                imported_obj.select_set(True)
                imported_obj.select_set(False)

        for other_col in obj_i.users_collection:
            other_col.objects.unlink(obj_i)
        if obj_i.name not in obj_i_collection.objects:
            obj_i_collection.objects.link(obj_i)

        bpy.context.view_layer.objects.active = obj_i

        dict_i["instances"].append(obj_i)
        obj_i.pass_index = j + 1
        dict_i["inds"].append(j)

        #    set location etc
        x_loc = random.uniform(min_location_x, max_location_x)
        y_loc = random.uniform(min_location_y, max_location_y)

        size = random.uniform(dict_i["sizing"][0], dict_i["sizing"][1])
        max_size = max(obj_i.dimensions[:3])
        if max_size == 0:
            print("obj_i", obj_i.name, obj_i.dimensions[:3])
        scale_factor = size / max_size
        new_size = [dim * scale_factor for dim in obj_i.dimensions[:3]]
        obj_i.dimensions = new_size

        # obj_i.dimensions = [size, size, size]
        obj_i.location[0] = x_loc
        obj_i.location[1] = y_loc
        obj_i.location[2] = 10

        obj_i.rotation_mode = "XYZ"

        obj_i.rotation_euler[0] = math.radians(
            random.uniform(CFG.objects.min_angle, CFG.objects.max_angle)
        )
        obj_i.rotation_euler[1] = math.radians(
            random.uniform(CFG.objects.min_angle, CFG.objects.max_angle)
        )
        obj_i.rotation_euler[2] = math.radians(
            random.uniform(CFG.objects.min_angle, CFG.objects.max_angle)
        )

        if CFG.objects.randomly_move_vertexs:
            for vert in obj_i.data.vertices:
                for co_i in range(3):
                    vert.co[co_i] += random.uniform(
                        -CFG.objects.randomly_move_vertexs_amount / 2,
                        CFG.objects.randomly_move_vertexs_amount / 2,
                    )

        bpy.ops.rigidbody.object_add()
        if CFG.dataset.dataset == "primitives":

            obj_i.data.materials.append(dict_i["material"])
        j += 1
    print(
        f"  time to make countable {j_n}*{num_of_type_x}: {time.perf_counter() - start_making_countable_i:.2f}s \n"
    )
    j_n += 1
print(
    f"--time to make all countables: {time.perf_counter() - start_making_countables:.2f}s"
)
for cnt_i, countable in enumerate(countables):
    for obj_i in countable["instances"]:
        bpy.ops.object.select_all(action="DESELECT")
        obj_i.select_set(True)

        bpy.context.view_layer.objects.active = obj_i

        bpy.ops.object.transform_apply(location=False, scale=True, rotation=True)
        obj_i.rigid_body.restitution = 0.0
        obj_i.select_set(False)
        bpy.ops.object.select_all(action="DESELECT")
print("Running Simulation")

scene.rigidbody_world.point_cache.frame_end = CFG.simulation.end_frame
sim_start = time.perf_counter()
# run the simulation
# for f in range(scene.frame_end):
for f in tqdm(range(scene.frame_end)):
    # print(f / scene.frame_end)
    scene.frame_set(f)
    # if taken more than 2.5 minutes and not a quater done kill it
    if time.perf_counter() - sim_start > 120 and f / scene.frame_end < 0.05:
        print("ERROR Simulation taking too long")
        error_str = f"{time.perf_counter() - sim_start:.2f}s to do {100 * f / scene.frame_end}% of the simulation, estimated simulation length {(time.perf_counter() - sim_start)/(f *60 / scene.frame_end):.2f}m"
        print(error_str)
        with open(
            f"BASE_PATH/time_out_seeds/{args.config}__{random_seed}.txt",
            "w",
        ) as f:
            f.write(error_str)
        exit()

print(f"--time to simulate im: {time.perf_counter() - sim_start:.2f}")

if CFG.rendering.output_video:

    #    render a video of each frame
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.filepath = f"{filepath}{random_seed}/anim"
    bpy.ops.render.render(write_still=1, animation=True)


def set_gpu_cycles(CFG, scene):
    scene.render.engine = "CYCLES"
    scene.cycles.preview_samples = CFG.rendering.cycles_samples
    scene.cycles.samples = CFG.rendering.cycles_samples
    bpy.context.scene.cycles.use_denoising = CFG.rendering.cycles_denoise

    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = (
        "CUDA"  # or "OPENCL"
    )
    for scene_i in bpy.data.scenes:
        scene_i.cycles.device = "GPU"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        if d["name"] == "Intel Core i7-6850K CPU @ 3.60GHz":
            d["use"] = 0  # dont use CPU
        else:
            d["use"] = 1  # Using all GPU devices

    print(
        f'  Using {set([d["name"] for d in bpy.context.preferences.addons["cycles"].preferences.devices if d["use"]])}  Not Using {[d["name"] for d in bpy.context.preferences.addons["cycles"].preferences.devices if not d["use"]]} '
    )


if CFG.rendering.image_renderer == "cycles":
    set_gpu_cycles(CFG, scene)
elif CFG.rendering.image_renderer == "eevee":
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = CFG.rendering.eevee_image_samples
    scene.eevee.taa_samples = CFG.rendering.eevee_image_samples


scene.view_layers["ViewLayer"].use_pass_object_index = True
scene.use_nodes = True
scene_nodes = scene.node_tree.nodes
scene_links = scene.node_tree.links

if CFG.rendering.render_output_image:
    render_im_time = time.perf_counter()

    render_layers = scene_nodes.new("CompositorNodeRLayers")
    output_file = scene_nodes.new("CompositorNodeOutputFile")
    output_file.base_path = f"{filepath}{random_seed}/img"
    output_file.file_slots[0].path = f"../img"
    new_link = scene_links.new(render_layers.outputs[0], output_file.inputs[0])

    with stdout_redirected():
        bpy.ops.render.render(write_still=1, animation=False)

    for node in scene_nodes:
        scene_nodes.remove(node)
    print(f"--time to render im: {time.perf_counter() - render_im_time:.2f}")


trees = []
start_get_coords = time.perf_counter()
for cnt_i, countable in enumerate(countables):
    countable_centers = []
    countable_centers_2 = []
    countable_verts = []
    for obj_i in countable["instances"]:
        pixel_coords = get_pixel_location_of_point(
            obj_i.matrix_world.translation, scene.camera, scene
        )
        countable_centers_2.append(pixel_coords[:])

        mw = obj_i.matrix_world

        if CFG.rendering.find_Vertex_collisions:
            # global vert locs
            verts = [mw @ v.co for v in obj_i.data.vertices]
            # vert locations in "region camera coords"
            vert_pixel_coords = [
                get_pixel_location_of_point(v, scene.camera, scene) for v in verts
            ]
            verts_px_list = [vp[:2] for vp in vert_pixel_coords]
            countable_verts.append(verts_px_list)

            depsgraph = (
                bpy.context.evaluated_depsgraph_get()
            )  # getting the dependency graph
            max_dist = 100
            limit = 0.01
            vert_collision = []
            for i, v in enumerate(verts):
                #
                co2D = world_to_camera_view(scene, scene.camera, v)
                if 0.0 <= co2D.x <= 1.0 and 0.0 <= co2D.y <= 1.0 and co2D.z > 0:
                    # Try a ray cast, in order to test the vertex visibility from the camera
                    #                    location= scene.ray_cast(bpy.context.window.view_layer, scene.camera.location, (v - scene.camera.location).normalized())
                    location = scene.ray_cast(
                        depsgraph,
                        scene.camera.location,
                        (v - scene.camera.location).normalized(),
                        distance=(v - scene.camera.location).length + 2 * limit,
                    )
                    # If the ray hits something and if this hit is close to the vertex, we assume this is the vertex
                    if location[0] and location[4].name != obj_i.name:
                        #                        print(v, location)
                        #                        print("4", location[4].type)
                        if "Plane" in location[4].name:
                            # not a collision with an object
                            vert_collision.append(0)
                        else:
                            #                            print(location[4].name,  obj_i.name)
                            obj_i.data.vertices[i].select = True
                            vert_collision.append(1)
                    else:
                        vert_collision.append(0)
                else:
                    #                   #not in frame
                    vert_collision.append(1)
            print(
                obj_i.name,
                sum(vert_collision) / len(vert_collision),
                sum(vert_collision),
                len(vert_collision),
            )
            if (sum(vert_collision) / len(vert_collision)) >= 0.75:
                print("big", obj_i.name)
            if (sum(vert_collision) / len(vert_collision)) <= 0.25:
                print("small", obj_i.name)

    countable["centers"] = countable_centers_2
    # countable["verticies"] = countable_verts
    countables[cnt_i] = countable

print(f"--time to get coords: {time.perf_counter() - start_get_coords:.2f}s")
if CFG.rendering.render_segmentation:
    render_seg_time = time.perf_counter()

    set_gpu_cycles(CFG, scene)

    render_layers = scene_nodes.new("CompositorNodeRLayers")
    output_seg = scene_nodes.new("CompositorNodeOutputFile")
    output_seg.base_path = f"{filepath}{random_seed}_seg"
    output_seg.format.color_mode = "BW"
    output_seg.file_slots[0].path = f"../{random_seed}/seg"

    math_255_1 = scene_nodes.new("CompositorNodeMath")
    math_255_1.operation = "DIVIDE"

    if CFG.scene.colour_depth == 16:
        output_seg.format.color_depth = "16"
        math_255_1.inputs[1].default_value = 65535
    else:
        if CFG.objects.max_number_per_type * CFG.objects.max_num_countables > 255:
            print("ERROR SHOULD BE USING COLOUR DEPTH 16")
            output_seg.format.color_depth = "16"
            math_255_1.inputs[1].default_value = 65535
        else:
            math_255_1.inputs[1].default_value = 255

    new_link = scene_links.new(render_layers.outputs[2], math_255_1.inputs[0])
    new_link = scene_links.new(math_255_1.outputs[0], output_seg.inputs[0])
    with stdout_redirected():
        bpy.ops.render.render(write_still=1, animation=False)
    for node in scene_nodes:
        scene_nodes.remove(node)

    if CFG.rendering.image_renderer == "eevee":
        scene.render.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = CFG.rendering.eevee_image_samples
        scene.eevee.taa_samples = CFG.rendering.eevee_image_samples

    print(f"--time to render seg: {time.perf_counter() - render_seg_time:.2f}s")


if CFG.rendering.render_individual_segmentation:

    print("Rendering individual unoccluded masks")
    render_indv_seg_time = time.perf_counter()
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = CFG.rendering.indv_seg_samples
    scene.eevee.taa_samples = CFG.rendering.indv_seg_samples

    for i, col in tqdm(enumerate(object_collections), total=len(object_collections)):
        object_i_layer = bpy.ops.scene.view_layer_add(type="NEW")
        bpy.context.view_layer.name = f"Object_{i}"
        bpy.context.window.view_layer = bpy.context.scene.view_layers[f"Object_{i}"]
        render_segmenation = scene_nodes.new("CompositorNodeRLayers")
        render_segmenation.layer = f"Object_{i}"

        for col_j_name in bpy.context.layer_collection.children.keys():
            if col_j_name == col.name:
                bpy.context.layer_collection.children[col_j_name].exclude = False
                bpy.context.layer_collection.children[col_j_name].holdout = True

            else:
                bpy.context.layer_collection.children[col_j_name].exclude = True
                bpy.context.layer_collection.children[col_j_name].holdout = False

        output_seg_indv = scene_nodes.new("CompositorNodeOutputFile")
        output_seg_indv.base_path = f"{filepath}{random_seed}/segind"
        output_seg_indv.format.color_mode = "BW"
        output_seg_indv.file_slots[0].path = f"../seginds/{i}_"

        colour_ramp = scene_nodes.new("CompositorNodeValToRGB")
        colour_ramp.color_ramp.elements[1].position = 0.01
        colour_ramp.color_ramp.elements[0].color = (1, 1, 1, 1)
        colour_ramp.color_ramp.elements[1].color = (0, 0, 0, 1)

        new_link = scene_links.new(render_segmenation.outputs[1], colour_ramp.inputs[0])
        new_link = scene_links.new(colour_ramp.outputs[0], output_seg_indv.inputs[0])
        bpy.context.window.view_layer = bpy.context.scene.view_layers[f"ViewLayer"]

        with stdout_redirected():
            bpy.ops.render.render(write_still=1, animation=False)

        bpy.context.scene.view_layers.remove(
            bpy.context.scene.view_layers[f"Object_{i}"]
        )

        for node in scene_nodes:
            scene_nodes.remove(node)

        if (time.perf_counter() - render_indv_seg_time) > 2 * 60:
            estimated_time = (
                (len(object_collections) / i)
                * ((time.perf_counter() - render_indv_seg_time))
                / 60
            )
            if estimated_time > 11:
                print("TAKING TOO LONG KILLING")
                error_str = f"{args.config} {random_seed}, {(time.perf_counter() - render_indv_seg_time):.2f}s or {(time.perf_counter() - render_indv_seg_time)/60:.2f}mins to do {i}/{len(object_collections)} predicted time: {estimated_time:.2f}mins"
                shutil.rmtree((f"{filepath}{random_seed}"))
                with open(
                    f"BASE_PATH/timeout_indvsegs/{args.config}__{random_seed}.txt",
                    "w",
                ) as f:
                    f.write(error_str)
                exit()

    print(
        f"--time to render indv_segs: {time.perf_counter() - render_indv_seg_time:.2f}s"
    )

# strip out instances
for v in countables:
    v.pop("instances", None)
    v.pop("material", None)
    v.pop("mesh", None)

info_dict["countables"] = countables
info_dict["lights"] = lights

with open(
    f"{filepath}{random_seed}/info{scene.frame_end-1:04}.json",
    "w",
) as fp:
    json.dump(info_dict, fp)

print(
    f"--total time: {time.perf_counter() - start:.2f}s or {(time.perf_counter() - start)/60:.2f}m"
)
if not being_run_in_blender:
    exit()
