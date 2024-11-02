import math
import os
import random
import sys

import bpy
import bpy_extras

sys.path.append("BASE_PATH/")

from contextlib import contextmanager


@contextmanager
def stdout_redirected(to=os.devnull):
    """
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    """
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


def get_random_color(pastel_factor=0, col_mins=[0, 0, 0], col_maxs=[1, 1, 1]):
    col = [
        (x + pastel_factor) / (1.0 + pastel_factor)
        for x in [
            random.uniform(c_min, c_max) for (c_min, c_max) in zip(col_mins, col_maxs)
        ]
    ]
    col.extend([1])
    return col


def color_distance(c1, c2):
    return sum([abs(x[0] - x[1]) for x in zip(c1, c2)])


def generate_new_color(
    existing_colors, pastel_factor=0, col_mins=[0, 0, 0], col_maxs=[1, 1, 1]
):
    max_distance = None
    best_color = None
    for _ in range(0, 10):
        color = get_random_color(
            pastel_factor=pastel_factor, col_mins=col_mins, col_maxs=col_maxs
        )
        if not existing_colors:
            return color
        best_distance = min([color_distance(color, c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color


def new_colour_noisey_colour_noisey_roughness(
    noise_scale=12,
    all_colours=[],
    maximally_different_colours=False,
    col_mins=[0, 0, 0],
    col_maxs=[1, 1, 1],
):

    # give floor materials
    floor_mat = bpy.data.materials.new(name="floor_mat")
    floor_mat.use_nodes = True
    nodes = floor_mat.node_tree.nodes
    mat_output = nodes.get("Material Output")
    bsdf = nodes.get("Principled BSDF")

    noise_texture = nodes.new("ShaderNodeTexNoise")
    noise_texture.inputs[2].default_value = noise_scale
    colour_ramp = nodes.new("ShaderNodeValToRGB")
    if maximally_different_colours:
        col_a = generate_new_color(all_colours, col_mins=col_mins, col_maxs=col_maxs)
        col_b = generate_new_color(all_colours, col_mins=col_mins, col_maxs=col_maxs)
    else:
        if col_mins == col_maxs:
            c = [col_maxs[0], col_maxs[1], col_maxs[2], 1]
            col_a = col_b = c
        else:
            col_a = get_random_color()
            col_b = get_random_color()

    colour_ramp.color_ramp.elements[0].color = col_a
    colour_ramp.color_ramp.elements[1].color = col_b

    # roughness
    noise_texture_roughness = nodes.new("ShaderNodeTexNoise")
    noise_texture_roughness.inputs[2].default_value = noise_scale
    colour_ramp_roughness = nodes.new("ShaderNodeValToRGB")
    roughness_min = random.uniform(0.2, 0.6)
    roughness_max = random.uniform(0.2, 0.6)
    colour_ramp_roughness.color_ramp.elements[0].color = (
        roughness_min,
        roughness_min,
        roughness_min,
        1,
    )
    colour_ramp_roughness.color_ramp.elements[1].color = (
        roughness_max,
        roughness_max,
        roughness_max,
        1,
    )

    floor_mat_links = floor_mat.node_tree.links

    new_link = floor_mat_links.new(noise_texture.outputs[0], colour_ramp.inputs[0])
    new_link = floor_mat_links.new(colour_ramp.outputs[0], bsdf.inputs[0])

    # randomise the noise seed (translate a random amount)
    noise_mapping = nodes.new("ShaderNodeMapping")
    noise_object_info = nodes.new("ShaderNodeObjectInfo")
    noise_uvmap = nodes.new("ShaderNodeUVMap")
    noise_math = nodes.new("ShaderNodeMath")
    noise_math.operation = "MULTIPLY"
    noise_math.inputs[1].default_value = random.random() * 10000
    new_link = floor_mat_links.new(noise_mapping.outputs[0], noise_texture.inputs[0])
    new_link = floor_mat_links.new(noise_uvmap.outputs[0], noise_mapping.inputs[0])
    new_link = floor_mat_links.new(noise_object_info.outputs[5], noise_math.inputs[0])
    new_link = floor_mat_links.new(noise_math.outputs[0], noise_mapping.inputs[1])

    new_link = floor_mat_links.new(
        noise_texture_roughness.outputs[0], colour_ramp_roughness.inputs[0]
    )
    new_link = floor_mat_links.new(colour_ramp_roughness.outputs[0], bsdf.inputs[9])

    new_link = floor_mat_links.new(bsdf.outputs[0], mat_output.inputs[0])

    return floor_mat, [col_a, col_b]


def move_to_collection(obj_i, obj_i_collection):
    for other_col in obj_i.users_collection:
        other_col.objects.unlink(obj_i)
    if obj_i.name not in obj_i_collection.objects:
        obj_i_collection.objects.link(obj_i)
    bpy.context.view_layer.objects.active = obj_i


def get_pixel_location_of_point(coord, camera, scene):
    # local to global coordinates
    co = coord
    # calculate 2d image coordinates
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, camera, co)
    return co_2d


def make_planes(floor_size, collection, floor_mat):
    # create planes
    bpy.ops.mesh.primitive_plane_add()
    floor_plane = bpy.context.active_object
    floor_plane.dimensions = [floor_size, floor_size, 0]
    floor_plane.data.materials.append(floor_mat)
    move_to_collection(floor_plane, collection)

    bpy.ops.mesh.primitive_plane_add()
    roof_plane = bpy.context.active_object
    roof_plane.dimensions = [floor_size, floor_size, 0]
    roof_plane.location = [
        0,
        0,
        floor_size,
    ]
    roof_plane.hide_render = True

    move_to_collection(roof_plane, collection)

    sizes = [[0, 1, 1], [0, -1, 1], [-1, 0, 1], [1, 0, 1]]
    rotation_angles = [0, 0, 1, 1]

    for angle, siz in zip(rotation_angles, sizes):
        bpy.ops.mesh.primitive_plane_add()
        wall_plane = bpy.context.active_object
        wall_plane.location = [
            floor_size / 2 * siz[0],
            floor_size / 2 * siz[1],
            floor_size / 2 * siz[2],
        ]
        wall_plane.dimensions = [floor_size, floor_size, 0]
        wall_plane.rotation_euler[angle] = math.radians(90)
        move_to_collection(wall_plane, collection)

    move_to_collection(wall_plane, collection)


def load_primitive(dict_i):
    if dict_i["type"] == "cube":
        bpy.ops.mesh.primitive_cube_add()
    elif dict_i["type"] == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add()
    elif dict_i["type"] == "cone":
        bpy.ops.mesh.primitive_cone_add()
    elif dict_i["type"] == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add()
    elif dict_i["type"] == "torus":
        bpy.ops.mesh.primitive_torus_add()
    elif dict_i["type"] == "monkey":
        bpy.ops.mesh.primitive_monkey_add()

    obj_i = bpy.context.active_object
    return obj_i
