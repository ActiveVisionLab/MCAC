import os
import random

import bpy
import numpy as np


def cleanup_mesh(asset_id: str, source_path: str, target_path: str, random_disp=False):
    # start from a clean slate
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.world = bpy.data.worlds.new("World")

    # import source mesh
    bpy.ops.import_scene.gltf(filepath=source_path, loglevel=50)

    bpy.ops.object.select_all(action="DESELECT")

    name = bpy.data.objects[0].name
    for obj in bpy.data.objects:
        # remove duplicate vertices
        bpy.context.view_layer.objects.active = obj
        print(obj.dimensions)

        # # disable auto-smoothing
        obj.data.use_auto_smooth = False
        # # split edges with an angle above 70 degrees (1.22 radians)
        m = obj.modifiers.new("EdgeSplit", "EDGE_SPLIT")
        m.split_angle = 1.22173
        bpy.ops.object.modifier_apply(modifier="EdgeSplit")
        # move every face an epsilon in the direction of its normal, to reduce clipping artifacts
        m = obj.modifiers.new("Displace", "DISPLACE")
        if random_disp:
            m.strength = random.random() * 0.000015
        else:
            # m.strength = np.mean(obj.dimensions) * 0.00001
            m.strength = np.mean(obj.dimensions) * 0.01
            # m.strength = 1
        bpy.ops.object.modifier_apply(modifier="Displace")

    bpy.ops.object.select_all(action="SELECT")

    if len(bpy.data.objects) > 1:
        # join all objects together
        bpy.ops.object.join()

    # set the name of the asset
    bpy.context.active_object.name = name

    # export cleaned up mesh
    bpy.ops.export_scene.gltf(
        filepath=str(target_path), check_existing=True, export_format="GLTF_EMBEDDED"
    )


root_path = "PATH/ShapeNetSem_models-OBJ/models"
suffix = "_fixed2.GLTF"
model_files = os.listdir(root_path)
glt_files = [f for f in model_files if f.endswith(".gltf")]
for file in glt_files:

    asset_id = file.split(".")[0]
    source_path = os.path.join(root_path, file)
    target_path = os.path.join(root_path, asset_id + suffix)

    print(source_path, target_path)
    cleanup_mesh(
        asset_id=asset_id,
        source_path=source_path,
        target_path=target_path,
    )
