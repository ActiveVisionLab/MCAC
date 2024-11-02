import bpy


def cleanup_mesh(asset_id: str, source_path: str, target_path: str):
    # start from a clean slate
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.world = bpy.data.worlds.new("World")

    # import source mesh
    bpy.ops.import_scene.gltf(filepath=source_path, loglevel=50)

    bpy.ops.object.select_all(action="DESELECT")

    for obj in bpy.data.objects:
        # remove duplicate vertices
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.remove_doubles(threshold=1e-06)
        bpy.ops.object.mode_set(mode="OBJECT")
        # disable auto-smoothing
        obj.data.use_auto_smooth = False
        # split edges with an angle above 70 degrees (1.22 radians)
        m = obj.modifiers.new("EdgeSplit", "EDGE_SPLIT")
        m.split_angle = 1.22173
        bpy.ops.object.modifier_apply(modifier="EdgeSplit")
        # move every face an epsilon in the direction of its normal, to reduce clipping artifacts
        m = obj.modifiers.new("Displace", "DISPLACE")
        m.strength = 0.00001
        bpy.ops.object.modifier_apply(modifier="Displace")

    # join all objects together
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.join()

    # set the name of the asset
    bpy.context.active_object.name = asset_id

    # export cleaned up mesh
    bpy.ops.export_scene.gltf(filepath=str(target_path), check_existing=True)


pth_g = "BASE_PATH/7f6bd9a933f6cbd33585ebacb5c964c2/models/model_normalized.gltf"
pth_o = "BASE_PATH/7f6bd9a933f6cbd33585ebacb5c964c2/models/model_normalized.obj"

# bpy.ops.import_scene.obj(filepath=pth_o)
# bpy.ops.import_scene.gltf(filepath=pth_g)


# objs_pre_import = set(bpy.context.scene.objects)
## bpy.ops.import_scene.obj(filepath=shapenet_filepath)
# bpy.ops.import_scene.gltf(filepath=pth_g)
# imported_objs = set(bpy.context.scene.objects) - objs_pre_import
# imported_obj = list(imported_objs)[0]


# bpy.ops.object.select_all(action="DESELECT")

# loop all scene objects


##bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_MASS")
# imported_obj.select_set(False)
# scene = bpy.context.scene
# for obj in scene.objects:
#    if obj.type == "MESH":
#        print(obj.name)
##        obj.select_set(True)
#        obj.data.use_auto_smooth = False
#        bpy.ops.object.modifier_add(type='EDGE_SPLIT')
#        bpy.context.object.modifiers["EdgeSplit"].split_angle = 1.22173
#    #    bpy.ops.object.modifier_add(type='DISPLACE')
#    #    bpy.context.object.modifiers["Displace"].strength = 5e-05
##        obj.select_set(False)


bpy.ops.import_scene.gltf(filepath=pth_g, loglevel=50)
bpy.ops.object.select_all(action="DESELECT")


for obj in bpy.data.objects:
    # remove duplicate vertices
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.remove_doubles(threshold=1e-06)
    bpy.ops.object.mode_set(mode="OBJECT")
    # disable auto-smoothing
    obj.data.use_auto_smooth = False
    # split edges with an angle above 70 degrees (1.22 radians)
    m = obj.modifiers.new("EdgeSplit", "EDGE_SPLIT")
    m.split_angle = 1.22173
    bpy.ops.object.modifier_apply(modifier="EdgeSplit")
    # move every face an epsilon in the direction of its normal, to reduce clipping artifacts
    m = obj.modifiers.new("Displace", "DISPLACE")
    m.strength = 0.00001
    bpy.ops.object.modifier_apply(modifier="Displace")

# join all objects together
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.join()


# convert to gltf then import into blender
# disable auto-smoothing
# split edges with an angle above 70 degrees (Split Edge modifier)
# move every face an epsilon in the direction of its normal to reduce / eliminate clipping artifacts (Displace modifier with tiny strength, e.g. 1e-5)
