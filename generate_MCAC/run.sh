Cropsize=672

Config=shapenetsem_009
Config=shapenetsem_009_lower_den
Config=shapenetsem_009_med_den

# generate images, run this N times and then run the post processing
python BASE_PATH/generate.py --config $Config --split train;
python BASE_PATH/generate.py --config $Config --split test;
python BASE_PATH/generate.py --config $Config --split val;


# post process and generate associated labels
python BASE_PATH/delete_non_completes.py --config $Config ;
python BASE_PATH/rename_files.py --config $Config
python BASE_PATH/compress_seginds.py  --config $Config;
python BASE_PATH/occlusion_finder.py --config $Config --crop_size $Cropsize;
python BASE_PATH/make_bboxes_json.py --config $Config --crop_size $Cropsize;
python BASE_PATH/make_gaussian_maps.py  --config $Config --occulsion_limit 70 --non_int_count  --crop_size $Cropsize --img_size 384;



