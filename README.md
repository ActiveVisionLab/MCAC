# Multi-Class Class-Agnostic Counting Dataset
**[Project Page](https://MCAC.active.vision/) |
[ArXiv](https://arxiv.org/abs/2309.04820) |
[Download](https://www.robots.ox.ac.uk/~lav/Datasets/MCAC/MCAC.zip)
**

[Michael Hobley](https://scholar.google.co.uk/citations?user=2EftbyIAAAAJ&hl=en), 
[Victor Adrian Prisacariu](http://www.robots.ox.ac.uk/~victor/). 

[Active Vision Lab (AVL)](https://www.robots.ox.ac.uk/~lav/),
University of Oxford.

![Example Image](MCAC_example.png)
Each object in the RGB image has an associated: Model ID, Class ID, Center Coordinate, Bounding Box and Occlusion.

MCAC is the first multi-class class-agnostic counting dataset. each image contains between 1 and 4 classes of
object and between 1 and 300 objects per class.
The classes of objects present in the Train, Test and Val splits are mutually exclusive, and where possible
aligned with the class splits in [FSC-133](https://github.com/ActiveVisionLab/LearningToCountAnything).
Each object is labeled with an instance, class and model number as well as its center coordinate, bounding box
coordinates and its percentage occlusion
Models are taken from [ShapeNetSem]. The original model IDs and manually
verified category labels are preserved.
MCAC-M1 is the single-class images from MCAC. This is useful when comparing methods that are not suited to
multi-class cases.

## Download 

Dowload MCAC [here](https://www.robots.ox.ac.uk/~lav/Datasets/MCAC/MCAC.zip).

File Hierarchy:

    ├── dataset_pytorch.py
    ├── make_gaussian_maps.py
    ├── test
    ├── train
    │   ├── 1511489148409439
    │   ├── 3527550462177290
    │   |   ├──img.png
    │   |   ├──info.json
    │   |   ├──seg.png
    │   ├──4109417696451021
    │   └── ...
    └── val
  
## Precompute Density Maps 
To precompute ground truth density maps for other resolutions, occlusion percentages, and gaussian standard deviations:

```sh
cd PATH/TO/MCAC/
python make_gaussian_maps.py  --occulsion_limit <desired_max_occlusion>  --crop_size 672 --img_size <desired_resolution> --gauss_constant <desired_gaussian_std>;
```

## Evaluation Bounding Boxes
For fair evaluation of methods which require exemplar bounding boxes we suggest using the 3 least occluded instances (lowest index breaking ties).
For ease of use, we have provided the indexs for all of these for the validation and training splits.

## Citation
```
@article{hobley2023abc,
    title={ABC Easy as 123: A Blind Counter for Exemplar-Free Multi-Class Class-agnostic Counting}, 
    author={Michael A. Hobley and Victor A. Prisacariu},
    journal={arXiv preprint arXiv:2309.04820},
    year={2023},
}
```
