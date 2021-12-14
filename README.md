<h1 align='center'>RoboSat</h1>

<p align=center>
  Generic ecosystem for feature extraction from aerial and satellite imagery

  <img src="assets/buildings.png" alt="RoboSat pipeline extracting buildings from aerial imagery" />
  <i>Berlin aerial imagery, segmentation mask, building outlines, simplified GeoJSON polygons</i>
</p>

<p align="center"><a href="https://travis-ci.org/mapbox/robosat"><img src="https://travis-ci.org/mapbox/robosat.svg?branch=master" /></a></p>


## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
    - [extract](#rs-extract)
    - [cover](#rs-cover)
    - [download](#rs-download)
    - [rasterize](#rs-rasterize)
    - [train](#rs-train)
    - [export](#rs-export)
    - [predict](#rs-predict)
    - [mask](#rs-mask)
    - [features](#rs-features)
    - [merge](#rs-merge)
    - [dedupe](#rs-dedupe)
    - [serve](#rs-serve)
    - [weights](#rs-weights)
    - [compare](#rs-compare)
    - [subset](#rs-subset)
4. [Extending](#extending)
    - [Bring your own imagery](#bring-your-own-imagery)
    - [Bring your own masks](#bring-your-own-masks)
    - [Add support for feature in pre-processing](#add-support-for-feature-in-pre-processing)
    - [Add support for feature in post-processing](#add-support-for-feature-in-post-processing)
5. [Contributing](#contributing)
6. [License](#license)


## Overview

RoboSat is an end-to-end pipeline written in Python 3 for feature extraction from aerial and satellite imagery.
Features can be anything visually distinguishable in the imagery for example: buildings, parking lots, roads, or cars.

Have a look at
- [this OpenStreetMap diary post](https://www.openstreetmap.org/user/daniel-j-h/diary/44145) where we first introduced RoboSat and show some results.
- [this OpenStreetMap diary post](https://www.openstreetmap.org/user/daniel-j-h/diary/44321) where we extract building footprints based on drone imagery in Tanzania.

The tools RoboSat comes with can be categorized as follows:
- data preparation: creating a dataset for training feature extraction models
- training and modeling: segmentation models for feature extraction in images
- post-processing: turning segmentation results into cleaned and simple geometries

Tools work with the [Slippy Map](https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames) tile format to abstract away geo-referenced imagery behind tiles of the same size.

![](./assets/pipeline-01.png)

The data preparation tools help you with getting started creating a dataset for training feature extraction models.
Such a dataset consists of aerial or satellite imagery and corresponding masks for the features you want to extract.
We provide convenient tools to automatically create these datasets downloading aerial imagery from the [Mapbox](mapbox.com) Maps API and generating masks from [OpenStreetMap](openstreetmap.org) geometries but we are not bound to these sources.

![](./assets/pipeline-02.png)

The modelling tools help you with training fully convolutional neural nets for segmentation.
We recommend using (potentially multiple) GPUs for these tools: we are running RoboSat on AWS p2/p3 instances and GTX 1080 TI GPUs.
After you trained a model you can save its checkpoint and run prediction either on GPUs or CPUs.

![](./assets/pipeline-03.png)

The post-processing tools help you with cleaning up the segmentation model's results.
They are responsible for denoising, simplifying geometries, transforming from pixels in Slippy Map tiles to world coordinates (GeoJSON features), and properly handling tile boundaries.

If this sounds almost like what you need, see the [extending section](#extending) for more details about extending RoboSat.
If you want to contribute, see the [contributing section](#contributing) for more details about getting involved with RoboSat.


## Installation

We provide pre-built Docker images packaging up everything you will need for both CPU as well as GPU environments on Docker Hub under the [mapbox/robosat](https://hub.docker.com/r/mapbox/robosat/tags/) namespace.
The pre-built GPU Docker images require the [NVIDIA Container Runtime for Docker](https://github.com/NVIDIA/nvidia-docker).

The following describes the installation from scratch.


- Install native system dependencies required for Python 3 bindings

```bash
apt-get install build-essential libboost-python-dev libexpat1-dev zlib1g-dev libbz2-dev libspatialindex-dev
```

- Use a virtualenv for installing this project locally

```bash
python3 -m venv .env
. .env/bin/activate
```

- Get the PyTorch wheel for your environment from http://pytorch.org. For example for Python 3.5 and CUDA 8

```bash
python3 -m pip install torch
```

- Install remaining dependencies

```bash
python3 -m pip install -r deps/requirements-lock.txt
```

## Development

**NOTE** 
Current working environment:

Based on `conda list`
```bash
# Name                    Version                   Build  Channel
affine                    2.3.0                      py_0    conda-forge
attrs                     21.2.0             pyhd8ed1ab_0    conda-forge
blosc                     1.21.0               h0e60522_0    conda-forge
boost-cpp                 1.74.0               h5b4e17d_5    conda-forge
bzip2                     1.0.8                h8ffe710_4    conda-forge
ca-certificates           2021.10.8            h5b45459_0    conda-forge
cairo                     1.16.0            h15b3021_1009    conda-forge
certifi                   2021.10.8        py38haa244fe_1    conda-forge
cfitsio                   4.0.0                hd67004f_0    conda-forge
charset-normalizer        2.0.9                    pypi_0    pypi
click                     7.1.2                    pypi_0    pypi
click-plugins             1.1.1                      py_0    conda-forge
cligj                     0.7.2              pyhd8ed1ab_1    conda-forge
colorama                  0.4.4              pyh9f0ad1d_0    conda-forge
curl                      7.80.0               h789b8ee_1    conda-forge
cycler                    0.11.0                   pypi_0    pypi
cython                    0.29.25          py38h885f38d_0    conda-forge
dataclasses               0.8                pyhc8e2a94_3    conda-forge
e                         1.4.5                    pypi_0    pypi
expat                     2.4.1                h39d44d4_0    conda-forge
flask                     1.1.4                    pypi_0    pypi
font-ttf-dejavu-sans-mono 2.37                 hab24e00_0    conda-forge
font-ttf-inconsolata      3.000                h77eed37_0    conda-forge
font-ttf-source-code-pro  2.038                h77eed37_0    conda-forge
font-ttf-ubuntu           0.83                 hab24e00_0    conda-forge
fontconfig                2.13.1            h1989441_1005    conda-forge
fonts-conda-ecosystem     1                             0    conda-forge
fonts-conda-forge         1                             0    conda-forge
fonttools                 4.28.3                   pypi_0    pypi
freetype                  2.10.4               h546665d_1    conda-forge
freexl                    1.0.6                ha8e266a_0    conda-forge
gdal                      3.4.0           py38h67fab55_12    conda-forge
geojson                   2.5.0                    pypi_0    pypi
geos                      3.10.1               h39d44d4_1    conda-forge
geotiff                   1.7.0                hc8731e1_5    conda-forge
gettext                   0.19.8.1          ha2e2712_1008    conda-forge
hdf4                      4.2.15               h0e5069d_3    conda-forge
hdf5                      1.12.1          nompi_h57737ce_103    conda-forge
icu                       69.1                 h0e60522_0    conda-forge
idna                      3.3                      pypi_0    pypi
intel-openmp              2021.4.0          h57928b3_3556    conda-forge
itsdangerous              1.1.0                    pypi_0    pypi
jbig                      2.1               h8d14728_2003    conda-forge
jinja2                    2.11.3                   pypi_0    pypi
joblib                    1.1.0              pyhd8ed1ab_0    conda-forge
jpeg                      9d                   h8ffe710_0    conda-forge
kealib                    1.4.14               h8995ca9_3    conda-forge
kiwisolver                1.3.2                    pypi_0    pypi
krb5                      1.19.2               h6da9e4a_3    conda-forge
landsurvey                0.0.1                     dev_0    <develop>
lcms2                     2.12                 h2a16943_0    conda-forge
lerc                      3.0                  h0e60522_0    conda-forge
libblas                   3.9.0              12_win64_mkl    conda-forge
libcblas                  3.9.0              12_win64_mkl    conda-forge
libcurl                   7.80.0               h789b8ee_1    conda-forge
libdeflate                1.8                  h8ffe710_0    conda-forge
libffi                    3.4.2                h8ffe710_5    conda-forge
libgdal                   3.4.0               h58f6a35_12    conda-forge
libglib                   2.70.2               h3be07f2_0    conda-forge
libiconv                  1.16                 he774522_0    conda-forge
libkml                    1.3.0             h9859afa_1014    conda-forge
liblapack                 3.9.0              12_win64_mkl    conda-forge
libnetcdf                 4.8.1           nompi_h1cc8e9d_101    conda-forge
libpng                    1.6.37               h1d00b33_2    conda-forge
libpq                     14.1                 h1ea2d34_1    conda-forge
librttopo                 1.1.0                he35e8ac_8    conda-forge
libspatialite             5.0.1               hf126459_12    conda-forge
libssh2                   1.10.0               h9a1e1f7_2    conda-forge
libtiff                   4.3.0                hd413186_2    conda-forge
libwebp-base              1.2.1                h8ffe710_0    conda-forge
libxml2                   2.9.12               hf5bbc77_1    conda-forge
libzip                    1.8.0                h519de47_1    conda-forge
libzlib                   1.2.11            h8ffe710_1013    conda-forge
lz4-c                     1.9.3                h8ffe710_1    conda-forge
m2w64-gcc-libgfortran     5.3.0                         6    conda-forge
m2w64-gcc-libs            5.3.0                         7    conda-forge
m2w64-gcc-libs-core       5.3.0                         7    conda-forge
m2w64-gmp                 6.1.0                         2    conda-forge
m2w64-libwinpthread-git   5.0.0.4634.697f757               2    conda-forge
markupsafe                2.0.1            py38h294d835_1    conda-forge
matplotlib                3.5.1                    pypi_0    pypi
mercantile                1.2.1                    pypi_0    pypi
mkl                       2021.4.0           h0e2418a_729    conda-forge
msys2-conda-epoch         20160418                      1    conda-forge
numpy                     1.21.4           py38h089cfbf_0    conda-forge
opencv-contrib-python-headless 4.5.4.60                 pypi_0    pypi
openjpeg                  2.4.0                hb211442_1    conda-forge
openssl                   3.0.0                h8ffe710_2    conda-forge
osmium                    2.15.2                   pypi_0    pypi
packaging                 21.3                     pypi_0    pypi
pcre                      8.45                 h0e60522_0    conda-forge
pillow                    6.2.2                    pypi_0    pypi
pip                       21.3.1             pyhd8ed1ab_0    conda-forge
pixman                    0.40.0               h8ffe710_0    conda-forge
poppler                   21.11.0              h24fffdf_0    conda-forge
poppler-data              0.4.11               hd8ed1ab_0    conda-forge
postgresql                14.1                 he353ca9_1    conda-forge
proj                      8.2.0                h1cfcee9_0    conda-forge
pyparsing                 3.0.6              pyhd8ed1ab_0    conda-forge
pyproj                    2.6.1.post1              pypi_0    pypi
python                    3.8.12          h900ac77_2_cpython    conda-forge
python-dateutil           2.8.2                    pypi_0    pypi
python_abi                3.8                      2_cp38    conda-forge
rasterio                  1.2.10           py38h48edd3a_3    conda-forge
requests                  2.26.0                   pypi_0    pypi
rtree                     0.9.7                    pypi_0    pypi
scikit-learn              1.0.1            py38hb60ee80_2    conda-forge
scipy                     1.8.0rc1                 pypi_0    pypi
setuptools                58.0.4           py38haa95532_0
shapely                   1.8.0                    pypi_0    pypi
six                       1.16.0                   pypi_0    pypi
snuggs                    1.4.7                      py_0    conda-forge
sqlite                    3.37.0               h8ffe710_0    conda-forge
supermercado              0.0.5                    pypi_0    pypi
tbb                       2021.4.0             h2d74725_1    conda-forge
threadpoolctl             3.0.0              pyh8a188c0_0    conda-forge
tiledb                    2.5.2                h47404fa_0    conda-forge
tk                        8.6.11               h8ffe710_1    conda-forge
toml                      0.10.2                   pypi_0    pypi
torch                     1.10.0                   pypi_0    pypi
torchvision               0.11.1                   pypi_0    pypi
tqdm                      4.62.3                   pypi_0    pypi
typer                     0.4.0                    pypi_0    pypi
typing-extensions         4.0.1                    pypi_0    pypi
ucrt                      10.0.20348.0         h57928b3_0    conda-forge
urllib3                   1.26.7                   pypi_0    pypi
vc                        14.2                 hb210afc_5    conda-forge
vs2015_runtime            14.29.30037          h902a5da_5    conda-forge
werkzeug                  1.0.1                    pypi_0    pypi
wheel                     0.37.0             pyhd8ed1ab_1    conda-forge
wincertstore              0.2                   py38_1003    conda-forge
xerces-c                  3.2.3                h0e60522_4    conda-forge
xz                        5.2.5                h62dcd97_1    conda-forge
zlib                      1.2.11            h8ffe710_1013    conda-forge
zstd                      1.5.0                h6255e5f_0    conda-forge
```

Based on `pip freeze`
```bash
affine==2.3.0
attrs @ file:///home/conda/feedstock_root/build_artifacts/attrs_1620387926260/work
certifi==2021.10.8
charset-normalizer==2.0.9
click==7.1.2
click-plugins==1.1.1
cligj @ file:///home/conda/feedstock_root/build_artifacts/cligj_1633637764473/work
colorama @ file:///home/conda/feedstock_root/build_artifacts/colorama_1602866480661/work
cycler==0.11.0
Cython @ file:///D:/bld/cython_1638830072925/work
dataclasses @ file:///home/conda/feedstock_root/build_artifacts/dataclasses_1628958434797/work
e==1.4.5
Flask==1.1.4
fonttools==4.28.3
GDAL==3.4.0
geojson==2.5.0
idna==3.3
itsdangerous==1.1.0
Jinja2==2.11.3
joblib @ file:///home/conda/feedstock_root/build_artifacts/joblib_1633637554808/work
kiwisolver==1.3.2
-e git+https://github.com/wslerry/robosat.git@7624d4bb8f0d553a2675ff6bc077a23d8d55ffe2#egg=landsurvey
MarkupSafe @ file:///D:/bld/markupsafe_1635833725355/work
matplotlib==3.5.1
mercantile==1.2.1
numpy @ file:///D:/bld/numpy_1636145500119/work
opencv-contrib-python-headless==4.5.4.60
osmium==2.15.2
packaging==21.3
Pillow==6.2.2
pyparsing @ file:///home/conda/feedstock_root/build_artifacts/pyparsing_1636757021002/work
pyproj==2.6.1.post1
python-dateutil==2.8.2
rasterio==1.2.10
requests==2.26.0
robosat==1.2.0
Rtree==0.9.7
scikit-learn @ file:///D:/bld/scikit-learn_1636784140859/work
scipy @ file:///C:/bld/scipy_1637806857964/work
Shapely==1.8.0
six==1.16.0
snuggs==1.4.7
supermercado==0.0.5
threadpoolctl @ file:///home/conda/feedstock_root/build_artifacts/threadpoolctl_1633102299089/work
toml==0.10.2
torch==1.10.0
torchvision==0.11.1
tqdm==4.62.3
typer==0.4.0
typing_extensions==4.0.1
urllib3==1.26.7
Werkzeug==1.0.1
wincertstore==0.2
```


## Usage

The following describes the tools making up the RoboSat pipeline.
All tools can be invoked via

    ./rs <tool> <args>

Also see the sub-command help available via

    ./rs --help
    ./rs <tool> --help

Most tools take a dataset or model configuration file. See examples in the [`configs`](./config) directory.
You will need to adapt these configuration files to your own dataset, for example setting your tile resolution (e.g. 256x256 pixel).
You will also need to adapt these configuration files to your specific deployment setup, for example using CUDA and setting batch sizes.


### rs extract

Extracts GeoJSON features from OpenStreetMap to build a training set from.

The result of `rs extract` is a GeoJSON file with the extracted feature geometries.

The `rs extract` tool walks OpenStreetMap `.osm.pbf` base map files (e.g. from [Geofabrik](http://download.geofabrik.de)) and gathers feature geometries.
These features are for example polygons for parking lots, buildings, or roads.


### rs cover

Generates a list of tiles covering GeoJSON features to build a training set from.

The result of `rs cover` is a file with tiles in `(x, y, z)` [Slippy Map](https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames) tile format covering GeoJSON features.

The `rs cover` tool reads in the GeoJSON features generated by `rs extract` and generates a list of tiles covering the feature geometries.


### rs download

Downloads aerial or satellite imagery from a Slippy Map endpoint (e.g. the Mapbox Maps API) based on a list of tiles.

The result of `rs download` is a Slippy Map directory with aerial or satellite images - the training set's images you will need for the model to learn on.

The `rs download` tool downloads images for a list of tiles in `(x, y, z)` [Slippy Map](https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames) tile format generated by `rs cover`.

The `rs download` tool expects a Slippy Map endpoint where placeholders for `{x}`, `{y}`, and `{z}` are formatted with each tile's ids.
For example, for the Mapbox Maps API: `https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}@2x.webp?access_token=TOKEN`.


### rs rasterize

Rasterizes GeoJSON features into mask images based on a list of tiles.

The result of `rs rasterize` is a Slippy Map directory with masks - the training set's masks you will need for the model to learn on.

The `rs rasterize` tool reads in GeoJSON features and rasterizes them into single-channel masks with a color palette attached for quick visual inspection.


### rs train

Trains a model on a training set made up of `(image, mask)` pairs.

The result of `rs train` is a checkpoint containing weights for the trained model.

The `rs train` tool trains a fully convolutional neural net for semantic segmentation on a dataset with `(image, mask)` pairs generated by `rs download` and `rs rasterize`.
We recommend using a GPU for training: we are working with the AWS p2 instances and GTX 1080 TI GPUs.

Before you can start training you need the following.

- You need a dataset which you should split into three parts: training and validation for `rs train` to train on and to calculate validation metrics on and a hold-out dataset for final model evaluation. The dataset's directory need to look like the following.

      dataset
      ├── training
      │   ├── images
      │   └── labels
      └── validation
          ├── images
          └── labels

- You need to calculate label class weights with `rs weights` on the training set's labels

- You need to add the path to the dataset's directory and the calculated class weights and statistics to the dataset config.


### rs export

Exports a trained model in [ONNX](https://onnx.ai/) format for prediction across different backends (like Caffe2, TensorFlow).

The result of `rs export` is an ONNX GraphProto `.pb` file which can be used with the ONNX ecosystem.

Note: the `rs predict` tool works with `.pth` checkpoints. In contrast to these `.pth` checkpoints the ONNX models neither depent on PyTorch or the Python code for the model class and can be used e.g. in resource constrained environments like AWS Lambda.


### rs predict

Predicts class probabilities for each image tile in a Slippy Map directory structure.

The result of `rs predict` is a Slippy Map directory with a class probability encoded in a `.png` file per tile.

The `rs predict` tool loads the checkpoint weights generated by `rs train` and predicts semantic segmentation class probabilities for a Slippy Map dataset consisting of image tiles.


### rs mask

Generates segmentation masks for each class probability `.png` file in a Slippy Map directory structure.

The result of `rs mask` is a Slippy Map directory with one single-channel image per tile with a color palette attached for quick visual inspection.

The `rs mask` tool loads in the `.png` tile segmentation class probabilities generated by `rs predict` and turns them into segmentation masks.
You can merge multiple Slippy Map directories with class probabilities into a single mask using this tool in case you want to make use of an ensemble of models.


### rs features

Extracts simplified GeoJSON features for segmentation masks in a Slippy Map directory structure.

The result of `rs features` is a GeoJSON file with the extracted simplified features.

The `rs features` tool loads the segmentation masks generated by `rs mask` and turns them into simplified GeoJSON features.


### rs merge

Merges close adjacent GeoJSON features into single features.

The result of `rs merge` is a GeoJSON file with the merged features.

The `rs merge` tool loads GeoJSON features and depending on a threshold merges adjacent geometries together.


### rs dedupe

Deduplicates predicted features against existing OpenStreetMap features.

The result of `rs dedupe` is a GeoJSON file with predicted features which are not in OpenStreetMap.

The `rs dedupe` deduplicates predicted features against OpenStreetMap.

Note: `rs extract` to generate a GeoJSON file with OpenStreetMap features.


### rs serve

Serves tile masks by providing an on-demand segmentation tileserver.

The `rs serve` tool implements a Slippy Map raster tileserver requesting satellite tiles and applying the segmentation model on the fly.

Notes: useful for visually inspecting the raw segmentation masks on the fly; for serious use-cases use `rs predict` and similar.


### rs weights

Calculates class weights for a Slippy Map directory with masks.

The result of `rs weights` is a list of class weights useful for `rs train` to adjust the loss based on the class distribution in the masks.

The `rs weights` tool computes the pixel-wise class distribution on the training dataset's masks and outputs weights for training.


### rs compare

Prepares images, labels and predicted masks, side-by-side for visual comparison.

The result of `rs compare` is a Slippy Map directory with images that have the raw image on the left, the label in the middle and the prediction on the right.


### rs subset

Filters a Slippy Map directory based on a list of tile ids.

The result of `rs subset` is a Slippy Map directory filtered by tile ids.

The main use-case for this tool is hard-negative mining where we want to filter false positives from a prediction run.


## Extending

There are multiple ways to extend RoboSat for your specific use-cases.
By default we use [Mapbox](mapbox.com) aerial imagery from the Maps API and feature masks generated from [OpenStreetMap](openstreetmap.org) geometries.
If you want to bring your own imagery, masks, or features to extract, the following will get you started.

### Bring your own imagery

RoboSat's main abstraction is the [Slippy Map](https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames) tile format.
As long as your imagery is geo-referenced and you can convert it to a Slippy Map directory structure to point the command lines to, you are good to go.
Make sure imagery and masks are properly aligned.

### Bring your own masks

RoboSat's main abstraction is the [Slippy Map](https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames) tile format.
As long as you can convert your masks to a Slippy Map directory structure to point the command lines to, you are good to go.
Masks have to be single-channel `.png` files with class indices starting from zero.
Make sure imagery and masks are properly aligned.

### Add support for feature in pre-processing

Pre-processing (`rs extract`) is responsible for turning OpenStreetMap geometries and tags into polygon feature masks.
If you want to add a new feature based on geometries in OpenStreetMap you have to:
- Implement an [osmium](https://docs.osmcode.org/pyosmium/latest/) handler which turns OpenStreetMap geometries into polygons; see [`robosat/osm/`](./robosat/osm/) for existing handlers.
- Import and register your handler in [`robosat/tools/extract.py`](./robosat/tools/extract.py).

And that's it! From there on the pipeline is fully generic.

### Add support for feature in post-processing

Post-processing (`rs features`) is responsible for turning segmentation masks into simplified GeoJSON features.
If you want to add custom post-processing for segmentation masks you have to:
- Implement a featurize handler turning masks into GeoJSON features; see [`robosat/features/`](./robosat/features/) for existing handlers.
- Import and register your handler in [`robosat/tools/features.py`](./robosat/tools/features.py).

And that's it! From there on the pipeline is fully generic.


## Contributing

We are thankful for contributions and are happy to help; that said there are some constraints to take into account:
- For non-trivial changes you should open a ticket first to outline and discuss ideas and implementation sketches. If you just send us a pull request with thousands of lines of changes we most likely won't accept your changeset.
- We follow the 80/20 rule where 80% of the effects come from 20% of the causes: we strive for simplicity and maintainability over pixel-perfect results. If you can improve the model's accuracy by two percent points but have to add thousands of lines of code we most likely won't accept your changeset.
- We take responsibility for changesets going into master: as soon as your changeset gets approved it is on us to maintain and debug it. If your changeset can not be tested, or maintained in the future by the core developers we most likely won't accept your changeset.


## License

Copyright (c) 2018 Mapbox

Distributed under the MIT License (MIT).
