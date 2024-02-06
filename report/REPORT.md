# ETH-ScopeM_Gkountidi

## Contact
Benoit Dehapiot  
benoit.dehapiot@scopem.ethz.ch  

## Overview

## Procedure

### 1. Lymphatic vessel segmentation

Comment

#### 1.1 Format training data 
`extract.py`  

- Open and convert `.avi` movies to `ndarray`
- Ramdomly select n frames in each movies
- Save selected frames as `.tif` files in the `data/train` folder
    - [movie_name]_[frame_number].tif

#### 1.2 Annotate training data
`annotate.py`  

- Sequentially open saved frames in `Napari` viewer
- Annotate vessels using the label brush tool
- Save annotated masks as `.tif` files in the `data/train` folder
    - [movie_name]_[frame_number]_mask.tif

#### 1.3 Train U-Net model
`train.py`  

- Open selected frames and associated masks
- Downscale ...
- Normalize frame images (0 to 1)  
- Setup U-Net architecture and parameters (epoch)

### 2. Measure local vessel contraction

#### 2.1 Normalize and predict
- Open and convert `.avi` movies to `ndarray`
- Normalize images (0 to 1) 

