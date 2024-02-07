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
- Save selected frames as `.tif` files in the `data/train` folder as   
`[movie_name]_[frame_number].tif`

#### 1.2 Annotate training data

`annotate.py`  

- Sequentially open saved frames in `Napari` viewer
- Annotate vessels using the label brush tool
- Save annotated masks as `.tif` files in the `data/train` folder as
`[movie_name]_[frame_number]_mask.tif`

<img src="fig1.png" alt="fig1" width="512" height="auto">

#### 1.3 Prepare data and train U-Net model

`train.py`  

- Open selected frames and associated masks
- Reduce image resolution 
- Normalize images (0 to 1)  
- Data augmentation (flip, rotate, distortion...)
- Setup U-Net architecture and parameters (epoch, batch size, loss...)
- Save model weights as `model_weights.h5`

### 2. Measure local vessel contraction

`analyse.py`

#### 2.1 Prepare data and predict
- Open and convert `.avi` movies to `ndarray`
- Reduce image resolution 
- Normalize images (0 to 1)  
- Get predictions (all frames)

<img src="fig2.png" alt="fig2" width="512" height="auto">

#### 2.2 ???
- Spatial registration

<img src="fig.png" alt="fig3" width="512" height="auto">

- Get prediction mask and euclidean distance map 

<img src="fig.png" alt="fig4" width="512" height="auto">