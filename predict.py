#%% Imports -------------------------------------------------------------------

import cv2
import napari
import numpy as np
from skimage import io
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import segmentation_models as sm
from joblib import Parallel, delayed
from skimage.transform import downscale_local_mean 

# Functions
from functions import avi2ndarray, get_patches, merge_patches

#%% Inputs --------------------------------------------------------------------

# Paths
local_path = Path("D:\local_Gkountidi\data")
predict_path = Path(Path.cwd(), "data", "predict")
model_path = Path(Path.cwd(), "model_weights.h5")

avi_name = "20231017-test 1+ 10nM erlotinib.avi"
# avi_name = "20231017-test 1+ PBS.avi"
# avi_name = "20231017-test 2+ 10nM erlotinib.avi"
# avi_name = "20231017-test 2+ PBS.avi"
# avi_name = "20231017-test 3+ 10nM erlotinib.avi"

# Frames
frame = "all"

# Patches
rescale_factor = 2
size = 512 // rescale_factor
overlap = size // 8

#%% Pre-processing ------------------------------------------------------------

# Open & normalize image
path = Path(local_path, avi_name)
arr = avi2ndarray(path, frame=frame)
if rescale_factor != 1:
    arr = downscale_local_mean(arr, (1, rescale_factor, rescale_factor))
pMax = np.percentile(arr, 99.9)
arr[arr > pMax] = pMax
arr = (arr / pMax).astype(float)

# 
arr = arr[0:50,...]

#%% Predict -------------------------------------------------------------------

# Define & compile model
model = sm.Unet(
    'resnet34', 
    input_shape=(None, None, 1), 
    classes=1, 
    activation='sigmoid', 
    encoder_weights=None,
    )
model.compile(
    optimizer='adam',
    loss='binary_crossentropy', 
    metrics=['mse']
    )

# Load weights
model.load_weights(model_path) 

# 
predict = []
for img in arr:
    img_patches = get_patches(img, size=size, overlap=overlap)
    imgs = np.stack([patches for patches in img_patches["patches"]])
    probs = model.predict(imgs).squeeze()
    prob_patches = img_patches.copy()
    for i, prob in enumerate(probs):
        prob_patches["patches"][i] = prob
    predict.append(merge_patches(prob_patches))
    
# # Predict
# probs = model.predict(imgs).squeeze()

# # 
# prob_patches = patches.copy()
# for i, prob in enumerate(probs):
#     prob_patches["patches"][i] = prob
# merged_probs = merge_patches(prob_patches)

#%% Display -------------------------------------------------------------------

viewer = napari.Viewer()
viewer.add_image(np.stack(arr)) 
viewer.add_image(np.stack(predict)) 

# viewer = napari.Viewer()
# viewer.add_image(img) 
# viewer.add_image(merged_probs)