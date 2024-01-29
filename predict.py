#%% Imports -------------------------------------------------------------------

import cv2
import time
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
from functions import avi2ndarray, preprocessing, get_patches, merge_patches

#%% Inputs --------------------------------------------------------------------

# Paths
local_path = Path("D:\local_Gkountidi\data")
predict_path = Path(Path.cwd(), "data", "predict")
model_path = Path(Path.cwd(), "model_weights.h5")

# avi_name = "20231017-test 1+ 10nM erlotinib.avi"
avi_name = "20231017-test 1+ PBS.avi"
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

# Open data
path = Path(local_path, avi_name)
print("Open data       :", end='')
t0 = time.time()
arr = avi2ndarray(path, frame=frame)
t1 = time.time()
print(f" {(t1-t0):<5.2f}s") 

# Preprocessing
print("Preprocessing   :", end='')
t0 = time.time()
arr = preprocessing(arr, rescale_factor, mask=False)
t1 = time.time()
print(f" {(t1-t0):<5.2f}s") 

# Extract patches
print("Extract patches :", end='')
t0 = time.time()
patches = np.stack(get_patches(arr, size, overlap))
t1 = time.time()
print(f" {(t1-t0):<5.2f}s") 

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

# Predict
predict = model.predict(patches).squeeze()

# Merge patches
print("Merge patches :", end='')
t0 = time.time()
predict = merge_patches(predict, arr.shape, size, overlap)
t1 = time.time()
print(f" {(t1-t0):<5.2f}s") 

# Display
viewer = napari.Viewer()
viewer.add_image(np.stack(arr)) 
viewer.add_image(np.stack(predict)) 

# Save
io.imsave(
    Path(local_path, avi_name.replace(".avi", "_predict.tif")),
    predict.astype("float32"), check_contrast=False
    )

