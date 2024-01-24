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

#%% Inputs --------------------------------------------------------------------

# Paths
local_path = Path("D:\local_Gkountidi\data")
predict_path = Path(Path.cwd(), "data", "predict")
model_path = Path(Path.cwd(), "model_weights.h5")
movie_name = "20231017-test 1+ 10nM erlotinib.avi"

#%% Pre-processing ------------------------------------------------------------

# Convert avi to numpy array
images = []
cap = cv2.VideoCapture(str(Path(local_path, movie_name)))
ret, frame = cap.read()
while ret:
    
    # Normalize image
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pMax = np.percentile(image, 99.9)
    image[image > pMax] = pMax
    image = (image / pMax).astype(float)
    images.append(image) 
    
    # Read next frame
    ret, frame = cap.read()
    
images = np.stack(images)
    
cap.release()

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
probs = model.predict(images).squeeze()

#%% Outputs -------------------------------------------------------------------

# # Display 
# viewer = napari.Viewer()
# viewer.add_image(images)
# viewer.add_image(masks)   
# viewer.add_image(probs)

# # Save
# io.imsave(
#     Path(test_path, "prediction.tif"),
#     probs[0].astype("float32"),
#     check_contrast=False,
#     )