#%% Imports -------------------------------------------------------------------

import cv2
import numpy as np
from skimage import io
from pathlib import Path

#%% Inputs --------------------------------------------------------------------

local_path = Path("D:\local_Gkountidi\data")
train_path = Path(Path.cwd(), 'data', 'train')

# Patch selection
np.random.seed(42)
nSelect = 10 # number of randomly selected frames per movie
pSize = 512  # size of selected patches
 
#%% Process -------------------------------------------------------------------

# Data structure
data = {
        "path": [], "name": [],
        "nT"  : [], "nY"  : [], "nX"  : [],
        }

# Read metadata
for path in local_path.iterdir():
    if path.suffix == ".avi":
        
        # Parse avi file
        cap = cv2.VideoCapture(str(path))
        nT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Read first frame
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        nY, nX = frame.shape
        
        # Append metadata
        data["path"].append(path)
        data["name"].append(path.stem)
        data["nT"  ].append(nT)
        data["nY"  ].append(nY)
        data["nX"  ].append(nX)
        
# Save patches   
for i in range(len(data["path"])):
    
    # Set random selection
    nT, nY, nX = data["nT"][i], data["nY"][i], data["nX"][i]  
    rT = np.random.randint(0, nT, size=nSelect)
    rY = np.random.randint(0, nY - pSize, size=nSelect)
    rX = np.random.randint(0, nX - pSize, size=nSelect)

    for t, y, x in zip(rT, rY, rX):
        
        # Extract patches
        cap = cv2.VideoCapture(str(data["path"][i]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        patch = frame[y:y + pSize, x:x + pSize]
        name = data["name"][i] + f"_({t:04d}-{y:04d}-{x:04d}).tif"
        
        # Save patches
        io.imsave(
            Path(train_path, name),
            patch, check_contrast=False
            )      