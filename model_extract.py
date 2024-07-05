#%% Imports -------------------------------------------------------------------

import cv2
import numpy as np
from skimage import io
from pathlib import Path
from functions import avi2ndarray

#%% Inputs --------------------------------------------------------------------

# Paths
local_path = Path("D:\local_Gkountidi\data")
train_path = Path(Path.cwd(), 'data', 'train')

# Frame selection
np.random.seed(42)
nFrame = 10 # number of randomly selected frames per movie

#%%

metadata = {"path": [], "name": [], "nT"  : []}
for path in local_path.iterdir():
    if path.suffix == ".avi":
        cap = cv2.VideoCapture(str(path))
        metadata["path"].append(path)
        metadata["name"].append(path.stem)
        metadata["nT"  ].append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

for i in range(len(metadata["path"])):
    
    # Set random selection
    nT = metadata["nT"][i]
    rT = np.random.randint(0, nT, size=nFrame)

    for t in rT:        
        
        # Extract frames
        arr = avi2ndarray(str(metadata["path"][i]), frame=int(t))
        name = metadata["name"][i] + f"_{t:04d}.tif"
        io.imsave(
            Path(train_path, name),
            arr, check_contrast=False
            )     