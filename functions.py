#%% Imports -------------------------------------------------------------------

import cv2
import time
import numpy as np

#%% Functions -----------------------------------------------------------------

def avi2ndarray(path, frame="all"):
    
    cap = cv2.VideoCapture(str(path))
    nT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    nY = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nX = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
    if frame == "all":
        
        arr = np.empty((nT, nY, nX), int)
        
        for t in range(nT):
            ret, img = cap.read()
            if ret:
                arr[t] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                break
            
    elif isinstance(frame, int):

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = cap.read()
        arr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    cap.release()
    
    return arr

def get_patches(img, size=256, overlap=32):
    
    # Get variables
    nY, nX = img.shape
    y0s = np.arange(0, nY, size - overlap)
    x0s = np.arange(0, nX, size - overlap)
    yMax = y0s[-1] + size
    xMax = x0s[-1] + size
    yPad = yMax - nY
    xPad = xMax - nX
    yPad1, yPad2 = yPad // 2, (yPad + 1) // 2
    xPad1, xPad2 = xPad // 2, (xPad + 1) // 2
    
    # Pad image
    img_pad = np.pad(img, ((yPad1, yPad2), (xPad1, xPad2)), mode='reflect') 
    
    # Extract patches
    patches = []
    for y0 in y0s:
        for x0 in x0s:
            patches.append(img_pad[y0:y0 + size, x0:x0 + size])
            
    # Output dict
    patches = {
        "nY"      : nY,
        "nX"      : nX,
        "size"    : size,
        "overlap" : overlap,
        "patches" : patches,
        }
            
    return patches

def merge_patches(patches):
    
    count = 0
    
    # Get variables
    nY = patches["nY"]
    nX = patches["nX"]
    size = patches["size"]
    overlap = patches["overlap"]
    patches = patches["patches"]
    y0s = np.arange(0, nY, size - overlap)
    x0s = np.arange(0, nX, size - overlap)
    yMax = y0s[-1] + size
    xMax = x0s[-1] + size
    yPad = yMax - nY
    xPad = xMax - nX
    yPad1, yPad2 = yPad // 2, (yPad + 1) // 2
    xPad1, xPad2 = xPad // 2, (xPad + 1) // 2

    # Merge patches
    img = np.full((2, nY + yPad, nX + xPad), np.nan)
    for i, y0 in enumerate(y0s):
        for j, x0 in enumerate(x0s):
            if i % 2 == j % 2:
                img[0, y0:y0 + size, x0:x0 + size] = patches[count]
            else:
                img[1, y0:y0 + size, x0:x0 + size] = patches[count]
            count += 1
    img = np.nanmean(img, axis=0).astype(int)
    
    # Remove padding
    img = img[yPad1:yPad1 + nY, xPad1:xPad1 + nX]
    
    return img

#%% Tests ---------------------------------------------------------------------

# from pathlib import Path

# local_path = Path("D:\local_Gkountidi\data")
# avi_name = "20231017-test 1+ 10nM erlotinib.avi"

# print("avi2ndarray :", end='')
# t0 = time.time()

# frames = avi2ndarray(Path(local_path, avi_name))

# t1 = time.time()
# print(f" {(t1-t0):<5.2f}s") 

# -----------------------------------------------------------------------------

# img = frames[0,...]

# print("get_patches :", end='')
# t0 = time.time()

# for _ in range(1):

#     patches = get_patches(img, size=256, overlap=32)
#     img_new = merge_patches(patches)

# t1 = time.time()
# print(f" {(t1-t0):<5.5f}s") 

# -----------------------------------------------------------------------------

# import napari
# viewer = napari.Viewer()
# viewer.add_image(img_pad)
# viewer.add_image(img_new)
# viewer.add_image(np.stack(patches))

