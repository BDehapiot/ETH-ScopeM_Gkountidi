#%% Imports -------------------------------------------------------------------

import cv2
import time
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed
from skimage.transform import downscale_local_mean

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

# -----------------------------------------------------------------------------

def preprocessing(arr, rescale_factor, mask=False):
       
    def _preprocessing(img):
        
        if mask:     
            img = img.astype(bool).astype(float)
            img = (downscale_local_mean(img, rescale_factor) > 0.5).astype(float)
            
        else:
            img = downscale_local_mean(img, rescale_factor)
            pMax = np.percentile(img, 99.9)
            img[img > pMax] = pMax
            img = (img / pMax).astype(float)
            
        return img
    
    if arr.ndim == 2:
        arr = _preprocessing(arr)
        
    if arr.ndim == 3:
        outputs = Parallel(n_jobs=-1)(
            delayed(_preprocessing)(img)
            for img in arr
            )
        arr = np.stack(outputs)
    
    return arr

# -----------------------------------------------------------------------------

def get_patches(arr, size, overlap):
    
    # Get dimensions
    if arr.ndim == 2: nT = 1; nY, nX = arr.shape 
    if arr.ndim == 3: nT, nY, nX = arr.shape
    
    # Get variables
    y0s = np.arange(0, nY, size - overlap)
    x0s = np.arange(0, nX, size - overlap)
    yMax = y0s[-1] + size
    xMax = x0s[-1] + size
    yPad = yMax - nY
    xPad = xMax - nX
    yPad1, yPad2 = yPad // 2, (yPad + 1) // 2
    xPad1, xPad2 = xPad // 2, (xPad + 1) // 2
    
    # Pad array
    if arr.ndim == 2:
        arr_pad = np.pad(
            arr, ((yPad1, yPad2), (xPad1, xPad2)), mode='reflect') 
    if arr.ndim == 3:
        arr_pad = np.pad(
            arr, ((0, 0), (yPad1, yPad2), (xPad1, xPad2)), mode='reflect')         
    
    # Extract patches
    patches = []
    if arr.ndim == 2:
        for y0 in y0s:
            for x0 in x0s:
                patches.append(arr_pad[y0:y0 + size, x0:x0 + size])
    if arr.ndim == 3:
        for t in range(nT):
            for y0 in y0s:
                for x0 in x0s:
                    patches.append(arr_pad[t, y0:y0 + size, x0:x0 + size])
            
    return patches

# -----------------------------------------------------------------------------

def merge_patches(patches, shape, size, overlap):
    
    # Get dimensions 
    if len(shape) == 2: nT = 1; nY, nX = shape
    if len(shape) == 3: nT, nY, nX = shape
    nPatch = len(patches) // nT

    # Get variables
    y0s = np.arange(0, nY, size - overlap)
    x0s = np.arange(0, nX, size - overlap)
    yMax = y0s[-1] + size
    xMax = x0s[-1] + size
    yPad = yMax - nY
    xPad = xMax - nX
    yPad1 = yPad // 2
    xPad1 = xPad // 2

    # Merge patches
    def _merge_patches(patches):
        count = 0
        arr = np.full((2, nY + yPad, nX + xPad), np.nan)
        for i, y0 in enumerate(y0s):
            for j, x0 in enumerate(x0s):
                if i % 2 == j % 2:
                    arr[0, y0:y0 + size, x0:x0 + size] = patches[count]
                else:
                    arr[1, y0:y0 + size, x0:x0 + size] = patches[count]
                count += 1 
        arr = np.nanmean(arr, axis=0)
        arr = arr[yPad1:yPad1 + nY, xPad1:xPad1 + nX]
        return arr
        
    if len(shape) == 2:
        arr = _merge_patches(patches)

    if len(shape) == 3:
        patches = np.stack(patches).reshape(nT, nPatch, size, size)
        arr = Parallel(n_jobs=-1)(
            delayed(_merge_patches)(patches[t,...])
            for t in range(nT)
            )
        arr = np.stack(arr)
        
    return arr

#%% Tests --------------------------------------------------------------------- 

# # Paths
# local_path = Path("D:\local_Gkountidi\data")
# avi_name = "20231017-test 1+ 10nM erlotinib.avi"

# # Parameters
# size = 512
# overlap = size // 8

# # -----------------------------------------------------------------------------

# # Open data (from tif)
# arr = io.imread(Path("D:\\local_Gkountidi\\data\\rscale.tif"))
# # arr = arr[0,...] # Test only first frame

# # # Open data (from avi)
# # t0 = time.time()
# # arr = avi2ndarray(Path(local_path, avi_name), frame="all")
# # t1 = time.time()
# # print(f"avi2ndarray : {(t1-t0):<5.2f}s") 

# # Preprocessing
# t0 = time.time()
# arr = preprocessing(arr, rescale_factor, mask=False)
# t1 = time.time()
# print(f"preprocessing : {(t1-t0):<5.2f}s")     

# # Get patches
# t0 = time.time()
# patches = get_patches(arr, size, overlap)
# t1 = time.time()
# print(f"get_patches : {(t1-t0):<5.2f}s") 

# # Merge patches
# t0 = time.time()
# shape = arr.shape
# arr_new = merge_patches(patches, shape, size, overlap)
# t1 = time.time()
# print(f"Merge patches : {(t1-t0):<5.2f}s") 

# # -----------------------------------------------------------------------------

# import napari
# viewer = napari.Viewer()
# viewer.add_image(arr)
# viewer.add_image(arr_new)   
