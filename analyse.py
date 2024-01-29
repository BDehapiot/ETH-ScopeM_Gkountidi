#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

#%% Inputs --------------------------------------------------------------------

local_path = Path("D:\local_Gkountidi\data")
# avi_name = "20231017-test 1+ 10nM erlotinib.avi"
# avi_name = "20231017-test 1+ PBS.avi"
# avi_name = "20231017-test 2+ 10nM erlotinib.avi"
# avi_name = "20231017-test 2+ PBS.avi"
# avi_name = "20231017-test 3+ 10nM erlotinib.avi"
# avi_name = "20231017-test 3+ PBS.avi"
avi_name = "20231017-test 4+ 1nM erlotinib.avi"
# avi_name = "20231017-test 4+ PBS.avi"
# avi_name = "20231017-test 5+ 1nM erlotinib.avi"
# avi_name = "20231017-test 5+ PBS.avi"
# avi_name = "20231017-test 6+ PBS.avi"

#%% 

# Skimage
from skimage.filters import gaussian
from skimage.morphology import remove_small_holes, skeletonize

# Scipy
from scipy.ndimage import distance_transform_edt

# -----------------------------------------------------------------------------

# Open data
rescale = io.imread(Path(local_path, avi_name.replace(".avi", "_rescale.tif")))
predict = io.imread(Path(local_path, avi_name.replace(".avi", "_predict.tif")))

#%%

from pystackreg import StackReg

def register(arr1, arr2):

    def _imreg(ref1, mov1, mov2):
        sr = StackReg(StackReg.RIGID_BODY)
        sr.register(ref1, mov1)
        mov1 = sr.transform(mov1)
        mov2 = sr.transform(mov2)
        return mov1, mov2
    
    outputs = Parallel(n_jobs=-1)(
        delayed(_imreg)(arr1[0,...], arr1[t,...], arr2[t,...])
        for t in range(1, arr1.shape[0])
        )
    arr1_reg = np.stack([data[0] for data in outputs])
    arr2_reg = np.stack([data[1] for data in outputs])
    return arr1_reg, arr2_reg

# -----------------------------------------------------------------------------

def binarize(predict):
    
    def _process(img):
        mask = gaussian(img, sigma=5) > 0.5
        mask = remove_small_holes(mask, area_threshold=1024)
        edm = distance_transform_edt(mask)
        return mask, edm

    outputs = Parallel(n_jobs=-1)(
        delayed(_process)(img)
        for img in predict
        )
    mask = np.stack([data[0] for data in outputs])
    edm  = np.stack([data[1] for data in outputs])
    return mask, edm

# -----------------------------------------------------------------------------

# Register
path = Path(local_path, avi_name)
print("Register :", end='')
t0 = time.time()
rescale_reg, predict_reg = register(rescale, predict)
t1 = time.time()
print(f" {(t1-t0):<5.2f}s") 

# Binarize
path = Path(local_path, avi_name)
print("Binarize :", end='')
t0 = time.time()
mask, edm = binarize(predict_reg)
t1 = time.time()
print(f" {(t1-t0):<5.2f}s") 

# # Display
# viewer = napari.Viewer()
# viewer.add_image(rescale_reg)
# viewer.add_image(mask)
# viewer.add_image(skel)
# viewer.add_image(edm)

#%%

from skel import pixconn

skel = skeletonize(np.median(mask, axis=0))
conn = pixconn(skel, conn=2)

viewer = napari.Viewer()
viewer.add_image(rescale_reg)
viewer.add_image(skel, blending="additive")
viewer.add_image(conn, blending="additive")

#%% Results

# # Plots
# plt.plot(np.sum(mask, axis=(1, 2)))

# # Display
# viewer = napari.Viewer()
# viewer.add_image(rescale)
# viewer.add_image(mask)
# viewer.add_image(skel)
# viewer.add_image(edm)

