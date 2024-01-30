#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
from skimage import io
from pathlib import Path
from pystackreg import StackReg
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Skimage
from skimage.filters import gaussian
from skimage.transform import downscale_local_mean
from skimage.morphology import (
    remove_small_holes, skeletonize, disk, dilation, binary_dilation
    )

# Scipy
from scipy.ndimage import distance_transform_edt

#%% Inputs --------------------------------------------------------------------

local_path = Path("D:\local_Gkountidi\data")
avi_name = "20231017-test 1+ 10nM erlotinib.avi"
# avi_name = "20231017-test 1+ PBS.avi"
# avi_name = "20231017-test 2+ 10nM erlotinib.avi"
# avi_name = "20231017-test 2+ PBS.avi"
# avi_name = "20231017-test 3+ 10nM erlotinib.avi"
# avi_name = "20231017-test 3+ PBS.avi"
# avi_name = "20231017-test 4+ 1nM erlotinib.avi"
# avi_name = "20231017-test 4+ PBS.avi"
# avi_name = "20231017-test 5+ 1nM erlotinib.avi"
# avi_name = "20231017-test 5+ PBS.avi"
# avi_name = "20231017-test 6+ PBS.avi"

# Parameters
rescale_factor = 2 # rescale factor for registration

#%% Comments ------------------------------------------------------------------

''' 
'''

#%% 

# Open data
rescale = io.imread(Path(local_path, avi_name.replace(".avi", "_rescale.tif")))
predict = io.imread(Path(local_path, avi_name.replace(".avi", "_predict.tif")))

#%%

def register(arr1, arr2, rescale_factor):
        
    def _imreg(ref1, mov1, mov2):
        sr = StackReg(StackReg.RIGID_BODY)
        sr.register(
            downscale_local_mean(ref1, rescale_factor),
            downscale_local_mean(mov1, rescale_factor),
            )
        sr._m[:, 0] *= rescale_factor
        mov1 = sr.transform(mov1)
        mov2 = sr.transform(mov2)
        return mov1, mov2

    outputs = Parallel(n_jobs=-1)(
        delayed(_imreg)(arr1[0,...], arr1[t,...], arr2[t,...])
        for t in range(1, arr1.shape[0])
        )
    arr1_reg = [data[0] for data in outputs]
    arr1_reg.insert(0, arr1[0,...])
    arr1_reg = np.stack(arr1_reg)
    arr2_reg = [data[1] for data in outputs]
    arr2_reg.insert(0, arr2[0,...])
    arr2_reg = np.stack(arr2_reg)
    return arr1_reg, arr2_reg

# -----------------------------------------------------------------------------

def binarize(predict):
    
    def _binarize(img):
        mask = gaussian(img, sigma=5) > 0.5 # size parameter
        mask = remove_small_holes(mask, area_threshold=1024) # size parameter  
        outl = binary_dilation(mask) ^ mask
        rmap = distance_transform_edt(mask)
        rmap = gaussian(rmap, sigma=5) # size parameter
        # rmap = dilation(rmap, footprint=disk(5))
        return mask, outl, rmap

    outputs = Parallel(n_jobs=-1)(
        delayed(_binarize)(img)
        for img in predict
        )
    mask = np.stack([data[0] for data in outputs])
    outl = np.stack([data[1] for data in outputs])
    rmap  = np.stack([data[2] for data in outputs])
    return mask, outl, rmap

# -----------------------------------------------------------------------------

# Register
path = Path(local_path, avi_name)
print("Register :", end='')
t0 = time.time()
rescale_reg, predict_reg = register(rescale, predict, rescale_factor)
t1 = time.time()
print(f" {(t1-t0):<5.2f}s") 

# Binarize
path = Path(local_path, avi_name)
print("Binarize :", end='')
t0 = time.time()
mask, outl, rmap = binarize(predict_reg)
t1 = time.time()
print(f" {(t1-t0):<5.2f}s") 

# # Display
# viewer = napari.Viewer()
# viewer.add_image(rescale_reg)
# viewer.add_image(mask)
# viewer.add_image(rmap)

#%%

from skimage.measure import label, regionprops
from nan import nanreplace, nanfilt

# -----------------------------------------------------------------------------

print("test :", end='')
t0 = time.time()

#
mean_mask = np.mean(mask, axis=0) > 0.5
labels = label(mean_mask)
props = regionprops(labels)
props_max = max(props, key=lambda r: r.area)
mean_mask = (labels == props_max.label)
skel = skeletonize(mean_mask, method="lee") > 0
rskel = rmap.copy()
rskel *= skel[np.newaxis, :, :]
rskel[rskel == 0] = np.nan

rskel = nanfilt(
    rskel, mask=mask,
    kernel_size=(9, 21, 21), # size parameter
    kernel_shape='cuboid',
    filt_method='mean',
    iterations=1,
    parallel=True
    )

drskel = np.gradient(rskel, axis=0) * -1

rskel = nanreplace(
    rskel, mask=mask,
    kernel_size=(1, 21, 21), # size parameter
    kernel_shape='cuboid',
    filt_method='mean',
    iterations=1,
    parallel=True,
    )

drskel = nanreplace(
    drskel, mask=mask,
    kernel_size=(1, 21, 21), # size parameter
    kernel_shape='cuboid',
    filt_method='mean',
    iterations=1,
    parallel=True,
    )

t1 = time.time()
print(f" {(t1-t0):<5.2f}s")

viewer = napari.Viewer()
viewer.add_image(rescale_reg)
viewer.add_image(drskel)

# viewer.add_image(rescale_reg)
# viewer.add_image(outl)
# viewer.add_image(rskel, colormap="plasma")
    
#%%
    
#
# skel_edm = []
# idxs = np.argwhere(skel) 
# for idx in idxs:
#     edm = np.zeros_like(skel)
#     edm[idx[0], idx[1]] = True
#     skel_edm.append(distance_transform_edt(np.invert(edm)))
# skel_edm = np.stack(skel_edm)
# skel_edm = np.argmin(skel_edm, axis=0)
# skel_edm = np.repeat(skel_edm[np.newaxis, :, :], rescale.shape[0], axis=0)
# skel_edm[mask == False] = 0

# #
# mask_edm = []
# idxs = np.where(skel)
# for t in range(rescale.shape[0]):
#     tmp = rmap[t,...]
#     mapping = tmp[idxs]
#     mapping[0] = 0
#     mask_edm.append(mapping[skel_edm[t,...]])
# mask_edm = np.stack(mask_edm)   
    
# viewer = napari.Viewer()
# viewer.add_image(skel_edm)
# viewer.add_image(mask_edm)

# rad_map = rmap.copy()
# rad_map = rad_map * skel[np.newaxis, :, :]

# radii = []
# idxs = np.argwhere(skel) 
# for idx in idxs:
#     radii.append(rmap[:, idx[0], idx[1]])
    
# i = 780
# plt.plot(radii[i]) 


#%% Results

# # Plots
# plt.plot(np.sum(mask, axis=(1, 2)))

# # Display
# viewer = napari.Viewer()
# viewer.add_image(rescale)
# viewer.add_image(mask)
# viewer.add_image(skel)
# viewer.add_image(rmap)

