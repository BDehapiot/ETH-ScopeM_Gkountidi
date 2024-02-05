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
# avi_name = "20231017-test 1+ 10nM erlotinib.avi"
# avi_name = "20231017-test 1+ PBS.avi"
# avi_name = "20231017-test 2+ 10nM erlotinib.avi"
# avi_name = "20231017-test 2+ PBS.avi"
# avi_name = "20231017-test 3+ 10nM erlotinib.avi"
avi_name = "20231017-test 3+ PBS.avi"
# avi_name = "20231017-test 4+ 1nM erlotinib.avi"
# avi_name = "20231017-test 4+ PBS.avi"
# # avi_name = "20231017-test 5+ 1nM erlotinib.avi"
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

from nan import nanreplace, nanfilt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from skimage.measure import label, regionprops

# -----------------------------------------------------------------------------

def interpolate(x_sig, y_sig, length):
    f = interp1d(x_sig, y_sig, kind='linear', fill_value="extrapolate")
    x_interp = np.linspace(0, length, length)   
    return f(x_interp)

def lowpass(sig, lowfreq, order):
    t = sig.shape[0]
    nyq = t / 2
    low = (lowfreq * t) / nyq
    b, a = butter(order, low, btype='low')
    return filtfilt(b, a, sig)

# -----------------------------------------------------------------------------

print("test :", end='')
t0 = time.time()

# mean mask
mmask = np.mean(mask, axis=0) > 0.5
labels = label(mmask)
props = regionprops(labels)
props_max = max(props, key=lambda r: r.area)
mmask = (labels == props_max.label)
skel = skeletonize(mmask, method="lee") > 0

# raw skel
raw_skel = rmap.copy()
raw_skel *= skel[np.newaxis, :, :]
raw_skel[raw_skel == 0] = np.nan
raw_skel = nanfilt(
    raw_skel, mask=mask,
    kernel_size=(3, 21, 21), # size parameter
    kernel_shape='ellipsoid',
    filt_method='mean',
    iterations=1,
    parallel=True
    )

rdata = {
    "idx" : [], 
    "raw" : [],
    "dRaw" : [],
    "norm" : [],
    "dNorm" : [],
    "baseline" : [],
    }

idxs = np.argwhere(skel) 
norm_skel = np.zeros_like(raw_skel)
for idx in idxs:
    
    # Extract data 
    raw = raw_skel[:, idx[0], idx[1]]
    dRaw = np.gradient(raw)

    # Normalized raw
    x_peaks, _ = find_peaks(raw, distance=10, prominence=0.01)
    y_peaks = raw[x_peaks]
    baseline = interpolate(x_peaks, y_peaks, raw.shape[0])
    baseline = lowpass(baseline, 0.075, 1)
    norm = raw / baseline 
    dNorm = np.gradient(norm)
    
    # Fill arrays
    norm_skel[:, idx[0], idx[1]] = norm
    
    # Append data
    rdata["idx"].append(idx)
    rdata["raw"].append(raw)
    rdata["dRaw"].append(dRaw)
    rdata["norm"].append(norm)
    rdata["dNorm"].append(dNorm)
    rdata["baseline"].append(baseline)
    
norm_skel[norm_skel == 0] = np.nan
norm_skel = nanreplace(
    norm_skel, mask=mask,
    kernel_size=(3, 21, 21), # size parameter
    kernel_shape='cuboid',
    filt_method='mean',
    iterations=3,
    parallel=True,
    )

norm_skel *= np.invert(skel[np.newaxis, :, :])
norm_skel[norm_skel == 0] = np.nan
norm_skel = nanreplace(
    norm_skel, mask=mask,
    kernel_size=(1, 21, 21), # size parameter
    kernel_shape='cuboid',
    filt_method='mean',
    iterations=1,
    parallel=True,
    )

t1 = time.time()
print(f" {(t1-t0):<5.2f}s")



#%% Display -------------------------------------------------------------------

viewer = napari.Viewer()
viewer.add_image(rescale_reg)
viewer.add_image(outl, blending="additive")
viewer.add_image(
    norm_skel, contrast_limits=(0.5, 1), colormap="viridis", opacity=0.33
    )

#%% Plot ----------------------------------------------------------------------

i = 200
t0, t1 = 0, 1200

# Plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 9))
ax1.plot(rdata["raw"][i][t0:t1])
ax1.plot(rdata["baseline"][i][t0:t1])
ax2.plot(rdata["norm"][i][t0:t1])
ax3.plot(rdata["dNorm"][i][t0:t1])
plt.tight_layout()
plt.show()
