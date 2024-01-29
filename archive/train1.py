#%% Imports -------------------------------------------------------------------

import napari
import random
import numpy as np
from skimage import io
from pathlib import Path
import albumentations as A
import matplotlib.pyplot as plt
import segmentation_models as sm
from joblib import Parallel, delayed 
from skimage.transform import downscale_local_mean

# TensorFlow
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Functions
from functions import get_patches

#%% Inputs --------------------------------------------------------------------

# Paths
train_path = Path(Path.cwd(), 'data', 'train')

# Patches
rescale_factor = 2
size = 512 // rescale_factor
overlap = size // 8

# Data augmentation
random.seed(42)
iterations = 1000

# Train model
validation_split = 0.2
n_epochs = 100
batch_size = 16

#%% Pre-processing ------------------------------------------------------------

iPatches, mPatches = [], []
for path in train_path.iterdir():
    if 'mask' in path.name:
        
        # Open data
        msk = io.imread(path).astype(bool).astype(float)
        img = io.imread(str(path).replace('_mask', ''))
        
        # Downscale images (optional)
        if rescale_factor != 1:
            msk = (downscale_local_mean(msk, rescale_factor) > 0.5).astype(float)
            img = downscale_local_mean(img, rescale_factor)
        
        # Open & normalize image
        pMax = np.percentile(img, 99.9)
        img[img > pMax] = pMax
        img = (img / pMax).astype(float)
        
        # Extract patches
        iPatches.append(get_patches(img, size, overlap))
        mPatches.append(get_patches(msk, size, overlap))
       
imgs = np.stack([patch for patches in iPatches for patch in patches])
msks = np.stack([patch for patches in mPatches for patch in patches])

# # Display 
# viewer = napari.Viewer()
# viewer.add_image(imgs)
# viewer.add_image(msks) 

#%% Augmentation --------------------------------------------------------------

augment = True if iterations > 0 else False

if augment:
    
    # Define augmentation operations
    operations = A.Compose([
        A.VerticalFlip(p=0.5),              
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.GridDistortion(p=0.5),
        ])

    # Augment data
    def augment_data(images, masks, operations):      
        idx = random.randint(0, len(images) - 1)
        outputs = operations(image=images[idx,...], mask=masks[idx,...])
        return outputs['image'], outputs['mask']
    outputs = Parallel(n_jobs=-1)(
        delayed(augment_data)(imgs, msks, operations)
        for i in range(iterations)
        )
    imgs = np.stack([data[0] for data in outputs])
    msks = np.stack([data[1] for data in outputs])
    
    # # Display 
    # viewer = napari.Viewer()
    # viewer.add_image(imgs)
    # viewer.add_image(msks) 

#%% Model training ------------------------------------------------------------

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

# Checkpoint & callbacks
model_checkpoint_callback = ModelCheckpoint(
    filepath="model_weights.h5",
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True
    )
callbacks = [
    EarlyStopping(patience=20, monitor='val_loss'),
    model_checkpoint_callback
    ]

# train model
history = model.fit(
    x=imgs, y=msks,
    validation_split=validation_split,
    batch_size=batch_size,
    epochs=n_epochs,
    callbacks=callbacks,
    )

# Plot training results
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()