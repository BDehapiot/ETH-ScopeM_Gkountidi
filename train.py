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

# TensorFlow
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#%% Inputs --------------------------------------------------------------------

# Data augmentation
iterations = 100
random.seed(42) 

# Train model
validation_split = 0.2
n_epochs = 100
batch_size = 8

# Paths
train_path = Path(Path.cwd(), "data", "train")

#%% Pre-processing ------------------------------------------------------------

# Open training data
images, masks = [], []
for path in train_path.iterdir():
    if 'mask' in path.name:
        
        # Open masks
        masks.append(io.imread(path).astype("float"))
        
        # Open & normalize images
        image = io.imread(str(path).replace('_mask', ''))
        pMax = np.percentile(image, 99.9)
        image[image > pMax] = pMax
        image = (image / pMax).astype(float)
        images.append(image)                  
        
images = np.stack(images)
masks = np.stack(masks)

# # Display 
# viewer = napari.Viewer()
# viewer.add_image(images)
# viewer.add_image(masks)        

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
        delayed(augment_data)(images, masks, operations)
        for i in range(iterations)
        )
    images = np.stack([data[0] for data in outputs])
    masks = np.stack([data[1] for data in outputs])
    
    # # Display 
    # viewer = napari.Viewer()
    # viewer.add_image(images)
    # viewer.add_image(masks)    
    
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

# Checkpoint & callback
model_checkpoint_callback = ModelCheckpoint(
    filepath="model_weights.h5",
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

callbacks = [
    EarlyStopping(patience=20, monitor='val_loss'),
    model_checkpoint_callback
]

# train model
history = model.fit(
    x=images,
    y=masks,
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

# Save model
# model.save_weights(Path(Path.cwd(), "model_weights.h5"))
# model.save(Path(Path.cwd(), "model.pb"), save_format="tf")