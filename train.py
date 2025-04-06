import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
IMG_LEN = 224  # Slightly smaller for faster training
IMG_SIZE = (IMG_LEN, IMG_LEN)
BATCH_SIZE = 16
EPOCHS = 10  # Limited for speed
DATASET_PATH = "dataset/PlantVillage"
MODEL_PATH = "plant_disease_model.h5"
CLASS_INDEX_PATH = "class_indices.json"

# Data Augmentation + Preprocessing
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)
val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Save class labels
with open(CLASS_INDEX_PATH, "w") as f:
    json.dump(train_generator.class_indices, f)

# Load base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_LEN, IMG_LEN, 3))
base_model.trainable = True

# Freeze all layers except the last 20 for faster fine-tuning
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Build model
model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True)
]

# Train model
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Save final model
model.save(MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")
