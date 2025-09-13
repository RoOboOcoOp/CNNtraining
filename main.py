import os
import time  # Added for timing

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Start timing after imports
start_time = time.time()

# Set memory growth to prevent GPU memory issues
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

# Data preparation
root = 'PetImages'
categories = [x[0] for x in os.walk(root) if x[0]][1:]
print("Categories found:", categories)

# Use Keras' built-in ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create data generators
datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.3,  # 30% for validation
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2
)

# Create training generator
train_generator = datagen.flow_from_directory(
    root,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Create validation generator
validation_generator = datagen.flow_from_directory(
    root,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Get the number of classes
num_classes = len(train_generator.class_indices)
print(f"Number of classes: {num_classes}")

# Model definition
input_shape = (128, 128, 3)

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Conv2D(64, (3, 3), padding='same'),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(64, (3, 3), padding='same'),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Dropout(0.25),

    keras.layers.Flatten(),
    keras.layers.Dense(128),
    keras.layers.Activation('relu'),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(num_classes),
    keras.layers.Activation('softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    ),
    keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True,
        monitor='val_loss'
    ),
    keras.callbacks.ReduceLROnPlateau(
        factor=0.2,
        patience=3,
        min_lr=1e-7
    )
]

# Training
print("Starting training...")
train_start_time = time.time()
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=callbacks,
    verbose=1
)
train_end_time = time.time()

# Evaluation
print("Evaluating on validation set...")
eval_start_time = time.time()
test_results = model.evaluate(validation_generator, verbose=1)
eval_end_time = time.time()

# Calculate timing results
total_time = time.time() - start_time
training_time = train_end_time - train_start_time
evaluation_time = eval_end_time - eval_start_time

print(f"\n=== Timing Results ===")
print(f"Total execution time: {total_time:.2f} seconds")
print(f"Training time: {training_time:.2f} seconds")
print(f"Evaluation time: {evaluation_time:.2f} seconds")
print(f"Test loss: {test_results[0]:.4f}, Test accuracy: {test_results[1]:.4f}")

# Plotting
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.ylim(0, 1)

plt.tight_layout()
plt.show()
