import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split

# Define image parameters
IMG_SIZE = (64, 64)  # Reduced image size
BATCH_SIZE = 32

# Load dataset from extracted folder
train_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    'dataset',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# Manually split the data
def split_data(data_iterator, split_ratio=0.8):
    data_list = list(data_iterator)
    train_size = int(len(data_list) * split_ratio)
    train_data, val_data = data_list[:train_size], data_list[train_size:]
    return train_data, val_data

train_data, val_data = split_data(train_data)

print("Classes: ", train_data[0][1])  # Print class indices for the first batch

# Build a simpler CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')  # 4 classes
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model with fewer epochs
history = model.fit(
    train_data,
    epochs=3,  # Reduced epochs
    validation_data=val_data,
    verbose=1
)

# Save the model
model.save('fruits_model.keras')