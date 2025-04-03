# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # Define image parameters
# IMG_SIZE = (100, 100)
# BATCH_SIZE = 32

# # Load dataset from extracted folder
# train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# train_data = train_datagen.flow_from_directory(
#     'dataset', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='training')

# val_data = train_datagen.flow_from_directory(
#     'dataset', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='validation')

# print("Classes: ", train_data.class_indices)

# # Build the CNN model
# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(4, activation='softmax')  # 4 classes
# ])

# # Compile the model
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# # Print the model summary
# model.summary()

# # Train the model
# history = model.fit(train_data, epochs=10, validation_data=val_data)

# # Save the model
# model.save('fruits_model.keras')

# # Load the saved model
# model = tf.keras.models.load_model('fruits_model.keras')

