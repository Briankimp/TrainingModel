import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('fruits_model.h5')

def predict_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(100, 100))  # Changed from 64 to 100
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    
    # Make prediction
    prediction = model.predict(img_array)
    
    # Get class names (make sure these match your training classes)
    class_names = ['class1', 'class2', 'class3', 'class4']  # Replace with your actual class names
    
    # Get the predicted class
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    return predicted_class, confidence

# Test the model with the image
test_image_path = 'download.webp'
predicted_class, confidence = predict_image(test_image_path)
print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")