from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Add CORS middleware to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your model
model = tf.keras.models.load_model('fruits_model.h5')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and process the uploaded image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    img = img.resize((100, 100))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Make prediction
    prediction = model.predict(img_array)
    class_names = ['class1', 'class2', 'class3', 'class4']  # Update these with your actual class names
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction) * 100)
    
    return {
        "predicted_class": predicted_class,
        "confidence": confidence
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)