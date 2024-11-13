import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('model_weights/vgg19_model_03.h5')

# Class labels dictionary
class_labels = {0: 'NORMAL', 1: 'PNEUMONIA'}

def preprocess_image(img_path, target_size=(128, 128)):
    """Preprocess the image to be used in the model prediction."""
    img = cv2.imread(img_path)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and prediction."""
    if request.method == 'POST':
        # Check if the file is uploaded
        if 'file' not in request.files:
            return "No file uploaded", 400
        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400

        if file:
            filepath = os.path.join('uploads', file.filename)
            file.save(filepath)

            # Preprocess the image
            img = preprocess_image(filepath)
            
            # Model prediction
            prediction = model.predict(img)
            class_index = np.argmax(prediction)
            result = class_labels[class_index]

            # Clean up uploaded file
            os.remove(filepath)

            return f'Prediction: {result}'

    return render_template('upload.html')

if __name__ == '__main__':
    # Create the uploads directory if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # Run the app
    app.run(debug=True)
