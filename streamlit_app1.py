import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Define paths
trained_model_path = "./trainedClassifier.h5"

# Function to preprocess image for model prediction
def preprocess_image(img):
    # Preprocess the image (resize, normalize, etc.)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0  # Normalize pixel values
    return img

# Function to predict foot arch type
def predict_arch(image):
    # Preprocess the image
    preprocessed_img = preprocess_image(image)
    # Make prediction using the loaded model
    prediction = model.predict(np.expand_dims(preprocessed_img, axis=0))
    # Decode prediction into arch type (e.g., flat, normal, high)
    arch_types = ['Flat', 'Normal', 'High']
    predicted_arch_type = arch_types[np.argmax(prediction)]
    return predicted_arch_type

# Load the pre-trained model
model = load_model(trained_model_path)

# Streamlit app code
st.title("Foot Arch Detector")

# Allow user to upload an image
uploaded_image = st.file_uploader("Upload an image of your foot", type=['png', 'jpg', 'jpeg'])

if uploaded_image is not None:
    # Read the uploaded image
    img = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    # Display the uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Predict foot arch type
    prediction = predict_arch(img)
    # Display prediction result
    st.write("Predicted foot arch type:", prediction)
