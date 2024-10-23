import pandas as pd
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
data_dir = "./data"
validation_dir = "./Validation"
trained_model_path = "./trainedClassifier.h5"

# Function to preprocess image for model prediction
def preprocess_image(img):
    # Preprocess the image (resize, normalize, etc.)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0  # Normalize pixel values
    return img

# Function to load and train the model
def load_and_train_model():
    # Define the model architecture
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Define data generators
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Generate data iterators
    train_generator = train_datagen.flow_from_directory(data_dir, target_size=(256, 256), batch_size=32, class_mode='categorical')
    val_generator = val_datagen.flow_from_directory(validation_dir, target_size=(256, 256), batch_size=32, class_mode='categorical')

    # Train the model
    model.fit(train_generator, epochs=10, validation_data=val_generator)
    
    # Save the trained model
    model.save(trained_model_path)

    return model

# Load and train the model
load_and_train_model()
