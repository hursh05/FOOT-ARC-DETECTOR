import streamlit as st
import cv2
import numpy as np
from my_utils import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Foot Arch Analyser", layout="centered")
test_dir = "./test"

st.title("Your Foot Doctor")

st.header("Examples")
st.subheader("Taking pictures like these will help you get the best results!")

with st.container():
    demo_imgs = ('./media/NORMAL1.jpg', './media/LOW1.jpg')
    cols = st.columns(2)
    for col_no in range(len(demo_imgs)):
        cols[col_no % len(cols)].image(demo_imgs[col_no], use_column_width=True)

st.header("Get Your Foot Arch Analysed!")

if st.button("Load Data and Train Model"):
    st.info("Loading Data")
    pre_df = get_data()
    pre_df["Images"] = pre_focus_crop(pre_df)
    df = pre_df[["Images", "Target"]]
    get_model()

img_file_upload = st.sidebar.file_uploader(label="Upload an image", type=['PNG', 'JPG'])

img_file_buffer = st.camera_input(label="Take a picture OR Upload an Image from the sidebar",
                                  help="Allow camera access and take a picture like the examples to help us analyse your foot shape for the best results")

def process_and_predict(image):
    try:
        cv2.imwrite(test_dir + "/NORMAL/test_img.jpg", image)
        model = load_model("./trainedClassifier.h5")

        test_data_generator = ImageDataGenerator(rescale=1./255)
        image_size = (256, 256)

        test_data_iterator = test_data_generator.flow_from_directory(
            test_dir,
            target_size=image_size,
            batch_size=1,
            class_mode='categorical',
            shuffle=False
        )

        predictions = model.predict(test_data_iterator)
        st.write("Predicted class probabilities on test set:", predictions)
    except Exception as e:
        st.error(f"An error occurred: {e}")

if img_file_upload is not None:
    img = np.asarray(Image.open(img_file_upload))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    process_and_predict(img)

elif img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    if cv2_img is not None:
        st.write(type(cv2_img))
        st.write(cv2_img.shape)

        img = cv2.cvtColor(cv2.resize(cv2_img, (1280, 720)), cv2.COLOR_RGB2BGR)
        st.image(img)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=3)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)
            crop_img = cv2.resize(img[y:y+h, x:x+w], (256, 256))

            np.reshape(crop_img, (256, 256, 3))
            st.write(crop_img.shape)

            st.image(crop_img)
            process_and_predict(crop_img)
        else:
            st.warning("Cannot identify feet in the image. Please try again with a different one.")
