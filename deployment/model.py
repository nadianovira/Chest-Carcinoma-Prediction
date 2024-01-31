import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.models import load_model


# Load your trained model (assuming model_improve is defined)
def run():
    model_improve = load_model('best2.hdf5')
    class_labels = ['adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib','large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa','normal','squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa']

    def prediction(file):
        img = tf.keras.utils.load_img(file, target_size=(400, 400))
        x = tf.keras.utils.img_to_array(img) / 255
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model_improve.predict(images, batch_size=10)
        idx = np.argmax(classes)
        clas = ['adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib','large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa','normal','squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa']
        print('Prediction is a {}'.format(clas[idx]))
        return class_labels[idx]

    # Streamlit app
    st.title("Chest Cancer Detection")

    # Choose prediction method: Upload or Link
    prediction_method = st.radio("Choose prediction method:", ("Upload", "Link"))

    if prediction_method == "Upload":
        # Image upload
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Make prediction when the user clicks the "Predict" button
            if st.button("Predict"):
                prediction_result = prediction(uploaded_file)
                st.success(f"Prediction: {prediction_result}")

    elif prediction_method == "Link":
        # Image link
        image_url = st.text_input("Enter image URL:")

        if image_url:
            # Display the image from the provided link
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption="Image from URL", use_column_width=True)

            # Make prediction when the user clicks the "Predict" button
            if st.button("Predict"):
                prediction_result = prediction(BytesIO(response.content))
                st.success(f"Prediction: {prediction_result}")