import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import streamlit as st
from PIL import Image
import requests
from PIL import Image
from io import BytesIO
import tensorflow as tf
from keras.models import load_model
from keras.models import load_model
from tensorflow.keras.models import load_model



st.title("Chest Cancer Detection")
st.write('Membuat machine learning yang dapat memprediksi Chest Carcinoma, dengan mengimplementasikan Computer Vision ANN')

st.title("Chest Cancer Classificatoin")
image = Image.open('output.png')
st.image(image, caption='chest_cancer')
#menampilkan penjelasan 
with st.expander("Explanation"):
    st.caption('Gambar diatas adalah jenis-jenis chest cancer')

st.title("Chest Cancer Augmentation")
image = Image.open('hasil_augmentasi.png')
st.image(image, caption='augmentation_chest_cancer')
#menampilkan penjelasan 
with st.expander("Explanation"):
    st.caption('Gambar diatas adalah hasil augmentasi')

st.write("Please upload an image or input URL")

file = st.file_uploader("Upload an image", type=["jpg", "png"])

model = load_model('best2.hdf5')
target_size=(400, 400)

def import_and_predict(image_data, model):
    image = load_img(image_data, target_size=(400, 400))
    img_array = img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Normalize the image
    img_array = img_array / 255.0

    # Make prediction
    predictions = model.predict(img_array)

    # Get the class with the highest probability
    idx = np.argmax(predictions)
    # predicted_class = np.argmax(predictions)

    jenis = ['adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib','large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa','normal','squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa']
    result = f"Prediction: {jenis[idx]}"

    return result

if file is None:
    st.text("Please upload an image file")
else:
    result = import_and_predict(file, model)
    st.image(file)
    st.write(result)

