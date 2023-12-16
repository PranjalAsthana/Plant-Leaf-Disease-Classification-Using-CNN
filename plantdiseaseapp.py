import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps
#import cv2
from keras.models import load_model
# #from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array

st.set_option('deprecation.showfileUploaderEncoding', False)
def getResult(image, model):
    size = (255,255)
    imagedata = ImageOps.fit(image, size, Image.ANTIALIAS)
    img = np.asarray(imagedata)
    imgresh = img[np.newaxis,...]
    predictions = model.predict(imgresh)
    return predictions


#@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model


model = load_model()
st.title("Plant Disease Classification using CNN")
file = st.file_uploader("Upload", type=["png", "jpg", "jpeg"])

labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    st.write("")
    st.write("Classifying...")
    predictions = getResult(image, model)
    score = tf.nn.softmax(predictions[0])
    #st.write(predictions)
    #st.write(score)
    st.write("This image most likely belongs to {} with a {:.2f} percent confidence."
             .format(labels[np.argmax(score)], 100 * np.max(score)))







