import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# Add some content to the app
st.title('Pizza Classifier')
st.write('Upload an image of a some dish and click the "Classify" button to predict')
uploaded_file = st.file_uploader("Choose file...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    bytes_image = BytesIO()
    image.save(bytes_image, format="JPEG")
    bytes_image = bytes_image.getvalue()
    st.image(image, caption='Uploaded Image', use_column_width=True)
    if st.button('Classify'):
        predicted = requests.post("http://127.0.0.1:5000/predict", files={'file': bytes_image})
        predicted = predicted.json()
        if predicted['prediction'] == 0:
            st.write('Prediction:', 'Not Pizza')
        else:
            st.write('Prediction:', 'Pizza!')