import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

# Load the trained model
model = load_model("model.h5")

st.title(" Handwritten Digit Classifier (MNIST)")
st.write("Upload a 28x28 grayscale image of a digit (0–9) to predict.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # convert to grayscale
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = ImageOps.invert(image)  # MNIST is white digit on black bg
    image = image.resize((28, 28))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    st.subheader(f" Predicted Digit: {predicted_digit}")
    st.write(f"Prediction Confidence: {np.max(prediction)*100:.2f}%")
