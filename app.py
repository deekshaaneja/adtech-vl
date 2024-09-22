import streamlit as st
from PIL import Image
import io

from qwen_2_inference import infer_new_model

st.title("VL Model Inference")

# Upload an image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

q1 = "Give me the title (in 25 characters) for personalized advertisement"
q2 = "Give me the description (in 90 characters) for personalized advertisement"
q3 = "Give me the keywords for advertisement of this product"
questions = [q1, q2, q3]

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Send image to backend for prediction
    if st.button("Get Prediction"):
        response = infer_new_model(image, questions)  # Call your model's inference function
        st.write("Model Output:", response)

