import streamlit as st
from PIL import Image
import io

from qwen_2_inference import infer_new_model, get_trained_model, get_default_model, get_quantized_train_model

# trained_model, trained_processor = get_quantized_train_model()  # Use your quantized model function
st.title("DB AdGenie")

# Upload an image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

q1 = "Give me the title for this Tech Product (in 25 characters) for advertisement"
q2 = "Give me the description for this Tech Product (in 90 characters) for advertisement"
q3 = "Give me 5 keywords for this Tech Product for advertising purpose"
questions = [q1, q2, q3]

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Send image to backend for prediction
    if st.button("Get Prediction"):
        response = infer_new_model(image, questions)
        print("==========================RESPONSE=======================================")
        print(response)
        # Extract title, description, and keywords from the response
        response_dict = {"title": response[0][0], "description": response[1][0], "keywords": response[2][0]}
        title = response_dict.get("title", "No Title Generated")  # Adjust based on response structure
        description = response_dict.get("description", "No Description Generated")  # Adjust based on response structure
        keywords = response_dict.get("keywords", ["No Keywords Generated"])
        # keywords_joined = '\n'.join(keywords)# Adjust based on response structure

        # Display structured output
        st.write(f"**Title Text:** {title}")
        st.write(f"**Body Text (Description):** {description}")
        st.write(f"**Keywords:** {keywords}")  # Join keywords into a single string for display
else:
    st.write("Please upload an image to see the prediction button.")
