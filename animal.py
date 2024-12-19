import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Set the page configuration
st.set_page_config(page_title="Animal Image Prediction", page_icon="🐾")

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = torch.load('animal.pth')  # Replace with your model path
    model.eval()  # Set the model to evaluation mode
    return model

model = load_model()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust size according to your model's input size
    transforms.ToTensor(),
])

# Function to preprocess the image
def import_and_predict(image_data, model):
    image = transform(image_data).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        prediction = model(image)
    return prediction

# Sidebar for the app
st.sidebar.title("Upload an Image")
file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    
    # Make prediction
    predictions = import_and_predict(image, model)
    
    # Assuming class_names is a list of your model's output classes
    class_names = ['ragno', 'cane', 'cavallo', 'elefante', 'scoiattolo', 'farfalla', 'gallina', 'mucca', 'pecora', 'gatto']  # Replace with your classes
    predicted_class = class_names[torch.argmax(predictions).item()]
    
    st.write(f"Predicted Class: {predicted_class}")
    st.sidebar.success(f"Prediction: {predicted_class}")

    # Display prediction confidence
    confidence = torch.max(predictions).item() * 100
    st.sidebar.info(f"Confidence: {confidence:.2f}%")
