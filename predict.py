import streamlit as st
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import os
import matplotlib.pyplot as plt
import numpy as np
from evaluation import generate_attention_map

# Set page config
st.set_page_config(
    page_title="Chest X-Ray Pneumonia Detection",
    page_icon="🫁",
    layout="wide"
)

# Custom CSS to improve the UI
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    # Load the model and processor
    model_path = "./vit-xray-output/checkpoint-1000"  # Use the latest checkpoint
    print(f"Loading model from: {model_path}")
    model = ViTForImageClassification.from_pretrained(model_path)
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    return model, processor

def process_image(image, processor):
    # Process image for model input
    # Convert grayscale to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    inputs = processor(image, return_tensors="pt")
    return inputs

def get_prediction(model, inputs):
    # Get model prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        # Add debug information
        st.sidebar.write("Raw logits:", outputs.logits.numpy())
        st.sidebar.write("Raw probabilities:", probabilities.numpy())
        
        # Use a higher threshold (0.7) for pneumonia predictions to counter the bias
        pneumonia_prob = probabilities[0][1].item()
        prediction = 1 if pneumonia_prob > 0.7 else 0
        
    return prediction, probabilities[0].tolist()

def main():
    # Header
    st.title("🫁 Chest X-Ray Pneumonia Detection")
    st.markdown("""
    This application uses a Vision Transformer (ViT) model to detect pneumonia in chest X-ray images.
    Upload an X-ray image to get started!
    """)

    # Load model and processor
    try:
        model, processor = load_model()
        model.eval()
    except Exception as e:
        st.error("Error loading the model. Please make sure the model is properly trained and saved.")
        return

    # File uploader
    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        # Create two columns for image and predictions
        col1, col2 = st.columns(2)

        # Display uploaded image
        with col1:
            st.subheader("Uploaded X-Ray")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)

        # Process image and get prediction
        inputs = process_image(image, processor)
        prediction, probabilities = get_prediction(model, inputs)
        
        # Display prediction results
        with col2:
            st.subheader("Analysis Results")
            st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
            
            # Display prediction
            prediction_label = "PNEUMONIA" if prediction == 1 else "NORMAL"
            prediction_color = "#ff4b4b" if prediction == 1 else "#00cc00"
            st.markdown(f"<h2 style='color: {prediction_color}; text-align: center;'>{prediction_label}</h2>", 
                      unsafe_allow_html=True)

            # Display confidence scores
            st.subheader("Confidence Scores:")
            normal_conf = probabilities[0] * 100
            pneumonia_conf = probabilities[1] * 100
            
            # Create confidence bars
            st.markdown("Normal:")
            st.progress(normal_conf / 100)
            st.markdown(f"{normal_conf:.1f}%")
            
            st.markdown("Pneumonia:")
            st.progress(pneumonia_conf / 100)
            st.markdown(f"{pneumonia_conf:.1f}%")
            
            st.markdown("</div>", unsafe_allow_html=True)

        # Generate and display attention map
        st.subheader("Attention Map Visualization")
        with st.spinner("Generating attention map..."):
            # Save uploaded image temporarily
            temp_path = "temp_image.jpg"
            image.save(temp_path)
            
            # Generate attention map
            attention_map_path = "temp_attention_map.jpg"
            generate_attention_map(model, temp_path, attention_map_path)
            
            # Display attention map
            if os.path.exists(attention_map_path):
                st.image(attention_map_path, use_column_width=True,
                        caption="Attention map showing regions the model focused on")
                
            # Clean up temporary files
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(attention_map_path):
                os.remove(attention_map_path)

    # Add information about the model
    with st.expander("About the Model"):
        st.markdown("""
        This application uses a Vision Transformer (ViT) model fine-tuned on a dataset of chest X-ray images.
        The model can detect patterns associated with pneumonia in chest X-rays with high accuracy.
        
        **Note:** This tool is for educational purposes only and should not be used as a substitute for 
        professional medical diagnosis.
        """)

if __name__ == "__main__":
    main() 