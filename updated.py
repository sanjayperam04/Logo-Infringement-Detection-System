import streamlit as st
import cv2
import numpy as np
import jellyfish
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import KMeans
import easyocr
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import torch
from torchvision import models, transforms
import io
import tempfile
import os

class LogoInfringementDetector:
    def __init__(self):
        """Initialize the Logo Infringement Detector with necessary components"""
        # Initialize OCR engine
        with st.spinner('Loading OCR engine...'):
            self.reader = easyocr.Reader(['en'])
            
        # Initialize VGG model for deep feature extraction
        with st.spinner('Loading VGG model...'):
            self.vgg_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
            self.vgg_model.eval()
            self.vgg_model.add_module('avgpool', torch.nn.AdaptiveAvgPool2d((1, 1)))
            # Define image transformations for VGG
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Set device for model inference
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.vgg_model = self.vgg_model.to(self.device)
    
    def extract_text(self, image_path):
        """Extract text from logo image using EasyOCR"""
        results = self.reader.readtext(image_path)
        text = ' '.join([result[1] for result in results])
        return text.strip().lower()
    
    def calculate_text_similarity(self, text1, text2):
        """Calculate text similarity using Levenshtein distance"""
        if not text1 or not text2:
            return 0.0
            
        # Calculate Levenshtein distance
        distance = jellyfish.levenshtein_distance(text1, text2)
        max_len = max(len(text1), len(text2))
        
        if max_len == 0:
            return 0.0
            
        # Convert distance to similarity percentage
        similarity = (1 - (distance / max_len)) * 100
        return similarity
    
    def extract_dominant_colors(self, image_path, n_colors=5):
        """Extract dominant colors from the logo"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Reshape image for KMeans
        pixels = image.reshape(-1, 3)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_colors, n_init=10)
        kmeans.fit(pixels)
        
        # Get dominant colors
        colors = kmeans.cluster_centers_
        
        return colors
    
    def calculate_color_similarity(self, colors1, colors2):
        """Calculate similarity between two sets of dominant colors"""
        # Calculate pairwise distances between colors
        min_distances = []
        
        for color1 in colors1:
            distances = [np.linalg.norm(color1 - color2) for color2 in colors2]
            min_distances.append(min(distances))
        
        # Calculate average minimum distance
        avg_distance = np.mean(min_distances)
        
        # Convert to similarity (max possible distance in RGB space is 255*sqrt(3))
        max_distance = 255 * np.sqrt(3)
        similarity = (1 - (avg_distance / max_distance)) * 100
        
        return similarity
    
    def extract_deep_features(self, image_path):
        """Extract deep features from the logo using VGG16"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.vgg_model(image_tensor)
        
        # Flatten features
        features = features.view(features.size(0), -1)
        
        return features.cpu().numpy().flatten()
    
    def calculate_visual_similarity(self, image_path1, image_path2):
        """Calculate visual similarity using SSIM and deep features"""
        # Calculate SSIM
        img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
        
        # Resize images to the same dimensions
        height = min(img1.shape[0], img2.shape[0])
        width = min(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (width, height))
        img2 = cv2.resize(img2, (width, height))
        
        ssim_score = ssim(img1, img2)
        
        # Calculate deep feature similarity
        features1 = self.extract_deep_features(image_path1)
        features2 = self.extract_deep_features(image_path2)
        
        deep_sim = 1 - cosine(features1, features2)
        
        # Combine SSIM and deep feature similarity
        visual_sim = (ssim_score + deep_sim) / 2 * 100
        
        return visual_sim
    
    def detect_infringement(self, logo1_path, logo2_path, threshold=55.0):
        """Detect potential trademark infringement between two logos"""
        results = {}
        
        # Step 1: Check text similarity
        with st.spinner('Analyzing text similarity...'):
            text1 = self.extract_text(logo1_path)
            text2 = self.extract_text(logo2_path)
            
            text_similarity = self.calculate_text_similarity(text1, text2)
            results['text_similarity'] = text_similarity
            results['text1'] = text1
            results['text2'] = text2
            
            if text_similarity >= threshold:
                results['infringement_detected'] = True
                results['infringement_type'] = 'Text Similarity'
                results['final_similarity'] = text_similarity
                return results
        
        # Step 2: Check visual/color similarity
        with st.spinner('Analyzing visual and color similarity...'):
            colors1 = self.extract_dominant_colors(logo1_path)
            colors2 = self.extract_dominant_colors(logo2_path)
            
            color_similarity = self.calculate_color_similarity(colors1, colors2)
            visual_similarity = self.calculate_visual_similarity(logo1_path, logo2_path)
            
            # Average of color and visual similarity
            visual_color_similarity = (color_similarity + visual_similarity) / 2
            results['visual_color_similarity'] = visual_color_similarity
            results['color_similarity'] = color_similarity
            results['visual_similarity'] = visual_similarity
            
            if visual_color_similarity >= threshold:
                results['infringement_detected'] = True
                results['infringement_type'] = 'Visual/Color Similarity'
                results['final_similarity'] = visual_color_similarity
                return results
        
        # No infringement detected
        results['infringement_detected'] = False
        results['final_similarity'] = max(text_similarity, visual_color_similarity)
        
        return results

def save_uploaded_file(uploaded_file):
    """Save an uploaded file to a temporary location and return the path"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
        temp_file.write(uploaded_file.getvalue())
        return temp_file.name

def main():
    st.set_page_config(
        page_title="Logo Infringement Detector",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("Logo Infringement Detection System")
    st.markdown("""
    This application helps detect potential trademark infringement between two logos by analyzing:
    - Text similarity through OCR
    - Visual and color similarity
    
    Upload two logo images to compare them.
    """)
    
    # Sidebar for threshold adjustment
    st.sidebar.title("Settings")
    threshold = st.sidebar.slider(
        "Similarity Threshold (%)", 
        min_value=40.0, 
        max_value=90.0, 
        value=55.0, 
        step=1.0,
        help="Minimum similarity percentage to flag potential infringement"
    )
    
    # File uploaders
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Original Logo")
        logo1_file = st.file_uploader("Upload original logo", type=["jpg", "jpeg", "png"])
        
    with col2:
        st.header("Compared Logo")
        logo2_file = st.file_uploader("Upload logo to compare", type=["jpg", "jpeg", "png"])
    
    # Initialize detector
    if 'detector' not in st.session_state:
        st.session_state.detector = LogoInfringementDetector()
    
    # Process logos if both are uploaded
    if logo1_file and logo2_file:
        # Save uploaded files
        logo1_path = save_uploaded_file(logo1_file)
        logo2_path = save_uploaded_file(logo2_file)
        
        # Display the uploaded images
        col1, col2 = st.columns(2)
        with col1:
            st.image(logo1_file, caption="Original Logo", use_container_width=True)
        with col2:
            st.image(logo2_file, caption="Compared Logo", use_container_width=True)
        
        # Add a button to start analysis
        if st.button("Analyze Logos for Infringement"):
            with st.spinner("Analyzing logos for potential infringement..."):
                # Detect infringement
                results = st.session_state.detector.detect_infringement(logo1_path, logo2_path, threshold)
                
                # Display results
                st.subheader("Analysis Results")
                
                # Infringement status with appropriate styling
                if results['infringement_detected']:
                    st.error(f"⚠️ POTENTIAL INFRINGEMENT DETECTED: {results['infringement_type']}")
                else:
                    st.success("✅ NO INFRINGEMENT DETECTED")
                
                # Create metrics for similarity scores
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Text Similarity", f"{results.get('text_similarity', 0):.1f}%")
                with col2:
                    st.metric("Visual/Color Similarity", f"{results.get('visual_color_similarity', 0):.1f}%")
                
                # Display extracted text
                st.subheader("Extracted Text")
                text_col1, text_col2 = st.columns(2)
                with text_col1:
                    st.info(f"Original Logo Text: {results.get('text1', 'None detected')}")
                with text_col2:
                    st.info(f"Compared Logo Text: {results.get('text2', 'None detected')}")
                
                # Detailed results in an expander
                with st.expander("Detailed Analysis Results"):
                    st.json({
                        "infringement_detected": results['infringement_detected'],
                        "infringement_type": results.get('infringement_type', 'None'),
                        "text_similarity": f"{results.get('text_similarity', 0):.2f}%",
                        "visual_similarity": f"{results.get('visual_similarity', 0):.2f}%",
                        "color_similarity": f"{results.get('color_similarity', 0):.2f}%",
                        "final_similarity": f"{results.get('final_similarity', 0):.2f}%",
                        "threshold": f"{threshold}%"
                    })
                
                # Clean up temporary files
                os.unlink(logo1_path)
                os.unlink(logo2_path)
    
    # Footer
    st.markdown("---")
    st.markdown("Logo Infringement Detection System © 2025")

if __name__ == "__main__":
    main()
