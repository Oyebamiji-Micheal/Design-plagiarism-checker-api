import os
import torch
import joblib
import random
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Dict, Any

import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics.pairwise import cosine_similarity

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import json

# Load image filename to S3 URL mapping from JSON file
def load_image_url_mapping(file_path='img-paths.json'):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Using empty mapping.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {file_path}. Using empty mapping.")
        return {}

# Load the image URL mapping
IMAGE_URL_MAPPING = load_image_url_mapping()

class ImageRequest(BaseModel):
    image_path: str

# Function to load VGG19 model
def load_model():
    vgg19_weights_path = 'vgg19/vgg19_weights.pth'

    # Check if the vgg19 folder exists, if not create and download the model
    if not os.path.exists('vgg19'):
        os.makedirs('vgg19')  # Create directory for vgg19 weights
        # Download the VGG19 pre-trained weights
        vgg19 = models.vgg19(pretrained=True)
        torch.save(vgg19.state_dict(), vgg19_weights_path)
    else:
        # Load the VGG19 model from saved weights
        vgg19 = models.vgg19()
        vgg19.load_state_dict(torch.load(vgg19_weights_path, map_location=torch.device('cpu')))
    
    # Remove the classifier part (only using the feature extractor)
    model = torch.nn.Sequential(*list(vgg19.children())[:-2])
    model.eval()
    
    return model

# Transformation to resize and normalize images for VGG19
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to extract image embeddings using VGG19
def get_image_features(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = vgg19_model(image).numpy().flatten()  # Flatten the output tensor
    return features

# Download image from URL
def download_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading image: {str(e)}")

# Initialize the app and model
app = FastAPI()
vgg19_model = load_model()

# Load pre-saved embeddings
embeddings = joblib.load('image-embeddings/image_embeddings.joblib')

@app.post("/find-similar-designs")
async def find_similar_designs(request: ImageRequest):
    # Get the input image URL
    input_url = request.image_path
    
    # Download and process the input image
    uploaded_image = download_image(input_url)
    
    # Get embeddings of the uploaded image
    uploaded_image_embedding = get_image_features(uploaded_image)
    
    # Calculate cosine similarity with all saved embeddings
    similarities = {}
    for filename, embedding in embeddings.items():
        similarity = cosine_similarity([uploaded_image_embedding], [embedding])[0][0]
        
        # Map local filename to S3 URL if possible
        if filename in IMAGE_URL_MAPPING:
            similarities[IMAGE_URL_MAPPING[filename]] = round(float(similarity), 2)
        else:
            # If no mapping exists, use the original filename
            similarities[filename] = round(float(similarity), 2)
    
    # Sort similarities in descending order and take top 5
    sorted_similarities = dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5])
    
    return sorted_similarities

# Add a health check endpoint
@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "Design Similarity API is running"}
