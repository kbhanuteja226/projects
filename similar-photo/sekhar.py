import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load pre-trained ResNet50 model
model = resnet50(pretrained=True)
model.eval()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path):
    """Extract features from an image using ResNet50."""
    img = Image.open(image_path)
    img = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        features = model(img).squeeze().numpy()
    return features

def find_similar_images(query_features, image_features):
    """Find images similar to the query image."""
    similarities = cosine_similarity([query_features], image_features)
    return similarities.argsort()[0][::-1]

# Example usage
image_dir = 'images'
image_files = os.listdir(image_dir)
image_features = []

# Extract features for all images in the directory
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    features = extract_features(image_path)
    image_features.append(features)

# Query with a specific image
query_image_path = 'MInimalist-landscape-wallpaper-night-moon-desktop-4k.png'
query_features = extract_features(query_image_path)

# Find similar images
similar_indices = find_similar_images(query_features, image_features)
similar_images = [image_files[i] for i in similar_indices]

print("Top similar images:", similar_images)