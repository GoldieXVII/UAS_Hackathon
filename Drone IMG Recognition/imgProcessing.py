import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import requests
import time
from datetime import datetime
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# URL to fetch the image from a live stream or camera feed
url = "http://192.168.11.1:8080/snapshot?topic=/main_camera/image_raw"

# Load pre-trained VGG16 model + higher level layers
model = VGG16(weights='imagenet', include_top=False)

reference_images_dir = "C://Users//brand//OneDrive//Documents//Drone IMG Recognition/datasetnew"

# Define similarity threshold for healthy crops
SIMILARITY_THRESHOLD = 0.85  # Adjust this based on experimentation

# Function to preprocess and extract features from images
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))  # VGG16 expects 224x224
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    # Get features from the VGG16 model
    features = model.predict(img_data)
    return features.flatten()

# Extract features for all reference images
reference_features = []
reference_images = []

for img_name in os.listdir(reference_images_dir):
    img_path = os.path.join(reference_images_dir, img_name)
    features = extract_features(img_path, model)
    reference_features.append(features)
    reference_images.append(img_name)

# Function to compare an incoming image to reference images
def compare_image(incoming_image_path):
    incoming_features = extract_features(incoming_image_path, model)

    similarities = []
    for ref_features in reference_features:
        similarity = cosine_similarity([incoming_features], [ref_features])[0][0]
        similarities.append(similarity)

    # Get the most similar reference image
    most_similar_idx = np.argmax(similarities)
    most_similar_image = reference_images[most_similar_idx]
    highest_similarity = similarities[most_similar_idx]

    print(f"Most similar image: {most_similar_image} with similarity: {highest_similarity}")
    return highest_similarity

# Function to determine crop health based on similarity score
def classify_crop(similarity_score):
    if similarity_score >= SIMILARITY_THRESHOLD:
        return "Healthy Crop"
    else:
        return "Unhealthy Crop"

# Function to fetch and save the image from live stream
def fetch_image():
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            # Generate a unique filename based on the current timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}.jpg"
            with open(filename, "wb") as file:
                file.write(response.content)
            print(f"Image fetched and saved as {filename}.")
            return filename
        else:
            print(f"Failed to fetch image. Status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None
time.sleep(30)
# Fetch image every 7 seconds and classify crop health
for i in range(5):
    time.sleep(7)  # Adjust the delay based on your requirements
    filename = fetch_image()
    # filename = 'snapshot_20240926_160248.jpg'
    
    if filename:
        # Compare the fetched image to reference images
        similarity_score = compare_image(filename)
        
        # Classify crop health based on similarity score
        crop_health = classify_crop(similarity_score)
        print(f"Crop health classification: {crop_health}")
