import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import requests
from datetime import datetime

# Define the path to the dataset
train_data_dir = 'C://Users//brand//OneDrive//Documents//Drone IMG Recognition//datasetnew'

# Preprocessing and augmenting the images
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [128, 128])
    image = image / 255.0  # Rescale pixel values to [0, 1]
    return image

def load_simple_dataset(data_dir):
    # Load images directly from the root directory, assuming .png format
    dataset = tf.data.Dataset.list_files(os.path.join(data_dir, '*.png'))
    dataset = dataset.map(lambda x: preprocess_image(x))  # Preprocess images (resize, rescale)
    dataset = dataset.batch(1)  # Batch size of 1 since there are few images
    return dataset

# Load dataset
train_dataset = load_simple_dataset(train_data_dir)

# Autoencoder model using tf.keras.Model
class Autoencoder(tf.keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder: downsampling
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(128, 128, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        ])
        
        # Decoder: upsampling
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Instantiate the autoencoder model
autoencoder = Autoencoder()

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')

# Train the model
autoencoder.fit(train_dataset, epochs=40, steps_per_epoch=5)
# Anomaly detection function based on reconstruction error
def detect_anomaly(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Rescale pixel values to [0, 1]

    # Reconstruct the image using the autoencoder
    reconstructed = autoencoder.predict(tf.convert_to_tensor(img, dtype=tf.float32))
    
    # Calculate MSE between original and reconstructed image
    mse = np.mean(np.square(img - reconstructed))

    # Threshold for anomaly detection (tune this value based on your results)
    threshold = 0.02
    
    if mse > threshold:
        return "Unhealthy crops in picture"
    else:
        return "Healthy crops in picture"

# URL to fetch the image from a live stream or camera feed
url = "http://192.168.11.1:8080/snapshot?topic=/main_camera/image_raw"

# Function to fetch and save the image
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
        else:
            print(f"Failed to fetch image. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

# Fetch image every 10 seconds and check for anomalies
for i in range (0,5):
    time.sleep(7)
    fetch_image()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"snapshot_{timestamp}.jpg"
    
    # Detect anomalies in fetched image
    result = detect_anomaly(filename)
    print(result)
