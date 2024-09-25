import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
import cv2
import time
import requests
from datetime import datetime

train_data_dir = '/dataset'

# Preprocessing and augmenting the images
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [128, 128])
    image = image / 255.0  # Rescale
    return image

def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.1)
    return image

def load_dataset(data_dir):
    dataset = tf.data.Dataset.list_files(os.path.join(data_dir, '*/*'))
    dataset = dataset.map(lambda x: preprocess_image(x))
    dataset = dataset.map(lambda x: augment_image(x))
    dataset = dataset.repeat()  # Repeat dataset for more training data
    dataset = dataset.batch(1)  # Use smaller batch size
    return dataset

train_dataset = load_dataset(train_data_dir)

# Autoencoder model
class Autoencoder(tf.keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(128, 128, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        ])
        
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

autoencoder = Autoencoder()

# Compile the autoencoder model
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')

# Add early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# Training autoencoder using model.fit
autoencoder.fit(train_dataset, epochs=100, callbacks=[early_stopping], steps_per_epoch=5)

# Computes reconstruction loss
def detect_anomaly(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0)  # Batch dimension
    img = img / 255.0  # Rescale

    # Reconstructing image using autoencoder
    reconstructed = autoencoder(tf.convert_to_tensor(img, dtype=tf.float32))
    
    # MSE between original and reconstructed
    mse = np.mean(np.square(img - reconstructed.numpy()))

    # Threshold for anomaly detection
    threshold = 0.02
    
    if mse > threshold:
        return "Unhealthy crops in picture"
    else:
        return "Healthy crops in picture"
    
# URL to fetch the image
url = "http://192.168.11.1:8080/snapshot?topic=/main_camera/image_raw"

# Function to fetch and save the image
def fetch_image():
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}.jpg"
            with open(filename, "wb") as file:
                file.write(response.content)
            print(f"Image fetched and saved as {filename}.")
        else:
            print(f"Failed to fetch image. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

# Fetch image every 10 seconds and detect anomalies
while True:
    fetch_image()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"snapshot_{timestamp}.jpg"
    
    # Detect anomalies in fetched image
    result = detect_anomaly(filename)
    print(result)
    
    time.sleep(10)
