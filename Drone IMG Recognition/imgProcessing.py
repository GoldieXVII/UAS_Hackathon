import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

train_data_dir = '/dataset'

# Preprocessing and augmenting the images
def preprocess_image(image):
    image = tf.image.resize(image, [128, 128])
    image = image / 255.0  # Rescale
    return image

def load_dataset(data_dir):
    dataset = tf.data.Dataset.list_files(os.path.join(data_dir, '*/*'))
    dataset = dataset.map(lambda x: preprocess_image(tf.io.read_file(x)))
    dataset = dataset.batch(32)
    return dataset

train_dataset = load_dataset(train_data_dir)

# Autoencoder model
class Autoencoder(tf.Module):
    def __init__(self):
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

    def __call__(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = Autoencoder()

# Loss and optimizer
loss_object = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Training step
@tf.function
def train_step(images):
    with tf.GradientTape() as tape:
        reconstructed = autoencoder(images)
        loss = loss_object(images, reconstructed)
    gradients = tape.gradient(loss, autoencoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))
    return loss

# Training loop
def train_autoencoder(dataset, epochs):
    for epoch in range(epochs):
        for batch in dataset:
            loss = train_step(batch)
        print(f'Epoch {epoch+1}, Loss: {loss.numpy()}')

train_autoencoder(train_dataset, epochs=10)

# Computes reconstruction loss (anomaly detection)
def detect_anomaly(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Rescale as the model expects

    # Reconstruct the image using autoencoder
    reconstructed = autoencoder(tf.convert_to_tensor(img, dtype=tf.float32))
    
    # MSE between original and reconstructed
    mse = np.mean(np.square(img - reconstructed.numpy()))

    # Threshold for anomaly detection
    threshold = 0.02
    
    if mse > threshold:
        return "Unhealthy crops in picture"
    else:
        return "Healthy crops in picture"

# image_path = 'image from raspberry pi 4' #Still need to integrate this
# result = detect_anomaly(image_path)
# print(f"The crop is: {result}")
