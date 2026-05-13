import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tarfile
import urllib.request
import shutil

import matplotlib.pyplot as plt
import numpy as np
import keras
import tensorflow as tf
import random
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Reproducibility ────────────────────────────────────────────────────────────
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# ── Hyperparameters ────────────────────────────────────────────────────────────
batch_size  = 32
n_epochs    = 10
img_rows, img_cols = 128, 128
input_shape = (img_rows, img_cols, 3)

# ── Download & extract dataset ─────────────────────────────────────────────────
url          = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar"
tar_filename = "aircraft-damage-dataset-v1.tar"
extracted_dir = "aircraft-damage-dataset-v1"

urllib.request.urlretrieve(url, tar_filename)
print(f"Downloaded {tar_filename}. Extracting...")

if os.path.exists(extracted_dir):
    shutil.rmtree(extracted_dir)
    print(f"Removed existing folder: {extracted_dir}")

with tarfile.open(tar_filename, "r") as tar_ref:
    tar_ref.extractall()
    print(f"Extracted {tar_filename} successfully")

# ── Directories ────────────────────────────────────────────────────────────────
extract_path = "aircraft_damage_dataset_v1"
train_dir = os.path.join(extract_path, 'train')
test_dir  = os.path.join(extract_path, 'test')
valid_dir = os.path.join(extract_path, 'valid')

# ── Data generators ────────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20
)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    class_mode='binary',
    seed=seed_value,
    batch_size=batch_size,
    shuffle=True,
    target_size=(img_rows, img_cols)
)

valid_generator = valid_datagen.flow_from_directory(
    directory=valid_dir,
    class_mode='binary',
    seed=seed_value,
    batch_size=batch_size,
    shuffle=False,
    target_size=(img_rows, img_cols)
)

test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    class_mode='binary',
    seed=seed_value,
    batch_size=batch_size,
    shuffle=False,
    target_size=(img_rows, img_cols)
)

# ── Build model (VGG16 + custom head) ─────────────────────────────────────────
vgg = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

output          = vgg.layers[-1].output
output_flatten  = keras.layers.Flatten()(output)
base_model      = Model(vgg.input, output_flatten)

for layer in base_model.layers:
    layer.trainable = False

model = Sequential([
    base_model,
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── Train ──────────────────────────────────────────────────────────────────────
history = model.fit(
    train_generator,
    epochs=n_epochs,
    validation_data=valid_generator
)

train_history = history.history

# ── Plot training curves ───────────────────────────────────────────────────────
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_history['loss'],     label='Train Loss')
plt.plot(train_history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_history['accuracy'],     label='Train Accuracy')
plt.plot(train_history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# ── Evaluate on test set ───────────────────────────────────────────────────────
test_loss, test_accuracy = model.evaluate(
    test_generator,
    steps=test_generator.samples // test_generator.batch_size
)
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# ── Visualise a prediction ─────────────────────────────────────────────────────
def plot_image_with_prediction(test_generator, model, index=0):
    images, labels = next(test_generator)
    predictions     = model.predict(images)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    class_names     = {v: k for k, v in test_generator.class_indices.items()}

    plt.figure(figsize=(5, 5))
    plt.imshow(images[index])
    plt.title(
        f"True: {class_names[int(labels[index])]}\n"
        f"Predicted: {class_names[predicted_classes[index]]}"
    )
    plt.axis('off')
    plt.show()

plot_image_with_prediction(test_generator, model, index=0)

# ── BLIP: load processor and model ────────────────────────────────────────────
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# ── Custom Keras layer: image captioning / summary ────────────────────────────
class BlipCaptionSummaryLayer(tf.keras.layers.Layer):
    def __init__(self, processor, model, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
        self.blip      = model

    def call(self, image_path, task):
        return tf.py_function(self.generate_text, [image_path, task], tf.string)

    def generate_text(self, image_path, task):
        try:
            image_path_str = image_path.numpy().decode("utf-8")
            img            = Image.open(image_path_str).convert("RGB")

            if task.numpy().decode("utf-8") == "caption":
                prompt = "This is a picture of"
            else:
                prompt = "This is a detailed photo showing"

            inputs = self.processor(images=img, text=prompt, return_tensors="pt")
            output = self.blip.generate(**inputs)
            result = self.processor.decode(output[0], skip_special_tokens=True)
            return result

        except Exception as e:
            print(f"Error: {e}")
            return "Error processing image"

# ── Run BLIP on a sample image ────────────────────────────────────────────────
blip_layer = BlipCaptionSummaryLayer(blip_processor, blip_model)

image_path = tf.constant(
    "aircraft_damage_dataset_v1/test/dent/144_10_JPG_jpg.rf.4d008cc33e217c1606b76585469d626b.jpg"
)

caption = blip_layer(image_path, tf.constant("caption"))
print("Caption:", caption.numpy().decode("utf-8"))

summary = blip_layer(image_path, tf.constant("summary"))
print("Summary:", summary.numpy().decode("utf-8"))

# ── Display a second sample image ─────────────────────────────────────────────
image_url = "aircraft_damage_dataset_v1/test/dent/149_22_JPG_jpg.rf.4899cbb6f4aad9588fa3811bb886c34d.jpg"
img = plt.imread(image_url)
plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.axis('off')
plt.title("Sample Test Image")
plt.show()