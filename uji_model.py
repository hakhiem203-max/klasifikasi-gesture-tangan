import tensorflow as tf
import numpy as np
import os

# Nama kelas
class_names = ["fist", "palm", "thumbs_up"]

# Model yang diuji
model_paths = [
    "models/mode1_mobilenetv2.keras",
    "models/mode2_mobilenetv2.keras",
    "models/mode3_mobilenetv2.keras"
]

# Gambar contoh untuk pengujian
gambar = "dataset_raw/fist/fist_001.jpg"

# Ukuran gambar
IMG_SIZE = (224, 224)

# Baca dan proses gambar
img = tf.keras.utils.load_img(gambar, target_size=IMG_SIZE)
img_array = tf.keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

print("\n===== HASIL PENGUJIAN MODEL =====")

for model_path in model_paths:

    print("\nModel:", model_path)

    model = tf.keras.models.load_model(model_path)

    prediction = model.predict(img_array, verbose=0)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    print("Prediksi :", predicted_class)
    print("Confidence : {:.2f}%".format(confidence))

print("\nPengujian selesai.")