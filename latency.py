import tensorflow as tf
import numpy as np
import time

# Model yang akan diukur
model_path = "models/mode2_mobilenetv2.keras"

# Load model
model = tf.keras.models.load_model(model_path)

# Buat data gambar dummy
input_data = np.random.rand(1, 224, 224, 3).astype(np.float32)

# Warm-up
for _ in range(5):
    model.predict(input_data, verbose=0)

# Pengukuran latency
jumlah_pengujian = 20
waktu_total = 0

for _ in range(jumlah_pengujian):

    mulai = time.perf_counter()

    model.predict(input_data, verbose=0)

    selesai = time.perf_counter()

    waktu_total += selesai - mulai

latency_rata_rata = (waktu_total / jumlah_pengujian) * 1000

print("\n===== HASIL LATENCY =====")
print("Model :", model_path)
print("Jumlah pengujian :", jumlah_pengujian)
print("Latency rata-rata : {:.2f} ms".format(latency_rata_rata))