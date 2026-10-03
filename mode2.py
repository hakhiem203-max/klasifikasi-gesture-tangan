import tensorflow as tf
import matplotlib.pyplot as plt
import os

# =========================
# 1. Pengaturan
# =========================

DATASET = "dataset_raw"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42

# =========================
# 2. Membaca dataset
# =========================

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names

print("\nKelas:", class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# =========================
# 3. MobileNetV2
# =========================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Awalnya semua layer dibekukan
base_model.trainable = False

# Buka 20 layer terakhir untuk fine-tuning
for layer in base_model.layers[-20:]:
    layer.trainable = True

# =========================
# 4. Membuat model
# =========================

inputs = tf.keras.Input(shape=(224, 224, 3))

x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.2)(x)

outputs = tf.keras.layers.Dense(
    3,
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

# =========================
# 5. Compile
# =========================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# 6. Training
# =========================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# =========================
# 7. Simpan model
# =========================

os.makedirs("models", exist_ok=True)

model.save("models/mode2_mobilenetv2.keras")

# =========================
# 8. Grafik
# =========================

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Mode 2 - MobileNetV2 Fine-Tuning")
plt.legend()
plt.grid()

plt.savefig(
    "results/accuracy_mode2.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\n==============================")
print("TRAINING MODE 2 SELESAI")
print("==============================")
print("Model : models/mode2_mobilenetv2.keras")
print("Grafik: results/accuracy_mode2.png")