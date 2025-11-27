import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import regularizers

# ================== CONFIG ==================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50

# ================== PATHS ==================
TRAIN_DIR = "./train"
VALID_DIR = "./valid"

AUTOTUNE = tf.data.AUTOTUNE

# ================== AUGMENTATION ==================
def augment_image(img, label):
    """
    Apply ALL required augmentations:
    - rotation
    - flipping
    - zoom (via random crop)
    - brightness
    - cropping (covered by zoom)
    """
    # 1. FLIP
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    # 2. ROTATION (0°, 90°, 180°, 270°)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    img = tf.image.rot90(img, k)

    # 3 & 5. ZOOM + CROPPING (random crop then resize)
    zoom_factor = tf.random.uniform([], 0.85, 1.0)   # slightly milder zoom
    crop_h = tf.cast(zoom_factor * IMG_SIZE[0], tf.int32)
    crop_w = tf.cast(zoom_factor * IMG_SIZE[1], tf.int32)
    img = tf.image.random_crop(img, size=[crop_h, crop_w, 3])
    img = tf.image.resize(img, IMG_SIZE)

    # 4. BRIGHTNESS
    img = tf.image.random_brightness(img, max_delta=0.08)

    return img, label


def normalize(img, label):
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


# ================== DATASETS ==================
print("📥 Loading training data from:", TRAIN_DIR)
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",     # 0.0 / 1.0
    image_size=IMG_SIZE,
    batch_size=None,         # unbatched for custom map
    shuffle=True,
)

print("📥 Loading validation data from:", VALID_DIR)
valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=None,         # unbatched
    shuffle=False,
)

# Normalize
train_ds = train_ds.map(normalize, num_parallel_calls=AUTOTUNE)
valid_ds = valid_ds.map(normalize, num_parallel_calls=AUTOTUNE)

# Augment only training data
train_ds = train_ds.map(augment_image, num_parallel_calls=AUTOTUNE)

# Batch + prefetch
train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
valid_ds = valid_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# ================== CUSTOM CNN MODEL ==================
weight_decay = 1e-4  # small L2 regularizer

model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), padding="same", activation="relu",
           kernel_regularizer=regularizers.l2(weight_decay),
           input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    # Block 2
    Conv2D(64, (3, 3), padding="same", activation="relu",
           kernel_regularizer=regularizers.l2(weight_decay)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    # Block 3
    Conv2D(128, (3, 3), padding="same", activation="relu",
           kernel_regularizer=regularizers.l2(weight_decay)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    # Block 4 (extra depth)
    Conv2D(256, (3, 3), padding="same", activation="relu",
           kernel_regularizer=regularizers.l2(weight_decay)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),

    Dense(256, activation="relu",
          kernel_regularizer=regularizers.l2(weight_decay)),
    BatchNormalization(),
    Dropout(0.5),

    Dense(1, activation="sigmoid")  # binary: Bird vs Drone
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ================== CALLBACKS ==================
callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=7,
        restore_best_weights=True,
        verbose=1,
    ),
    ModelCheckpoint(
        "best_custom_cnn.keras",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    ),
]

# ================== TRAIN ==================
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

model.save("final_custom_cnn_model.keras")
print("✅ Training complete. Models saved as best_custom_cnn.keras and final_custom_cnn_model.keras")
