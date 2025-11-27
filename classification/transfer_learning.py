import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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
    - zoom (crop + resize)
    - brightness
    - cropping
    """
    # Flips
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    # Rotation (0°, 90°, 180°, 270°)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    img = tf.image.rot90(img, k)

    # Zoom + Crop
    zoom_factor = tf.random.uniform([], 0.8, 1.0)
    crop_h = tf.cast(zoom_factor * IMG_SIZE[0], tf.int32)
    crop_w = tf.cast(zoom_factor * IMG_SIZE[1], tf.int32)
    img = tf.image.random_crop(img, size=[crop_h, crop_w, 3])
    img = tf.image.resize(img, IMG_SIZE)

    # Brightness
    img = tf.image.random_brightness(img, max_delta=0.1)

    return img, label

def normalize(img, label):
    img = tf.cast(img, tf.float32) / 255.0   # [0,1]
    return img, label

# ================== LOAD DATASET ==================
print("📥 Loading TRAIN dataset...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=None,   # unbatched for augment_image
    shuffle=True
)

print("📥 Loading VALID dataset...")
valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=None,
    shuffle=False
)

# Normalize
train_ds = train_ds.map(normalize, num_parallel_calls=AUTOTUNE)
valid_ds = valid_ds.map(normalize, num_parallel_calls=AUTOTUNE)

# Augment only training data
train_ds = train_ds.map(augment_image, num_parallel_calls=AUTOTUNE)

# Batch + prefetch
train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
valid_ds = valid_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# ================== BUILD TRANSFER LEARNING MODEL (MobileNetV2) ==================
print("🔧 Loading MobileNetV2 with ImageNet weights...")

base_model = MobileNetV2(
    weights="imagenet",              # TRUE transfer learning
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Freeze base model first
base_model.trainable = False

inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
outputs = Dense(1, activation="sigmoid")(x)   # Bird vs Drone

model = Model(inputs, outputs, name="MobileNetV2_TransferLearning")

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ================== CALLBACKS ==================
callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        mode="max",
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        "best_transfer_learning.keras",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
]

# ================== TRAIN: FEATURE EXTRACTION ==================
print("🚀 Step 1: Training classification head (frozen backbone)...")
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ================== FINE-TUNING: UNFREEZE TOP LAYERS ==================
print("🎯 Step 2: Fine-tuning top layers of MobileNetV2...")

base_model.trainable = True

# Unfreeze last ~30 layers for fine-tuning
fine_tune_at = len(base_model.layers) - 30
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # small LR for fine-tuning
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=10,
    callbacks=callbacks
)

model.save("final_transfer_learning_model.keras")
print("✅ Training complete! Saved as best_transfer_learning.keras and final_transfer_learning_model.keras")
