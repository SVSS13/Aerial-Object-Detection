import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from ultralytics import YOLO
import torch


# ================== CONFIG ==================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

BASE_DIR = Path(__file__).resolve().parent

# Keras models (in classification folder)
CUSTOM_CNN_PATH = BASE_DIR / "best_custom_cnn.keras"
TRANSFER_LEARNING_PATH = BASE_DIR / "best_transfer_learning.keras"

# YOLO model (adjust if needed)
YOLO_MODEL_PATH = (
    BASE_DIR.parent
    / "streamlit_app"
    / "models"
    / "Checkpoint"
    / "YoloV8_Detection"
    / "best.pt"
)

TEST_DIR = BASE_DIR / "test"   # ./test/bird, ./test/drone
CLASS_NAMES = ["bird", "drone"]   # 0 -> bird, 1 -> drone


# ================== LOAD TEST IMAGE PATHS + LABELS ==================
def load_test_paths_and_labels(test_dir: Path):
    image_paths = []
    labels = []

    # Ensure classes are in sorted order: bird -> 0, drone -> 1
    class_dirs = [d for d in sorted(os.listdir(test_dir)) if (test_dir / d).is_dir()]

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_dirs)}
    print(f"Class mapping (folder -> label): {class_to_idx}")

    for cls_name in class_dirs:
        cls_idx = class_to_idx[cls_name]
        cls_folder = test_dir / cls_name
        for fname in sorted(os.listdir(cls_folder)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                fpath = cls_folder / fname
                image_paths.append(str(fpath))
                labels.append(cls_idx)

    return image_paths, np.array(labels, dtype=int)


print(f"📥 Scanning TEST directory: {TEST_DIR}")
image_paths, y_true = load_test_paths_and_labels(TEST_DIR)
print(f"Total test samples: {len(y_true)}")


# ================== TF DATASET FOR KERAS MODELS ==================
def build_tf_dataset(image_paths, labels):
    paths_tensor = tf.constant(image_paths)
    labels_tensor = tf.constant(labels, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths_tensor, labels_tensor))

    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


test_ds = build_tf_dataset(image_paths, y_true)


# ================== EVALUATE A KERAS MODEL ==================
def evaluate_keras_model(model_path: Path, model_name: str):
    print(f"\n==============================")
    print(f"🔧 Evaluating Keras model: {model_name}")
    print(f"Path: {model_path}")
    print(f"==============================")

    model = tf.keras.models.load_model(str(model_path))

    y_pred_probs = model.predict(test_ds)
    y_pred = (y_pred_probs.ravel() > 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n📊 Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    print("\n🧮 Confusion Matrix:\n")
    print(cm)

    return acc, cm


# ================== EVALUATE YOLO MODEL AS CLASSIFIER ==================
def load_yolo_model(yolo_path: Path):
    if not yolo_path.exists():
        print(f"❌ YOLO model not found at: {yolo_path}")
        return None

    print(f"\n==============================")
    print(f"🔧 Loading YOLOv8 model from: {yolo_path}")
    print(f"==============================")

    # Patch torch.load to force weights_only=False (PyTorch 2.6 change)
    real_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return real_load(*args, **kwargs)

    torch.load = patched_load

    model = YOLO(str(yolo_path))
    return model


def evaluate_yolo_model(yolo_model, image_paths, y_true):
    print(f"\n==============================")
    print(f"🧪 Evaluating YOLOv8 as a classifier")
    print(f"==============================")

    names = yolo_model.names  # dict: {class_id: name}
    print(f"YOLO class names: {names}")

    # Try to detect which id is bird/drone
    inv_names = {v.lower(): k for k, v in names.items()}
    bird_id = inv_names.get("bird", 0)
    drone_id = inv_names.get("drone", 1)

    print(f"Assuming: bird -> id {bird_id}, drone -> id {drone_id}")

    y_pred = []

    for path in image_paths:
        # Run YOLO on each image
        results = yolo_model(path, verbose=False)
        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            # No detection: default to 'bird' (0) or choose a rule you like
            pred_label = 0
        else:
            boxes = r.boxes
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)

            # Pick the highest confidence detection
            best_idx = int(np.argmax(confs))
            cls_id = cls_ids[best_idx]

            if cls_id == drone_id:
                pred_label = 1
            elif cls_id == bird_id:
                pred_label = 0
            else:
                # Unexpected class -> fallback; here assume bird
                pred_label = 0

        y_pred.append(pred_label)

    y_pred = np.array(y_pred, dtype=int)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n📊 Classification Report (YOLO-based):\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    print("\n🧮 Confusion Matrix (YOLO-based):\n")
    print(cm)

    return acc, cm


# ================== RUN ALL THREE EVALUATIONS ==================
acc_cnn, cm_cnn = evaluate_keras_model(CUSTOM_CNN_PATH, "Custom CNN")
acc_tl, cm_tl = evaluate_keras_model(TRANSFER_LEARNING_PATH, "Transfer Learning")

yolo_model = load_yolo_model(YOLO_MODEL_PATH)
if yolo_model is not None:
    acc_yolo, cm_yolo = evaluate_yolo_model(yolo_model, image_paths, y_true)
else:
    acc_yolo, cm_yolo = None, None


# ================== CREATE COMPARISON IMAGE ==================
print("\n🖼 Generating comparison image: model_comparison.png")

models = []
cms = []
accs = []

models.append("Custom CNN")
cms.append(cm_cnn)
accs.append(acc_cnn)

models.append("Transfer Learning")
cms.append(cm_tl)
accs.append(acc_tl)

if cm_yolo is not None:
    models.append("YOLOv8")
    cms.append(cm_yolo)
    accs.append(acc_yolo)

n_models = len(models)

fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))

if n_models == 1:
    axes = [axes]

for ax, cm, name, acc in zip(axes, cms, models, accs):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"{name}\nAccuracy: {acc*100:.2f}%")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

plt.tight_layout()
out_path = BASE_DIR / "model_comparison.png"
plt.savefig(out_path, dpi=300)
plt.close(fig)

print(f"✅ Saved comparison image at: {out_path}")
