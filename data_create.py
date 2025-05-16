import cv2
import numpy as np
import os
import random
from glob import glob
import sys

# === PATHS ===
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
LOGO_DIR       = os.path.join(BASE_DIR, "logo identification","Zoom")
BG_DIR         = os.path.join(BASE_DIR, "logo identification","backgrounds")
SAVE_IMG_DIR   = os.path.join(BASE_DIR, "logo identification","augmented", "images", "train")
SAVE_LABEL_DIR = os.path.join(BASE_DIR, "logo identification","augmented", "labels", "train")


# === CREATE OUTPUT FOLDERS IF NOT EXISTS ===
os.makedirs(SAVE_IMG_DIR, exist_ok=True)
os.makedirs(SAVE_LABEL_DIR, exist_ok=True)

# === AUGMENTATION FUNCTIONS ===
def random_rotate(img):
    angle = random.uniform(-30, 30)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return rotated

def random_scale(img):
    scale = random.uniform(0.5, 1.2)
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)))

def add_noise(img):
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    return cv2.add(img, noise)

def random_blur(img):
    k = random.choice([3, 5])
    return cv2.GaussianBlur(img, (k, k), 0)

def apply_color_jitter(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[..., 1] *= random.uniform(0.7, 1.3)
    hsv[..., 2] *= random.uniform(0.7, 1.3)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# === LOAD LOGO AND BACKGROUND IMAGES ===
logo_paths = glob(os.path.join(LOGO_DIR, "*.png")) + glob(os.path.join(LOGO_DIR, "*.jpg"))
backgrounds = glob(os.path.join(BG_DIR, "*.jpg")) + glob(os.path.join(BG_DIR, "*.png"))

if not backgrounds:
    raise RuntimeError("❌ No background images found. Please check BG_DIR.")

img_count = 0

# === MAIN LOOP ===
for logo_path in logo_paths:
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    if logo is None:
        print(f"⚠️ Could not load {logo_path}")
        continue

    for i in range(20):  # 20 augmentations per logo
        # Ensure 4-channel logo
        if logo.shape[2] == 3:
            logo = cv2.cvtColor(logo, cv2.COLOR_BGR2BGRA)

        aug = random_rotate(logo)
        aug = random_scale(aug)

        # Apply color jitter on BGR channels, not alpha
        color_jittered = apply_color_jitter(aug[..., :3])
        aug[..., :3] = color_jittered
        aug = add_noise(aug)
        aug = random_blur(aug)

        # Make sure it's still 4 channels
        if aug.shape[2] == 3:
            aug = cv2.cvtColor(aug, cv2.COLOR_BGR2BGRA)

        # Select a random background
        bg_path = random.choice(backgrounds)
        bg = cv2.imread(bg_path)

        if bg is None:
            continue

        # Resize background if too small
        bg_h, bg_w = bg.shape[:2]
        if bg_h < 512 or bg_w < 512:
            bg = cv2.resize(bg, (max(512, bg_w), max(512, bg_h)))
            bg_h, bg_w = bg.shape[:2]

        # Skip if logo is larger than background
        if aug.shape[0] >= bg_h or aug.shape[1] >= bg_w:
            continue

        # Random position to paste logo
        x_offset = random.randint(0, bg_w - aug.shape[1])
        y_offset = random.randint(0, bg_h - aug.shape[0])

        # Create composite image
        overlay = bg.copy()
        alpha = aug[..., 3] / 255.0

        for c in range(3):
            overlay[y_offset:y_offset + aug.shape[0], x_offset:x_offset + aug.shape[1], c] = (
                alpha * aug[..., c] +
                (1 - alpha) * overlay[y_offset:y_offset + aug.shape[0], x_offset:x_offset + aug.shape[1], c]
            )

        # Save image
        img_filename = f"aug_{img_count:04d}.jpg"
        label_filename = f"aug_{img_count:04d}.txt"
        img_path = os.path.join(SAVE_IMG_DIR, img_filename)
        label_path = os.path.join(SAVE_LABEL_DIR, label_filename)

        cv2.imwrite(img_path, overlay)

        # Save YOLO label (class_id x_center y_center width height)
        cx = (x_offset + aug.shape[1] / 2) / bg_w
        cy = (y_offset + aug.shape[0] / 2) / bg_h
        w = aug.shape[1] / bg_w
        h = aug.shape[0] / bg_h

        with open(label_path, 'w') as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        img_count += 1

print(f"✅ Finished generating {img_count} images and labels.")