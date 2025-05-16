import os, shutil
from sklearn.model_selection import train_test_split
import yaml

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
AUG_IMG_TRAIN   = os.path.join(BASE_DIR, "logo identification", "augmented", "images", "train")
AUG_LABEL_TRAIN = os.path.join(BASE_DIR, "logo identification", "augmented", "labels", "train")
RAW_IMG_VAL     = os.path.join(BASE_DIR, "logo identification", "images", "val")
RAW_LABEL_VAL   = os.path.join(BASE_DIR, "logo identification", "labels", "val")


# 1) ensure raw-val dirs exist, then split-out 20% from augmented train
os.makedirs(RAW_IMG_VAL,   exist_ok=True)
os.makedirs(RAW_LABEL_VAL, exist_ok=True)
imgs = [f for f in os.listdir(AUG_IMG_TRAIN) if f.lower().endswith((".jpg",".png"))]
_, val_imgs = train_test_split(imgs, test_size=0.2, random_state=42)
for fn in val_imgs:
    lb = os.path.splitext(fn)[0] + ".txt"
    shutil.move(os.path.join(AUG_IMG_TRAIN, fn),   os.path.join(RAW_IMG_VAL, fn))
    shutil.move(os.path.join(AUG_LABEL_TRAIN, lb), os.path.join(RAW_LABEL_VAL, lb))

# 2) assemble final dataset/
DST = os.path.join(BASE_DIR, "dataset")
IMG_DIR   = os.path.join(DST, "images")
LABEL_DIR = os.path.join(DST, "labels")
for split in ("train","val"):
    os.makedirs(os.path.join(IMG_DIR,   split), exist_ok=True)
    os.makedirs(os.path.join(LABEL_DIR, split), exist_ok=True)

# copy train
for fname in os.listdir(AUG_IMG_TRAIN):
    shutil.copy(os.path.join(AUG_IMG_TRAIN, fname),
                os.path.join(IMG_DIR, "train", fname))
for fname in os.listdir(AUG_LABEL_TRAIN):
    shutil.copy(os.path.join(AUG_LABEL_TRAIN, fname),
                os.path.join(LABEL_DIR, "train", fname))

# copy val
for fname in os.listdir(RAW_IMG_VAL):
    shutil.copy(os.path.join(RAW_IMG_VAL, fname),
                os.path.join(IMG_DIR, "val", fname))
for fname in os.listdir(RAW_LABEL_VAL):
    shutil.copy(os.path.join(RAW_LABEL_VAL, fname),
                os.path.join(LABEL_DIR, "val", fname))

# 3) write data.yaml
data = {"train":"images/train", "val":"images/val", "nc":1, "names":["logo"]}
with open(os.path.join(DST, "data.yaml"), "w") as f:
    yaml.dump(data, f, sort_keys=False)

print("✅ Done. `dataset/` created with `data.yaml`.")