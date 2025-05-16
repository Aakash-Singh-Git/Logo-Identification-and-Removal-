import yaml
from ultralytics import YOLO

# ─── paths ─────────────────────────────────────────────────────────────────────
DATA_YAML    = r"C:\Users\Aakash\Downloads\logo identification\dataset\data.yaml"
BASE_MODEL   = "yolov8n.pt"           # you can also try yolov8s.pt, yolov8m.pt, etc.
TUNE_DIR     = "runs/tune/logo"       # where tuning logs & best_hyperparameters.yaml will go
FINAL_PROJECT= "runs/train"
FINAL_NAME   = "logo_final"

# ─── 1) initialize model ───────────────────────────────────────────────────────
model = YOLO(BASE_MODEL)

# ─── 2) hyperparameter tuning (“evolution”) ────────────────────────────────────
#    - epochs=5 per trial (short)
#    - iterations=20 different hyperparameter sets
#    - results (best_hyperparameters.yaml) written to TUNE_DIR
model.tune(
    data=DATA_YAML,
    epochs=5,
    iterations=20,
    project=TUNE_DIR,
    name="evolve",
    save=False,    # don’t save every trial’s weights (to save disk)
    plots=False    # disable intermediate plots
)
# :contentReference[oaicite:0]{index=0}

# ─── 3) load best hyperparameters ──────────────────────────────────────────────
best_hyp_file = f"{TUNE_DIR}/evolve/best_hyperparameters.yaml"
with open(best_hyp_file) as f:
    best_hyp = yaml.safe_load(f)

print("Best hyperparameters found:")
for k,v in best_hyp.items():
    print(f"  {k}: {v}")

# ─── 4) final training with best hyperparameters ───────────────────────────────
#    - epochs=80 on the full dataset
#    - best_hyp contains keys like lr0, momentum, weight_decay, etc.
model.train(
    data=DATA_YAML,
    epochs=80,
    project=FINAL_PROJECT,
    name=FINAL_NAME,
    **best_hyp
)

print(f"✅ Final model saved to {FINAL_PROJECT}/{FINAL_NAME}/weights/best.pt")