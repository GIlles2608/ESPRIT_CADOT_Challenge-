# -*- coding: utf-8 -*-

from ultralytics import YOLO
import os
import yaml

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # ← évite le bug d'OpenMP sur Windows

def main():
    # Charger la config
    with open("../models/config.yaml") as f:
        config = yaml.safe_load(f)

    # Entraînement
    model = YOLO("yolo11n.pt")
    results = model.train(
        data="../config.yaml",               # <- Important : fichier YAML
        epochs=config["epochs"],
        imgsz=config["img_size"],
        batch=config["batch_size"],
        patience=10,
        save=True,
        save_period=5,
        device="cuda"                        # Active la RTX 3050
    )

    

if __name__ == "__main__":
    main()
