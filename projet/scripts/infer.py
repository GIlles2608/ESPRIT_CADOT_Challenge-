# -*- coding: utf-8 -*-
"""
Created on Sun May 11 22:28:52 2025

@author: Gilles2608
"""

import os
import json
from ultralytics import YOLO

# Assure la compatibilité Windows avec multiprocessing
if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Évite les crashs OpenMP (Intel)

    # Charger le modèle
    model = YOLO("../models/model_weights.pt")

    # Prédictions
    results = model.predict(
    source="../../CADOT_Dataset/test",
    save=True,                      # Sauvegarde les images avec bounding boxes
    save_txt=True,                  # Sauvegarde les annotations au format YOLO
    save_conf=True,                 # Sauvegarde les scores de confiance
    project="../visual_examples",   # Dossier de sortie
    workers = 0  # ← IMPORTANT pour éviter les erreurs multiprocessing sous Windows
    )

    # Extraire les résultats dans un format JSON
    predictions = []
    for result in results:
        image_path = result.path
        image_name = os.path.basename(image_path)
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            predictions.append({
                "image": image_name,
                "class_id": int(cls),
                "confidence": round(conf, 4),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
            })

    # Sauvegarder dans predictions.json
    os.makedirs("../results", exist_ok=True)
    with open("../results/predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)

    print("✅ Prédictions sauvegardées dans ../results/predictions.json")