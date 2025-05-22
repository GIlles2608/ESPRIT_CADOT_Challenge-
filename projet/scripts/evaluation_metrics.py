# -*- coding: utf-8 -*-
"""
Created on Sun May 11 22:51:38 2025
@author: Gilles2608
"""

from ultralytics import YOLO
import pandas as pd


def main():
    # Charger le modèle entraîné
    model = YOLO("runs/detect/train25/weights/best.pt")

    # Évaluation sur le jeu de validation
    metrics = model.val(
        data="../models/config.yaml",  # Pointe vers val_data dans config.yaml
        split="val",            # Forcer l'évaluation sur le jeu de validation
        save_json=True          # Génère un fichier JSON avec les métriques
    )

    # Exporter les métriques en CSV
    pd.DataFrame([metrics.results_dict]).to_csv("../results/metrics.csv", index=False)
    

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Optionnel mais recommandé sur Windows
    main()
