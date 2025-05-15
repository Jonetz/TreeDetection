import itertools
import os
import subprocess

# Hyperparameter-Werte
confidence_thresholds = [0.3, 0.4, 0.5]
containment_thresholds = [0.6, 0.75, 0.9]
iou_thresholds = [0.6, 0.4, 0.8]
area_thresholds = [1]
ndvi_mean_thresholds = [0.05, 0.15, 0.1, 0.2]
ndvi_var_thresholds = [0.1, 0.05, 0.15]

# Alle Kombinationen der Hyperparameter durchgehen
for i, (conf, ndvi_var, area, cont, iou, ndvi_mean) in enumerate(
    itertools.product(
        confidence_thresholds,
        ndvi_var_thresholds,
        area_thresholds,
        containment_thresholds,
        iou_thresholds,
        ndvi_mean_thresholds
    )
):
    if i < 2:
        continue
    # Erstelle ein eindeutiges Output-Verzeichnis für diese Kombination
    output_dir = f"output_run_{i}_conf{conf}_cont{cont}_iou{iou}_ndvi{ndvi_mean}_{ndvi_var}"
    os.makedirs(output_dir, exist_ok=True)
    
    # copy every file from output/geojson_predictions to output_dir that does not start with processed
    
    cmd = [
        "cp", "-r", "output/geojson_predictions", output_dir
    ]
    
    subprocess.run(cmd)  # Führt das Skript als separaten Prozess aus
    

    # Starte ein separates Skript mit den aktuellen Parametern
    cmd = [
        "python", "main.py",
        "--conf", str(conf),
        "--cont", str(cont),
        "--iou", str(iou),
        "--area", str(area),
        "--ndvi_mean", str(ndvi_mean),
        "--ndvi_var", str(ndvi_var),
        "--output", output_dir
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)  # Führt das Skript als separaten Prozess aus
    
    """ 
    Add this to main.py


    1.)
    import argparse
    import gc
    import torch
    from config import Config
    # Argumente parsen
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=float)
    parser.add_argument("--cont", type=float)
    parser.add_argument("--iou", type=float)
    parser.add_argument("--area", type=int)
    parser.add_argument("--ndvi_mean", type=float)
    parser.add_argument("--ndvi_var", type=float)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    # Konfiguration laden
    config, config_cls = get_config("config.yml")

    # Setze die Hyperparameter in die Config
    config["confidence_threshold"] = args.conf
    config["containment_threshold"] = args.cont
    config["iou_threshold"] = args.iou
    config["area_threshold"] = args.area
    config["ndvi_mean_threshold"] = args.ndvi_mean
    config["ndvi_var_threshold"] = args.ndvi_var
    config["output_directory"] = args.output
    
    # Erstelle das Config-Objekt
    config_obj = Config()
    config_obj._load_into_config(config)

    print(f"Running with config: {config}")
    process_files(config)

    """
