from confluent_kafka import Producer
import pandas as pd
import json
import os
from pathlib import Path
import numpy as np
from scipy.stats import kurtosis, skew
from datetime import datetime
import csv
import time

# Configuración del productor
producer_config = {
    'bootstrap.servers': 'localhost:9092',
}
producer = Producer(producer_config)

# Crear carpetas Bronze
os.makedirs("datalake/bronze", exist_ok=True)

bronze_raw_data_file = open("datalake/bronze/raw_data.csv", "a", newline="")

writer_raw_data = None


PATH = Path.cwd()

DATA_DIR = Path("production")

def temporal_features(signal):
    return {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "rms": np.sqrt(np.mean(signal**2)),
        "max": np.max(signal),
        "min": np.min(signal),
        "ptp": np.ptp(signal),
        "kurtosis": kurtosis(signal),
        "skewness": skew(signal),
        "zero_crossings": np.sum(np.diff(np.sign(signal)) != 0)
    }

# Nombres de los sensores (en el orden correcto)
sensor_names = [
    "tachometer",
    "acc_under_axial",
    "acc_under_radial",
    "acc_under_tangential",
    "acc_over_axial",
    "acc_over_radial",
    "acc_over_tangential",
    "microphone"
]

def extract_features_from_csv(csv_file: Path) -> pd.DataFrame:
    # Cargar CSV SIN header y asignar nombres de sensores
    signal = pd.read_csv(csv_file, header=None, names=sensor_names)
    
    # Diccionario donde guardaremos todas las features
    all_feats = {}

    for sensor in sensor_names:
        feats = temporal_features(signal[sensor].values)
        # Redondeamos a 6 decimales y añadimos prefijo del sensor
        for k, v in feats.items():
            all_feats[f"{sensor}_{k}"] = round(float(v), 6)

    # Creamos un DataFrame de 1 fila con todas las features
    df_features = pd.DataFrame([all_feats])
    return df_features

def prepare_message(csv_file: Path) -> pd.DataFrame:
    df_features = extract_features_from_csv(csv_file)
    frequency_value = csv_file.stem
    df_features.insert(0, "frequency(Hz)", frequency_value)
    df_features["produced_timestamp"] = datetime.now().isoformat()

    return df_features

def delivery_report(err, msg):
    if err:
        print(f"Error al enviar mensaje: {err}")
    else:
        print(f"Mensaje enviado: {msg.key()}")

n = 1
# Loop del producer
for csv_file in DATA_DIR.glob("*.csv"):
    row_df = prepare_message(csv_file)

    row = row_df.iloc[0].to_dict() 

    if writer_raw_data is None:
        writer_raw_data = csv.DictWriter(bronze_raw_data_file, fieldnames=row.keys())
        if not os.path.exists(bronze_raw_data_file):
            writer_raw_data.writeheader()
    writer_raw_data.writerow(row)

    print(f"Medición {n}")
    print(f"   - Frequency(Hz): {csv_file.stem}")
    n += 1

    # Publicar en Kafka
    producer.produce("rodamientos", json.dumps(row).encode("utf-8"))
    time.sleep(1)

    producer.flush()

bronze_raw_data_file.close()