from confluent_kafka import Producer
import pandas as pd
import json
import os
from pathlib import Path
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import butter, filtfilt, hilbert, welch
from datetime import datetime
import csv
import time
import joblib
import random

# Configuración del producer
producer_config = {
    'bootstrap.servers': '127.0.0.1:9092',
}
producer = Producer(producer_config)

# Datalake

# Crear carpetas
os.makedirs("datalake/raw", exist_ok=True)

RAW_DATA_FILE = Path("datalake/raw/raw_data.csv")

writer_raw_data = None

PATH = Path.cwd()
DATA_DIR = Path("production")

# Parámetros del modelo
FS = 51200
window_sec = 0.2
window_size = int(window_sec * FS)      # 10240
step = window_size // 2                 # 5120

CHUNK_SEC = 0.05
SLEEP_REAL_TIME = False


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

# Cargar top20
top_features = joblib.load("top20_features_windows.joblib")

# Funciones auxiliares
def rpm_from_filename(p: Path) -> float:
    return float(p.stem) * 60.0

def time_feats(x):
    x = np.asarray(x)
    rms = np.sqrt(np.mean(x**2))
    return {
        "rms": float(rms),
        "std": float(np.std(x)),
        "mean": float(np.mean(x)),
        "kurtosis": float(kurtosis(x, fisher=False)),
        "skew": float(skew(x)),
        "crest_factor": float(np.max(np.abs(x)) / (rms + 1e-12)),
        "ptp": float(np.ptp(x)),
    }

def bandpass(x, fs, low, high, order=4):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype="band")
    return filtfilt(b, a, x)

def band_energy(freqs, psd, f0, tol=0.1):
    mask = (freqs >= f0*(1-tol)) & (freqs <= f0*(1+tol))
    if np.any(mask):
        return float(np.trapezoid(psd[mask], freqs[mask]))
    return 0.0

def order_features(x, fs, fr_hz):
    freqs, psd = welch(x - np.mean(x), fs=fs, nperseg=8192)
    out = {}
    for o in [1,2,3,4,5]:
        out[f"ord_{o}x"] = band_energy(freqs, psd, o*fr_hz)
    out["psd_total"] = float(np.trapezoid(psd, freqs))
    return out

def envelope_features(x, fs, fr_hz):
    xf = bandpass(x - np.mean(x), fs, low=500, high=8000)
    env = np.abs(hilbert(xf))
    freqs, psd = welch(env, fs=fs, nperseg=8192)

    targets = {
        "env_FTF": 0.375 * fr_hz,
        "env_BSF": 1.87  * fr_hz,
        "env_BPFO": 3.0  * fr_hz,
        "env_BPFI": 5.0  * fr_hz,
    }

    out = {k: band_energy(freqs, psd, f0) for k, f0 in targets.items()}
    out["env_total"] = float(np.trapezoid(psd, freqs))
    return out


def extract_last_window_features(buffer_df, rpm):
    """ Produce features SOLO de la última ventana """
    if len(buffer_df) < window_size:
        return None

    fr_hz = rpm / 60.0

    over = buffer_df["acc_over_radial"].to_numpy()[-window_size:]
    under = buffer_df["acc_under_radial"].to_numpy()[-window_size:]

    row = {}

    # Over
    row.update({f"acc_over_radial_{k}": v for k,v in time_feats(over).items()})
    row.update({f"acc_over_radial_{k}": v for k,v in order_features(over, FS, fr_hz).items()})
    row.update({f"acc_over_radial_{k}": v for k,v in envelope_features(over, FS, fr_hz).items()})

    # Under
    row.update({f"acc_under_radial_{k}": v for k,v in time_feats(under).items()})
    row.update({f"acc_under_radial_{k}": v for k,v in order_features(under, FS, fr_hz).items()})
    row.update({f"acc_under_radial_{k}": v for k,v in envelope_features(under, FS, fr_hz).items()})

    # rpm la usabas en entrenamiento
    row["rpm"] = float(rpm)

    # filtrar top20
    row20 = {k: row[k] for k in top_features}
    return row20


def delivery_report(err, msg):
    if err:
        print(f"Error al enviar mensaje: {err}")
    else:
        print(f"Mensaje entregado al topic '{msg.topic()}'")


if __name__ == "__main__":
    counter = 1
    try:
        # Loop del producer
        all_csv = list(DATA_DIR.glob("*.csv"))
        random.shuffle(all_csv)

        for csv_path in all_csv:
            df = pd.read_csv(csv_path, header=None, names=sensor_names)
            frequency = csv_path.stem
            rpm = rpm_from_filename(csv_path)
            rpm = round(rpm, 5)

            feats = extract_last_window_features(df, rpm)  # Solo la última ventana
            if feats is not None:
                    produced_timestamp = datetime.now().isoformat()

                    payload = {
                        "frequency(Hz)": frequency,
                        "rpm": float(rpm),
                        "produced_timestamp": produced_timestamp,
                        **feats  # aplanamos aquí
                    }

                    print(f"Medición {counter}")
                    print(f"   - Frecuencia del motor (Hz): {csv_path.stem}")
                    print(f"   - RPM: {rpm}")
                    print(f"   - Fecha producida: {produced_timestamp}")
                    counter += 1

                    # Publicar en Kafka
                    producer.produce(
                        "rodamientos",
                        value=json.dumps(payload).encode("utf-8"),
                        callback=delivery_report)
                    producer.poll(0)

                    print("-" * 60)

                    # Escritura en Datalake
                    write_header = not RAW_DATA_FILE.exists() or RAW_DATA_FILE.stat().st_size == 0

                    with open(RAW_DATA_FILE, "a", newline="") as f:
                        writer_raw_data = csv.DictWriter(f, fieldnames=payload.keys())
                        # Escribir header solo si el archivo no existía antes
                        if write_header:
                            writer_raw_data.writeheader()
                        # Escribir la fila
                        writer_raw_data.writerow(payload)

            producer.flush()
            time.sleep(0.25)

    except KeyboardInterrupt:
        print("\n[Producer] Parado por consola (Ctrl+C).")