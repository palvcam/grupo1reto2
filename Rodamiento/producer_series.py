import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from confluent_kafka import Producer
from scipy.stats import kurtosis, skew
from scipy.signal import butter, filtfilt, hilbert, welch

import joblib

# ------------------------
# Configuración Kafka
# ------------------------
BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC_OUT = "bearing_features"

producer = Producer({"bootstrap.servers": BOOTSTRAP_SERVERS})

# ------------------------
# Parámetros del modelo
# ------------------------
FS = 51200
window_sec = 0.2
window_size = int(window_sec * FS)      # 10240
step = window_size // 2                 # 5120

CHUNK_SEC = 0.05
SLEEP_REAL_TIME = False

sensor_names = [
    "tachometer",
    "acc_under_axial","acc_under_radial","acc_under_tangential",
    "acc_over_axial","acc_over_radial","acc_over_tangential",
    "microphone"
]

# Cargar top20
top_features = joblib.load("top20_features_windows.joblib")

# ------------------------
# Funciones auxiliares
# ------------------------
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


# ------------------------
# Main streaming
# ------------------------
def stream_folder(root: Path):
    files = sorted(root.rglob("*.csv"))

    for csv_path in files:
        df = pd.read_csv(csv_path, header=None, names=sensor_names)
        rpm = rpm_from_filename(csv_path)
        series_id = str(csv_path)

        buffer = pd.DataFrame(columns=sensor_names)
        chunk_size = int(CHUNK_SEC * FS)

        for start in range(0, len(df), chunk_size):
            end = min(start + chunk_size, len(df))
            buffer = pd.concat([buffer, df.iloc[start:end]], ignore_index=True)

            # evita que el buffer crezca infinito
            if len(buffer) > window_size + step:
                buffer = buffer.iloc[-(window_size + step):].reset_index(drop=True)

            # si hay ventana completa, calculamos features
            if len(buffer) >= window_size:
                feats = extract_last_window_features(buffer, rpm)
                if feats is None:
                    continue

                payload = {
                    "series_id": series_id,
                    "rpm": float(rpm),
                    "features_top20": feats
                }

                producer.produce(TOPIC_OUT, value=json.dumps(payload))
                producer.poll(0)

            if SLEEP_REAL_TIME:
                time.sleep(len(df.iloc[start:end]) / FS)

        print("[OK] Enviado:", csv_path)

    producer.flush()


if __name__ == "__main__":
    root = Path.cwd() / "bearing_fault_detection_reduced" / "normal"
    stream_folder(root)
