import json
import time
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
from kafka import KafkaProducer

from scipy.stats import kurtosis, skew
from scipy.signal import butter, filtfilt, hilbert, welch

import joblib

# ------------------------
# Config
# ------------------------
BOOTSTRAP_SERVERS = ["localhost:9092"]
TOPIC_OUT = "bearing_features_top20"

FS = 51200
window_sec = 0.2
window_size = int(window_sec * FS)      # 10240
step = window_size // 2                 # 5120

# Si simulas streaming desde CSV:
CHUNK_SEC = 0.05                        # tamaño de chunk enviado al "buffer"
SLEEP_REAL_TIME = False                 # True para simular tiempo real

sensor_names = [
    "tachometer",
    "acc_under_axial","acc_under_radial","acc_under_tangential",
    "acc_over_axial","acc_over_radial","acc_over_tangential",
    "microphone"
]

# Carga de top20 (el mismo fichero que ya guardaste)
TOP_FEATURES_PATH = "top20_features_windows.joblib"
top_features = joblib.load(TOP_FEATURES_PATH)

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# ------------------------
# Utilidades
# ------------------------
def rpm_from_filename(p: Path) -> float:
    # "12.288.csv" -> 12.288 Hz -> 737.28 rpm
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
    for o in [1, 2, 3, 4, 5]:
        out[f"ord_{o}x"] = band_energy(freqs, psd, f0=o*fr_hz, tol=0.05)
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
    out = {k: band_energy(freqs, psd, v, tol=0.1) for k, v in targets.items()}
    out["env_total"] = float(np.trapezoid(psd, freqs))
    return out

def features_last_window(buffer_df: pd.DataFrame, rpm: float) -> dict:
    """
    Extrae features SOLO de la ultima ventana de buffer_df
    usando acc_over_radial y acc_under_radial, como en entrenamiento.
    """
    if len(buffer_df) < window_size:
        return {}

    fr_hz = rpm / 60.0

    over = buffer_df["acc_over_radial"].to_numpy()[-window_size:]
    under = buffer_df["acc_under_radial"].to_numpy()[-window_size:]

    row = {}
    row.update({f"acc_over_radial_{k}": v for k, v in time_feats(over).items()})
    row.update({f"acc_over_radial_{k}": v for k, v in order_features(over, FS, fr_hz).items()})
    row.update({f"acc_over_radial_{k}": v for k, v in envelope_features(over, FS, fr_hz).items()})

    row.update({f"acc_under_radial_{k}": v for k, v in time_feats(under).items()})
    row.update({f"acc_under_radial_{k}": v for k, v in order_features(under, FS, fr_hz).items()})
    row.update({f"acc_under_radial_{k}": v for k, v in envelope_features(under, FS, fr_hz).items()})

    # añade rpm porque está en tus features entrenadas
    row["rpm"] = float(rpm)

    # nos quedamos SOLO con top20 (y en el orden correcto)
    # Si falta alguna, mejor fallar con error explícito:
    missing = [c for c in top_features if c not in row]
    if missing:
        raise ValueError(f"Faltan features top20: {missing}")

    row_top20 = {k: row[k] for k in top_features}
    return row_top20

# ------------------------
# Streaming desde CSV -> buffer -> emite features cada "step"
# ------------------------
def run_from_csv_folder(root: Path):
    files = sorted(root.rglob("*.csv"))
    if not files:
        raise RuntimeError(f"No hay CSV en {root}")

    chunk_size = int(CHUNK_SEC * FS)

    for csv_path in files:
        df = pd.read_csv(csv_path, header=None, names=sensor_names)
        rpm = rpm_from_filename(csv_path)
        series_id = str(csv_path)

        # buffer (vamos acumulando filas)
        buffer = pd.DataFrame(columns=sensor_names)
        last_emit_end = 0  # para controlar el "step" de emisión

        n = len(df)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = df.iloc[start:end]
            buffer = pd.concat([buffer, chunk], ignore_index=True)

            # mantener buffer acotado (no crecer infinito):
            # deja margen para poder sacar ultima ventana
            if len(buffer) > window_size + step:
                buffer = buffer.iloc[-(window_size + step):].reset_index(drop=True)

            # si ya podemos emitir, emitimos cada "step"
            # (cada vez que haya llegado al menos "step" muestras nuevas desde última emisión)
            if len(buffer) >= window_size:
                # end absoluto en serie original aproximado (no exacto por el recorte)
                # usamos un contador interno simple:
                if (end - last_emit_end) >= step:
                    feat_top20 = features_last_window(buffer, rpm=rpm)

                    payload = {
                        "series_id": series_id,
                        "rpm": float(rpm),
                        "fs": FS,
                        "window_sec": window_sec,
                        "window_size": window_size,
                        "step": step,
                        "chunk_end_sample": int(end),
                        "features_top20": feat_top20,
                        "feature_order": top_features  # útil para debug
                    }

                    producer.send(TOPIC_OUT, payload)
                    producer.flush()
                    last_emit_end = end

            if SLEEP_REAL_TIME:
                time.sleep((end - start) / FS)

        print(f"[OK] Enviado features top20 para: {csv_path}")

if __name__ == "__main__":
    # carpeta desde la que simulas streaming
    root = Path.cwd() / "bearing_fault_detection_reduced" / "normal"  # cambia a lo que quieras
    run_from_csv_folder(root)

