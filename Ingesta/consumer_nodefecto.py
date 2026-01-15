# Ingesta/consumer_no_defecto.py
import os
from pathlib import Path
import cv2

from confluent_kafka import Consumer
from modelos import decode_img

# =========================
# KAFKA
# =========================
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC_NO_DEFECTO = os.getenv("TOPIC_NO_DEFECTO", "imagenes.no_defecto")
GROUP_ID = os.getenv("GROUP_NO_DEFECTO", "grp_no_defecto")

# =========================
# RUTAS (datalake/no_defecto)
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../grupo1reto2
OUT_DIR = PROJECT_ROOT / "datalake" / "no_defecto"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def headers_to_dict(msg) -> dict:
    hs = msg.headers() or []
    out = {}
    for k, v in hs:
        if isinstance(v, (bytes, bytearray)):
            out[k] = v.decode("utf-8", errors="ignore")
        else:
            out[k] = str(v)
    return out


def build_filename(meta: dict, key_bytes: bytes | None) -> str:
    """
    Nombre final: <stem>__aug-<tag>__conf-<conf>__ts-<ts>.jpg
    Usa orig_name si existe; si no, usa key.
    """
    orig_name = meta.get("orig_name", "")
    ts_ms = meta.get("ts_ms", "0")
    aug = meta.get("aug", "none")
    conf = meta.get("pred_conf", "0")

    if orig_name:
        stem = Path(orig_name).stem
    elif key_bytes:
        stem = key_bytes.decode("utf-8", errors="ignore")
    else:
        stem = "unknown"

    # Limpieza básica para que no haya caracteres raros en Windows
    safe_aug = aug.replace(" ", "").replace("/", "_").replace("\\", "_")
    safe_conf = conf.replace(" ", "")

    return f"{stem}__aug-{safe_aug}__conf-{safe_conf}__ts-{ts_ms}.jpg"


def main():
    print("[Consumer NO_DEFECTO] Topic:", TOPIC_NO_DEFECTO)
    print("[Consumer NO_DEFECTO] Guardando en:", OUT_DIR)

    consumer = Consumer({
        "bootstrap.servers": BOOTSTRAP_SERVERS,
        "group.id": GROUP_ID,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
    })

    consumer.subscribe([TOPIC_NO_DEFECTO])

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print("[Consumer NO_DEFECTO] Error:", msg.error())
                continue

            # 1) reconstruir imagen
            img = decode_img(msg.value())

            # 2) metadatos
            meta = headers_to_dict(msg)
            key = msg.key()  # bytes o None

            # 3) nombre y guardado
            filename = build_filename(meta, key)
            out_path = OUT_DIR / filename

            ok = cv2.imwrite(str(out_path), img)
            if ok:
                print(f"[Consumer NO_DEFECTO] Guardado: {out_path.name}")
            else:
                print(f"[Consumer NO_DEFECTO] ERROR guardando: {out_path}")

    except KeyboardInterrupt:
        print("\n[Consumer NO_DEFECTO] Parado.")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
