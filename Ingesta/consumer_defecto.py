# Ingesta/consumer_defecto_detect.py
import os
from pathlib import Path
import cv2

from confluent_kafka import Consumer
from ultralytics import YOLO

from modelos import decode_img

# =========================
# KAFKA
# =========================
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC_DEFECTO = os.getenv("TOPIC_DEFECTO", "imagenes.defecto")
GROUP_ID = os.getenv("GROUP_DEFECTO", "grp_defecto")

# =========================
# RUTAS (datalake/defecto)
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../grupo1reto2
OUT_DIR = PROJECT_ROOT / "datalake" / "defecto"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# MODELO DETECCIÓN (auto-detecta best.pt)
# =========================
DET_RUNS_DIR = PROJECT_ROOT / "Deteccion" / "runs"
CONF_DET = float(os.getenv("CONF_DET", "0.30"))
IOU_DET = float(os.getenv("IOU_DET", "0.30"))


def headers_to_dict(msg) -> dict:
    hs = msg.headers() or []
    out = {}
    for k, v in hs:
        if isinstance(v, (bytes, bytearray)):
            out[k] = v.decode("utf-8", errors="ignore")
        else:
            out[k] = str(v)
    return out


def latest_best_pt(search_dir: Path) -> Path:
    if not search_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de runs: {search_dir}")

    bests = list(search_dir.rglob("weights/best.pt"))
    if not bests:
        raise FileNotFoundError(f"No encontré ningún weights/best.pt dentro de {search_dir}")

    bests.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return bests[0]


def resolve_det_model_path() -> Path:
    """
    Si defines MODEL_DET_PATH en env, lo usa.
    Si no, busca el best.pt más reciente en Deteccion/runs/**/weights/best.pt
    """
    env_path = os.getenv("MODEL_DET_PATH", "").strip()
    if env_path:
        p = Path(env_path)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        return p

    return latest_best_pt(DET_RUNS_DIR)


def build_filename(meta: dict, key_bytes: bytes | None, det_n: int) -> str:
    orig_name = meta.get("orig_name", "")
    ts_ms = meta.get("ts_ms", "0")
    aug = meta.get("aug", "none")
    cls_conf = meta.get("pred_conf", "0")

    if orig_name:
        stem = Path(orig_name).stem
    elif key_bytes:
        stem = key_bytes.decode("utf-8", errors="ignore")
    else:
        stem = "unknown"

    safe_aug = aug.replace(" ", "").replace("/", "_").replace("\\", "_")
    safe_conf = cls_conf.replace(" ", "")
    return f"{stem}__aug-{safe_aug}__clsconf-{safe_conf}__detN-{det_n}__ts-{ts_ms}.jpg"


def draw_boxes(img, result) -> tuple[any, int]:
    """
    Dibuja todas las detecciones.
    Devuelve (imagen_con_boxes, num_detecciones)
    """
    if result.boxes is None or len(result.boxes) == 0:
        return img, 0

    out = img.copy()
    n = len(result.boxes)

    for b in result.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        conf = float(b.conf[0])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{conf:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return out, n


def main():
    det_model_path = resolve_det_model_path()
    print("[Consumer DEFECTO] Topic:", TOPIC_DEFECTO)
    print("[Consumer DEFECTO] Guardando en:", OUT_DIR)
    print("[Consumer DEFECTO] DET_MODEL_PATH:", det_model_path, "| exists =", det_model_path.exists())

    if not det_model_path.exists():
        raise FileNotFoundError(f"No existe el modelo de detección: {det_model_path}")

    model_det = YOLO(str(det_model_path))

    consumer = Consumer({
        "bootstrap.servers": BOOTSTRAP_SERVERS,
        "group.id": GROUP_ID,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
    })
    consumer.subscribe([TOPIC_DEFECTO])

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print("[Consumer DEFECTO] Error:", msg.error())
                continue

            img = decode_img(msg.value())
            meta = headers_to_dict(msg)
            key = msg.key()

            # 1) inferencia detección (sobre numpy array)
            res = model_det.predict(img, conf=CONF_DET, iou=IOU_DET, verbose=False)[0]

            # 2) dibujar bounding boxes
            img_out, det_n = draw_boxes(img, res)

            # 3) guardar resultado
            filename = build_filename(meta, key, det_n)
            out_path = OUT_DIR / filename

            ok = cv2.imwrite(str(out_path), img_out)
            if ok:
                print(f"[Consumer DEFECTO] Guardado: {out_path.name} (detecciones={det_n})")
            else:
                print(f"[Consumer DEFECTO] ERROR guardando: {out_path}")

    except KeyboardInterrupt:
        print("\n[Consumer DEFECTO] Parado.")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
