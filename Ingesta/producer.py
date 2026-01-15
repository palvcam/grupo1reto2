# Ingesta/producer.py
from pathlib import Path
import os
import time
import cv2
import random

from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic
from ultralytics import YOLO

from modelos import IMG_EXTS, encode_jpg, simple_augment, now_ms

# =========================
# KAFKA
# =========================
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC_DEFECTO = "imagenes.defecto"
TOPIC_NO_DEFECTO = "imagenes.no_defecto"

# =========================
# PROYECTO / RUTAS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../grupo1reto2
IMAGENES_DEFECTOS_DIR = PROJECT_ROOT / "Ingesta/stream_pool"

# Modelo clasificación (auto-detecta el best.pt más reciente)
CLS_DIR_1 = PROJECT_ROOT / "Clasificacion" / "runs_cls"
CLS_DIR_2 = PROJECT_ROOT / "Clasificacion" / "runs"

# =========================
# INFERENCIA / STREAM
# =========================
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
THRESH_DEFECTO = float(os.getenv("THRESH_DEFECTO", "0.50"))
POSITIVE_CLASS_NAME = os.getenv("POSITIVE_CLASS_NAME", "defecto")

AUG_PER_IMAGE = int(os.getenv("AUG_PER_IMAGE", "2"))
SLEEP_SEC = float(os.getenv("SLEEP_SEC", "0.01"))

# Opcional: mezclar orden de imágenes en cada vuelta
SHUFFLE_EACH_EPOCH = os.getenv("SHUFFLE_EACH_EPOCH", "1") == "1"


def ensure_topics():
    admin = AdminClient({"bootstrap.servers": BOOTSTRAP_SERVERS})
    md = admin.list_topics(timeout=5)
    existing = set(md.topics.keys())

    to_create = []
    for t in [TOPIC_DEFECTO, TOPIC_NO_DEFECTO]:
        if t not in existing:
            to_create.append(NewTopic(t, num_partitions=1, replication_factor=1))

    if not to_create:
        print("[Producer] Topics OK:", TOPIC_DEFECTO, TOPIC_NO_DEFECTO)
        return

    fs = admin.create_topics(to_create, request_timeout=10)
    for topic, f in fs.items():
        try:
            f.result()
            print(f"[Producer] Topic creado: {topic}")
        except Exception as e:
            print(f"[Producer] No pude crear topic {topic}: {e}")


def delivery_report(err, msg):
    if err is not None:
        print(f"[Producer] Delivery failed: {err}")


def latest_best_pt(search_dir: Path) -> Path | None:
    if not search_dir.exists():
        return None
    bests = list(search_dir.rglob("weights/best.pt"))
    if not bests:
        return None
    bests.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return bests[0]


def resolve_cls_model_path() -> Path:
    env_path = os.getenv("MODEL_CLS_PATH", "").strip()
    if env_path:
        p = Path(env_path)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        return p

    p1 = latest_best_pt(CLS_DIR_1)
    if p1 is not None:
        return p1

    p2 = latest_best_pt(CLS_DIR_2)
    if p2 is not None:
        return p2

    raise FileNotFoundError(
        "No encontré ningún 'weights/best.pt' de CLASIFICACIÓN.\n"
        f"Busqué en:\n- {CLS_DIR_1}\n- {CLS_DIR_2}\n"
        "Solución: entrena el modelo o define MODEL_CLS_PATH."
    )


def list_images_only(base_dir: Path) -> list[Path]:
    paths = []
    for p in base_dir.rglob("*"):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf not in IMG_EXTS:
            continue
        if "_label" in p.stem.lower():
            continue
        paths.append(p)
    return sorted(paths)


def main():
    if not IMAGENES_DEFECTOS_DIR.exists():
        raise FileNotFoundError(f"No existe: {IMAGENES_DEFECTOS_DIR}")

    cls_model_path = resolve_cls_model_path()
    print("IMAGENES_DEFECTOS_DIR:", IMAGENES_DEFECTOS_DIR, "| exists =", IMAGENES_DEFECTOS_DIR.exists())
    print("CLS_MODEL_PATH:", cls_model_path, "| exists =", cls_model_path.exists())
    if not cls_model_path.exists():
        raise FileNotFoundError(f"No existe el modelo cls: {cls_model_path}")

    ensure_topics()

    model_cls = YOLO(str(cls_model_path))
    producer = Producer({"bootstrap.servers": BOOTSTRAP_SERVERS})

    img_paths_master = list_images_only(IMAGENES_DEFECTOS_DIR)
    print(f"[Producer] Imágenes (sin labels) encontradas: {len(img_paths_master)}")
    if not img_paths_master:
        return

    epoch = 0

    try:
        while True:
            epoch += 1
            img_paths = img_paths_master.copy()
            if SHUFFLE_EACH_EPOCH:
                random.shuffle(img_paths)

            print(f"[Producer] === EPOCH {epoch} === enviando {len(img_paths)} imágenes x {AUG_PER_IMAGE} aug")

            for img_path in img_paths:
                img = cv2.imread(str(img_path))
                if img is None:
                    print("[Producer] No pude leer:", img_path)
                    continue

                for k in range(AUG_PER_IMAGE):
                    img_aug, aug_tag = simple_augment(img)

                    r = model_cls.predict(img_aug, imgsz=IMG_SIZE, verbose=False)[0]
                    pred_id = int(r.probs.top1)
                    conf = float(r.probs.top1conf)
                    pred_name = r.names[pred_id]

                    is_def = (pred_name == POSITIVE_CLASS_NAME) and (conf >= THRESH_DEFECTO)
                    topic = TOPIC_DEFECTO if is_def else TOPIC_NO_DEFECTO

                    payload = encode_jpg(img_aug, quality=90)

                    headers = [
                        ("orig_name", img_path.name.encode("utf-8")),
                        ("aug", f"{aug_tag}#{k+1}".encode("utf-8")),
                        ("pred_class", str(pred_name).encode("utf-8")),
                        ("pred_conf", f"{conf:.6f}".encode("utf-8")),
                        ("ts_ms", str(now_ms()).encode("utf-8")),
                        ("epoch", str(epoch).encode("utf-8")),
                    ]

                    key = f"{img_path.stem}_{epoch}_{k+1}".encode("utf-8")

                    producer.produce(
                        topic=topic,
                        key=key,
                        value=payload,
                        headers=headers,
                        callback=delivery_report
                    )
                    producer.poll(0)

                    print(f"[Producer] {img_path.name} (aug {k+1}/{AUG_PER_IMAGE}) -> {topic} "
                          f"(pred={pred_name} conf={conf:.3f} tag={aug_tag})")

                    if SLEEP_SEC > 0:
                        time.sleep(SLEEP_SEC)

            # asegura que se envía todo al final de cada vuelta
            producer.flush()

    except KeyboardInterrupt:
        print("\n[Producer] Parado por consola (Ctrl+C).")
    finally:
        # flush final por si quedaba algo
        try:
            producer.flush(5)
        except Exception:
            pass


if __name__ == "__main__":
    main()
