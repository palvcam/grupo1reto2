from confluent_kafka import Consumer
import pandas as pd
import os
import json
import joblib
from datetime import datetime
import csv
from pathlib import Path

# Cargar el modelo entrenado
model = joblib.load('gb_top20_windows_model.pkl')
print("Modelo cargado desde 'gb_top20_windows_model.pkl'\n")

# Configuración del consumer
consumer_config = {
    'bootstrap.servers': '127.0.0.1:9092',
    'group.id': 'datalake_processing_group',
    'auto.offset.reset': 'earliest',
}
consumer = Consumer(consumer_config)
consumer.subscribe(['rodamientos'])

# Crear carpetas
os.makedirs("datalake/processed/normal", exist_ok=True)
os.makedirs("datalake/processed/horizontal_misalignment", exist_ok=True)
os.makedirs("datalake/processed/vertical_misalignment", exist_ok=True)
os.makedirs("datalake/processed/imbalance", exist_ok=True)
os.makedirs("datalake/processed/ball_fault", exist_ok=True)
os.makedirs("datalake/processed/cage_fault", exist_ok=True)
os.makedirs("datalake/processed/outer_race", exist_ok=True)

# Definir paths
PROCESSED_NORMAL_FILE = Path("datalake/processed/normal/normal.csv")
PROCESSED_HM_FILE = Path("datalake/processed/horizontal_misalignment/horizontal_misalignment.csv")
PROCESSED_VM_FILE = Path("datalake/processed/vertical_misalignment/vertical_misalignment.csv")
PROCESSED_IMBALANCE_FILE = Path("datalake/processed/imbalance/imbalance.csv")
PROCESSED_BF_FILE = Path("datalake/processed/ball_fault/ball_fault.csv")
PROCESSED_CF_FILE = Path("datalake/processed/cage_fault/cage_fault.csv")
PROCESSED_OR_FILE = Path("datalake/processed/outer_race/outer_race.csv")

files = {
    "normal": open(PROCESSED_NORMAL_FILE, "a", newline=""),
    "hm": open(PROCESSED_HM_FILE, "a", newline=""),
    "vm": open(PROCESSED_VM_FILE, "a", newline=""),
    "imbalance": open(PROCESSED_IMBALANCE_FILE, "a", newline=""),
    "bf": open(PROCESSED_BF_FILE, "a", newline=""),
    "cf": open(PROCESSED_CF_FILE, "a", newline=""),
    "or": open(PROCESSED_OR_FILE, "a", newline="")
}

writer_normal = None
writer_hm = None
writer_vm = None
writer_imbalance = None
writer_bf = None
writer_cf = None
writer_or = None

# Cargar CSV de etiquetas de test
test_labels_file = Path("test_labels.csv")
test_labels = {}
with open(test_labels_file, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Guardamos como {frequency: fault_type}
        test_labels[row["frequency"]] = row["fault_type"]

# Diccionario para llevar conteo de predicciones correctas
prediction_results = {
    "total": 0,
    "correct": 0
}

def prepare_message(message):
    """Prepara el mensaje para predicción"""
    df_temp = pd.DataFrame([message])
    df_temp = df_temp.drop(["frequency(Hz)", "produced_timestamp"], axis=1)
    rpm_col = df_temp.pop("rpm")  # Extrae la columna
    df_temp.insert(6, "rpm", rpm_col)

    return df_temp

if __name__ == "__main__":
    print("Consumer iniciado. Leyendo del topic 'rodamientos'")
    print("-" * 80)
    try:
        counter = 1
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"Error: {msg.error()}")
                continue
            # Leer el mensaje y decodificarlo
            try:
                # Procesar el mensaje recibido
                data = json.loads(msg.value().decode('utf-8'))

                # Preparar los datos para la predicción
                feats_df = prepare_message(data)

                # Hacer la predicción
                fault_prediction = model.predict(feats_df)[0]
                prediction_timestamp = datetime.now().isoformat()

                print(f"Medición {counter}: ")
                print(f"   - Frecuencia(Hz): {data['frequency(Hz)']}")
                print(f"   - Predicción de error: {fault_prediction}")
                print(f"   - Fecha producida: {data['produced_timestamp']}")
                print(f"   - Fecha predicción: {prediction_timestamp}")
                print("-" * 80)
                    
                data["prediction_timestamp"] = datetime.now().isoformat()
                data["fault_type"] = fault_prediction

                frequency = data["frequency(Hz)"]

                # Comprobamos si la frecuencia está en el CSV de test
                if frequency in test_labels:
                    prediction_results["total"] += 1
                    if fault_prediction == test_labels[frequency]:
                        prediction_results["correct"] += 1

                # Escribir en datalake dependiendo de la predicción
                match fault_prediction:
                    case "normal":
                            writer_normal = csv.DictWriter(files["normal"], fieldnames=data.keys())
                            if PROCESSED_NORMAL_FILE.stat().st_size == 0:
                                writer_normal.writeheader()
                            writer_normal.writerow(data)

                    case "horizontal_misalignment":
                            writer_hm = csv.DictWriter(files["hm"], fieldnames=data.keys())
                            if PROCESSED_HM_FILE.stat().st_size == 0:
                                writer_hm.writeheader()
                            writer_hm.writerow(data)

                    case "vertical_misalignment":
                            writer_vm = csv.DictWriter(files["vm"], fieldnames=data.keys())
                            if PROCESSED_VM_FILE.stat().st_size == 0:
                                writer_vm.writeheader()
                            writer_vm.writerow(data)

                    case "imbalance":
                            writer_imbalance = csv.DictWriter(files["imbalance"], fieldnames=data.keys())
                            if PROCESSED_IMBALANCE_FILE.stat().st_size == 0:
                                writer_imbalance.writeheader()
                            writer_imbalance.writerow(data)

                    case "ball_fault":
                            writer_bf = csv.DictWriter(files["bf"], fieldnames=data.keys())
                            if PROCESSED_BF_FILE.stat().st_size == 0:
                                writer_bf.writeheader()
                            writer_bf.writerow(data)

                    case "cage_fault":
                        writer_cf = csv.DictWriter(files["cf"], fieldnames=data.keys())
                        if PROCESSED_CF_FILE.stat().st_size == 0:
                            writer_cf.writeheader()
                        writer_cf.writerow(data)

                    case "outer_race":
                        writer_or = csv.DictWriter(files["or"], fieldnames=data.keys())
                        if PROCESSED_OR_FILE.stat().st_size == 0:
                            writer_or.writeheader()
                        writer_or.writerow(data)

                counter += 1

            except json.JSONDecodeError as e:
                print(f"Error decodificando JSON: {e}")
            except Exception as e:
                print(f"Error procesando datos: {e}")

    except KeyboardInterrupt:
        print("Interrumpido por el usuario.")

    finally:
        # Cerrar todos los archivos al final
        for f in files.values():
            f.close()
        consumer.close()
        correct = prediction_results["correct"]
        total = prediction_results["total"]
        if total > 0:
            precision = correct / total * 100
            print(f"\nResumen de predicciones sobre test:")
            print(f"Correctas: {correct}/{total}")
            print(f"Precisión: {precision:.2f}%")
