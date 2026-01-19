from confluent_kafka import Consumer
import pandas as pd
import os
import json

# Cargar el modelo entrenado
model = joblib.load('.pkl')
print("Modelo cargado desde 'model.pkl'")

# Configuración del consumidor
consumer_config = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'datalake_processing_group',
    'auto.offset.reset': 'earliest',
}
consumer = Consumer(consumer_config)
consumer.subscribe(['rodamientos'])

# Crear carpetas Silver
processed_path = "datalake/silver"

os.makedirs("datalake/silver/normal", exist_ok=True)
os.makedirs("datalake/silver/horizontal_misalignment", exist_ok=True)
os.makedirs("datalake/silver/vertical_misalignment", exist_ok=True)
os.makedirs("datalake/silver/imbalance", exist_ok=True)
os.makedirs("datalake/silver/ball_fault", exist_ok=True)
os.makedirs("datalake/silver/cage_fault", exist_ok=True)
os.makedirs("datalake/silver/outer_race", exist_ok=True)

silver_normal_file = open("datalake/silver/normal/normal.csv", "w", newline="")
silver_hm_file = open("datalake/silver/horizontal_misalignment/horizontal_misalignment.csv", "w", newline="")
silver_vm_file = open("datalake/silver/vertical_misalignment/vertical_misalignment.csv", "w", newline="")
silver_imbalance_file = open("datalake/silver/imbalance/imbalance.csv", "w", newline="")
silver_bf_file = open("datalake/silver/ball_fault/ball_fault.csv", "w", newline="")
silver_cf_file = open("datalake/silver/cage_fault/cage_fault.csv", "w", newline="")
silver_or_file = open("datalake/silver/outer_race/outer_race.csv", "w", newline="")

writer_normal = None
writer_hm = None
writer_vm = None
writer_imbalance = None
writer_bf = None
writer_cf = None
writer_or = None


def prepare_message(data):
    """Prepara el mensaje para predicción"""
    df_temp = pd.DataFrame([data])
    df_temp = df_temp.drop("fecha_producida", axis=1)

    # Asegurar que tenga todas las columnas esperadas
    for col in expected_columns:
        if col not in df_temp.columns:
            df_temp[col] = 0

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
                data_df = pd.DataFrame([data])

                # Preparar los datos para la predicción
                X_data = prepare_message(data)

                # Hacer la predicción
                data_prediction = model.predict(X_data)[0]
                prediction_timestamp = datetime.now().isoformat()

                print(f"Medición {counter}: ")
                print(f"   - Frecuencia(Hz)": {data_df["frequency(Hz)"]})
                print(f"   - Predicción de error: {data_prediction}")
                print(f"   - Timestamp: {data["timestamp"]}")
                print(f"   - Prediction timestamp: {prediction_timestamp}")
                print("-" * 80)
                    
                data_df["prediction_timestamp"] = datetime.now().isoformat()
                data_df["fault_type"] = data_prediction

                # Escribir en datalake Silver dependiendo de la predicción
                match:
                    case "normal":
                        if writer_normal is None:
                            writer_normal = csv.DictWriter(silver_normal_file, fieldnames=row.keys())
                            writer_normal.writeheader()
                        writer_normal.writerow(row)

                    case "horizontal_misalignment":
                        if writer_hm is None:
                            writer_hm = csv.DictWriter(silver_hm_file, fieldnames=row.keys())
                            writer_hm.writeheader()
                        writer_hm.writerow(row)

                    case "vertical_misalignment":
                        if writer_vm is None:
                            writer_vm = csv.DictWriter(silver_vm_file, fieldnames=row.keys())
                            writer_vm.writeheader()
                        writer_vm.writerow(row)

                    case "imbalance":
                        if writer_imbalance is None:
                            writer_imbalance = csv.DictWriter(silver_imbalance_file, fieldnames=row.keys())
                            writer_imbalance.writeheader()
                        writer_imbalance.writerow(row)

                    case "ball_fault":
                        if writer_bf is None:
                            writer_bf = csv.DictWriter(silver_bf_file, fieldnames=row.keys())
                            writer_bf.writeheader()
                        writer_bf.writerow(row)

                    case "cage_fault":
                        if writer_cf is None:
                            writer_cf = csv.DictWriter(silver_cf_file, fieldnames=row.keys())
                            writer_cf.writeheader()
                        writer_cf.writerow(row)

                    case "outer_race":
                        if writer_or is None:
                            writer_or = csv.DictWriter(silver_or_file, fieldnames=row.keys())
                            writer_or.writeheader()
                        writer_or.writerow(row)

            except json.JSONDecodeError as e:
                print(f"Error decodificando JSON: {e}")
            except Exception as e:
                print(f"Error procesando transacción: {e}")

    except KeyboardInterrupt:
        print("Interrumpido por el usuario.")

    finally:
        consumer.close()
