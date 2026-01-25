# grupo1reto2 - Plataforma Inteligente para Mantenimiento Predictivo
Repositorio del Grupo 1 para el Reto 2.

Instrucciones de uso del proyecto:

- Instalar las librerías necesarias con ````pip install -r requirements.txt````

## Pipeline 1 - Rodamientos
1. Descargar las carpetas **bearing_fault_detection** y **bearing_fault_detection_reduced** y guardarlas en la carpeta **Rodamientos**
2. Los archivos **gb_top20_windows_model.pkl** y **top20_features_windows.joblib** han sido generados por el notebook **preprocesamiento_y_modelos.ipynb** previamente para ahorrar tiempo de ejecución. Si por algún motivo no existen esos archivos, ejecutar el notebook.
3. En la carpeta **Rodamientos**, hacer ````docker compose up -d```` para levantar los servicios **Kafka** y **Zookeeper** 
4. Ejecutar simultáneamente los scripts **producer.py** y **consumer.py** en terminales diferentes (en Windows es necesario tener **Docker Desktop** abierto)

## Pipeline 2 - Piezas
1. Descargar la carpeta **Imagenes_defectos**
2. Ejecutar el script **primera_particion_stream** para guardar un conjunto de imágenes y que los modelos no las vean
3. En la carpeta **Clasificación** ejecutar el script **clasificacion_defecto**.
4. Ejecutar el script **clasificacion**
5. En la carpeta **Deteccion** ejecutar el notebook **deteccion.ipynb**
6. En la carpeta **Ingesta** ejecutar el script **modelo**
7. En la consola ejecutar el pipeline de Kafka
