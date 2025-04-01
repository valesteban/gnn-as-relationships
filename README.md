# Trabajo de Tesis: Clasificación Binaria con Redes Neuronales en Grafos

Este proyecto implementa un modelo de clasificación binaria utilizando redes neuronales en grafos (GNNs) con la biblioteca [DGL (Deep Graph Library)](https://www.dgl.ai/). El objetivo principal es entrenar y evaluar modelos de aprendizaje profundo para tareas de predicción de relaciones en grafos.

## Estructura del Proyecto

```
TrabajoTesis/ 
├── data/ # Directorio para los datos de entrada 
├── modules/ # Implementaciones de modelos y predictores 
│       ├── gnn.py # Clase principal para manejar grafos 
│       ├── models.py # Definición de modelos GNN (GCN, GraphSAGE, etc.) 
│       ├── predictors.py # Predictores (DotPredictor, MLPPredictor, etc.) 
├── results/ # Resultados generados (CSV, gráficos, etc.) 
│       ├── binary_classification/ 
├── utils.py # Funciones auxiliares (métricas, visualización, etc.) 
├── binary_classification.ipynb # Notebook principal para ejecutar el flujo de trabajo 
├── requirements.txt # Dependencias del proyecto └── README.md # Este archivo
```



## Requisitos

Este proyecto utiliza Python 3.10 y las siguientes bibliotecas:

- `torch`
- `dgl`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tqdm`

Instala las dependencias ejecutando:

```bash
pip install -r requirements.txt
```

## Flujo de Trabajo
- Carga de Datos: Los datos se cargan desde el directorio data/ y se procesan utilizando la clase GNN en modules/gnn.py.

- Inicialización del Modelo: Los modelos GNN (como GCN o GraphSAGE) y los predictores (como DotPredictor o MLPPredictor) se inicializan en el notebook binary_classification.ipynb.

- Entrenamiento: Los modelos se entrenan utilizando diferentes estrategias de muestreo:
        - NeighborSampler: Muestreo basado en vecinos.
        - ClusterGCNSampler: Muestreo basado en clústeres.

- Evaluación: Se calculan métricas como precisión, recall y F1-score, y se generan gráficos de curvas ROC.

- Resultados: Los resultados (logits y etiquetas) se guardan en archivos CSV en el directorio results/binary_classification/.