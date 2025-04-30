# Trabajo:


## Estructura del Proyecto

```
TrabajoTesis/ 
├── data/                               # Directorio para los datos de entrada, embeddings,e tc
├── modules/                            # Implementaciones de modelos y clases auxiliares
│       ├── gnn_models.py               # Definición de modelos GNN (GCN, GraphSAGE, etc.)  
│       ├── gnn.py                      # Clase principal para trabajar tarea con grafo dgl: prepara los datos, hace splits, etc.
│       ├── graph.py                    # Crea grafos a partir de RIBs exportadas a .csv para entrada a DGL 
├── results/                            # Resultados generados (CSV, gráficos, etc.) 
├── utils.py                            # Funciones auxiliares (métricas, visualización, etc.) 
├── train_embeddings.ipynb              # Notebook que genera embeddings usando distintos métodos
├── AS_relationships_inference.ipynb    # Notebook importa embeddings y infiere relaciones entre ASes
```



## Flujo de Trabajo

- El notebook `train_embeddings.ipynb` crea un grafo en DGL utilizando el archivo `sanitized_ribs.txt`, ubicado en la carpeta `data/`.  
  Luego se entrenan distintos modelos de GNN para generar embeddings representativos de la topología.  
  Las tareas de entrenamiento consideradas:

  - **Predicción de enlaces** (Link Prediction)
  - **Predicción de atributos** (Attribute Prediction)

  Una vez generados, los embeddings se guardan con nombres representativos.

- En el notebook `AS_relationships_inference.ipynb`:

  - Se importan los embeddings previamente generados.
  - Se utiliza el dataset de relaciones AS proporcionado por **CAIDA**, que sirve para etiquetar y evaluar.
  - Se entrena una red neuronal (RNN) que toma como entrada dos embeddings y genera como salida la inferencia del tipo de relación entre ASes.
