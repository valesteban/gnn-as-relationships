# Trabajo: Inferencia de Relaciones AS usando GNNs

Este repositorio contiene la implementación de un sistema para inferir relaciones entre Sistemas Autónomos (AS) utilizando Redes Neuronales de Grafos (GNNs). Este recive de entrada archivos RIBs pre-procesados de datos de BGP y construye grafos de la topología de Internet. Luego entrena modelos GNN para predecir tipos de relaciones entre ASes (P2P, P2C, C2P).

## Pre-instalación

- Python 3.10
- Acceso a datos de BGP (RIBs) extraidos del [script para crear rutas BGP (create_bgp_routes.py)](https://github.com/niclabs/BenchmarckASRelationships/blob/main/create_bgp_routes.py)
- Archivo CAIDA AS Relationships para las fechas seleccionadas
- Acceso a Internet para la descarga de archivos PeeringDB

## Instalación

1. Crear y activar un entorno virtual:
```bash
python3 -m venv env310
source env310/bin/activate  # En Linux/Mac
```

2. Instalar las dependencias:
```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
pip install pandas numpy matplotlib seaborn scikit-learn requests pyyaml networkx
```
Estas se encuentran comentadas al comienzo de los notebooks .ipynb

## Estructura del Proyecto

```
TrabajoTesis/ 
├── data/                                       # Directorio para datos de entrada
│   ├── CAIDA_AS_Relationships/                # Dataset de relaciones AS de CAIDA
│   ├── RIBs/                                  # Archivos RIB sanitizados (rutas BGP)
│   │   └── sanitized_rib_{mes}_{año}.txt    
│   ├── peeringdb/                             # Datos de PeeringDB
│   │   ├── json/                              # Archivos JSON descargados
│   │   └── attr/                              # Atributos procesados por mes
│   └── dgl_graph/                             # Grafos DGL generados
│       ├── {año}_peeringdb/                   # Grafos con atributos de PeeringDB
│       ├── {año}_degree/                      # Grafos con atributos de grado
│       └── {año}_random/                      # Grafos con atributos aleatorios
│
├── modules/                                    # Implementaciones de modelos y clases auxiliares
│   ├── gnn_models.py                          # Definición de modelos GNN (GCN, GraphSAGE, GAT)  
│   ├── gnn.py                                 # Clase principal para trabajar con grafos DGL
│   └── graph.py                               # Funciones para crear grafos desde RIBs
│
├── results/                                    # Resultados generados (métricas, gráficos)
│   ├── bgp2vec/
│   └── deepWalk/
│
├── utils.py                                    # Funciones auxiliares (conversión de fechas, etc.)
│
├── create_as_attr.ipynb                       # Notebook 1: Descarga y procesa atributos de PeeringDB
├── create_graph.ipynb                         # Notebook 2: Crea grafos DGL desde RIBs
├── AS_relationship_inference_gnn.ipynb        # Notebook 3: Entrena GNNs e infiere relaciones AS
├── train_embeddings.ipynb                     # Notebook alternativo: Genera embeddings
├── extra.ipynb                                # Experimentos adicionales
└── README.md                                  # Este archivo
```

Se puede indicar el path de la carpeta data/ en caso de estar ubicada fuera del repositorio.

---

### Flujo de Trabajo Detallado

#### 1. Procesamiento de Atributos de Nodos (`create_as_attr.ipynb`)

Este notebook descarga y procesa datos de **PeeringDB** para extraer atributos de los Sistemas Autónomos:

- **Descarga**: Obtiene dumps de PeeringDB desde el repositorio de CAIDA.
- **Extracción de atributos**:
  - Numéricos: `ix_count`, `fac_count`, `info_prefixes4`, `info_prefixes6`
  - Categóricos: `policy_general`, `policy_locations`, `info_traffic`, `info_scope`, `info_type`, etc.
- **Normalización**: Aplica transformación logarítmica y normalización a atributos numéricos.
- **One-hot encoding**: Convierte atributos categóricos a vectores.
- **Salida**: Archivos CSV `peeringdb_as_attr_{año}_{mes}.csv` con 69 features por AS

#### 2. Creación de Grafos (`create_graph.ipynb` o función `create_graphs_from_ribs`)

Construye grafos multi-mensuales de la topología de Internet:

**Entrada**: 
- Archivos RIB sanitizados (`sanitized_rib_{mes}_{año}.txt`) con rutas BGP en formato AS-path
- Atributos de nodos procesados (de la etapa anterior)

**Proceso**:
```python
from modules.graph import create_graphs_from_ribs

# Opción 1: Grafos con atributos de PeeringDB (69+ features)
graph = create_graphs_from_ribs(
    data_path=DATA_PATH, 
    year="2024", 
    max_num_routes=1000000, 
    attr="peeringdb"
)

# Opción 2: Grafos con atributos de grado (2 features: in_degree, out_degree)
graph = create_graphs_from_ribs(
    data_path=DATA_PATH, 
    year="2024", 
    max_num_routes=1000000, 
    attr="degree"
)

# Opción 3: Grafos con atributos aleatorios (69 features random)
graph = create_graphs_from_ribs(
    data_path=DATA_PATH, 
    year="2024", 
    max_num_routes=1000000, 
    attr="random"
)
```

**Salida**: 
- `edges.csv`: Conexiones entre ASes (formato: graph_id, src_id, dst_id)
- `nodes.csv`: Nodos con sus features (formato: graph_id, node_id, feat)
- `graphs.csv`: Lista de IDs de grafos (meses procesados)
- `meta.yaml`: Metadatos del dataset para DGL

Cada tipo de atributo se guarda en carpetas separadas:
- `dgl_graph/2024/_peeringdb/`
- `dgl_graph/2024/degree/`
- `dgl_graph/2024/random/`

#### 3. Entrenamiento e Inferencia 

##### 3.1 Enfoque end-to-end (`AS_relationship_inference_gnn.ipynb`)

Entrena modelos GNN con un enfoque end-to-enf para predecir relaciones entre ASes:

**Carga del Dataset**:
```python
from modules.gnn import GNN

gnn = GNN(debug=True)
gnn.load_dataset(
    data_path=INTERNETGRAPH_FILE,
    index_graph=1,  # Seleccionar mes específico
    caida_path=RELATIONSHIPS_FILE,
    simplify=True
)
```

**Etiquetado de Aristas**:
- Usa dataset CAIDA AS Relationships para etiquetar conexiones
- Tipos de relación: **P2P** (peer-to-peer), **P2C** (provider-to-customer), **C2P** (customer-to-provider)

**Modelos Disponibles**:
- **GCN** (Graph Convolutional Network)
- **GraphSAGE** (SAmple and aggreGatE)
- **GAT** (Graph Attention Network)

**Entrenamiento**:
```python
from modules.gnn_models import GraphSAGE

model = GraphSAGE(
    in_feats=num_features,
    hidden_feats=128,
    out_feats=64,
    num_layers=3
)

# Entrenar modelo
gnn.train(model, epochs=100, lr=0.001)
```

**Evaluación**:
- Métricas: Accuracy, Precision, Recall, F1-Score
- Matrices de confusión
- Análisis por clase de relación



##### 3.1 Enfoque por partes (`train_embeddings.ipynb` y  `AS_relationship_inference.ipynb`)

 Este enfoque divide el proceso de inferencia de relaciones AS en dos etapas principales:
 
 **1. Generación de Embeddings (`train_embeddings.ipynb`)**
 - Se entrenan modelos GNN (GCN, GraphSAGE, GAT) sobre el grafo de Internet para obtener representaciones vectoriales (embeddings) de los Sistemas Autónomos (AS).
 - Las tareas de entrenamiento pueden incluir:
     - Predicción de enlaces (link prediction) para capturar la estructura del grafo.
     - Predicción de atributos de nodos para incorporar información adicional de los AS.
 - Los embeddings generados se guardan para su uso posterior en la inferencia de relaciones.
 
 **2. Inferencia de Relaciones (`AS_relationship_inference.ipynb`)**
 - Utiliza los embeddings previamente generados como features para entrenar un modelo de clasificación.
 - El objetivo es predecir el tipo de relación entre pares de AS (P2P, P2C, C2P) usando los embeddings y el dataset de relaciones de CAIDA como etiquetas.
 - Se evalúa el desempeño del modelo mediante métricas como accuracy, precision, recall y F1-score.
 - Permite comparar el impacto de diferentes métodos de generación de embeddings y arquitecturas GNN en la tarea de inferencia de relaciones.
 



### Notebooks Adicionales

- **`extra.ipynb`**: Experimentos y análisis exploratorios adicionales





