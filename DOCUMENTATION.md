# Documentación técnica — `gnn-as-relationships`

> Inferencia del tipo de relación entre Sistemas Autónomos (AS) de Internet con Graph Neural Networks (PyTorch Geometric + PyTorch).
> Documento generado a partir de una lectura completa del código. Esta versión corresponde a la rama `migracion-pyg` (código sobre **PyTorch Geometric**); la versión original sobre DGL 1.1.3 se conserva en `main` / tag `v1.0-dgl`. Todas las rutas, nombres de funciones, hiperparámetros y cifras citadas provienen del código o de los outputs guardados en los notebooks.

---

## Versiones del código

| Versión | Dónde | Librería de grafos |
|---|---|---|
| **Nueva (este documento)** | rama `migracion-pyg` | **PyTorch Geometric 2.8** |
| Original | rama principal `main`, congelada en el **tag `v1.0-dgl`** (commit `4aff18d`) | DGL 1.1.3 |

La migración reemplazó únicamente las llamadas a DGL por sus equivalentes en PyG: la organización del repositorio, las clases, los métodos, la lógica de carga y de splits, las pérdidas y los hiperparámetros son los mismos (los splits con igual semilla producen exactamente los mismos tamaños). Para volver a la versión DGL: `git switch main` o `git checkout v1.0-dgl`. Para comparar: `git diff v1.0-dgl migracion-pyg -- modules/`.

## Índice

1. [Visión General y Propósito del Proyecto](#1-visión-general-y-propósito-del-proyecto)
2. [Estructura del Repositorio](#2-estructura-del-repositorio)
3. [Arquitectura y Decisiones de Implementación](#3-arquitectura-y-decisiones-de-implementación)
4. [Guía de Ejecución y Pipeline (Step-by-Step)](#4-guía-de-ejecución-y-pipeline-step-by-step)
5. [Propuestas de Mejora y Deuda Técnica](#5-propuestas-de-mejora-y-deuda-técnica)
6. [Pendientes](#6-pendientes)

---

## 1. Visión General y Propósito del Proyecto

### 1.1 Resumen ejecutivo

Internet está formado por decenas de miles de Sistemas Autónomos (AS) que intercambian tráfico bajo acuerdos comerciales que **no son públicos**. Los dos tipos canónicos son:

| Relación | Significado | Etiqueta en este proyecto |
|---|---|---|
| **P2P** (peer-to-peer) | Ambos AS intercambian tráfico de sus clientes sin pago | `0` |
| **C2P** (customer-to-provider) | El AS origen es *cliente* del AS destino | `1` |
| **P2C** (provider-to-customer) | El AS origen es *proveedor* del AS destino | `2` |
| Desconocida | Arista observada en BGP sin etiqueta de referencia | `-1` |

Conocer estas relaciones (*Type of Relationship*, ToR) es fundamental para modelar el enrutamiento (*valley-free routing*), estudiar la resiliencia de Internet, medir la jerarquía de proveedores y analizar ecosistemas regionales (el proyecto incluye un filtro específico para Chile/LACNIC).

Este repositorio es el código de una tesis (NIC Chile Research Labs / DCC, Universidad de Chile) que aborda el problema como **clasificación de aristas multiclase sobre el grafo AS-level de Internet**:

1. Se construye la topología a partir de tablas de enrutamiento BGP (RIBs de RouteViews / RIPE RIS, vía BGPStream). Desde octubre de 2026 esto lo hace el repositorio hermano [`as-level-internet-graph`](https://github.com/valesteban/as-level-internet-graph).
2. Cada AS (nodo) se enriquece con atributos estructurales (grados, PageRank, k-core) y de negocio (PeeringDB). También lo hace `as-level-internet-graph`.
3. Las aristas se etiquetaron en este repositorio con el dataset de referencia **CAIDA AS-Relationships** (`as-rel.txt.bz2`), mediante `GNN.label_edges_from_caida` (hoy eliminado; ver §3.2.2).
4. Una GNN aprende *embeddings* de los AS y un *decoder* de aristas predice la clase P2P / C2P / P2C.

Se exploran dos familias de experimentos, cada una en su notebook:

* **End-to-end** (`AS_relationship_inference_end_to_end.ipynb`): el encoder GNN y el decoder de aristas se entrenan conjuntamente con la pérdida de clasificación.
* **Por partes** (`AS_relationship_inference_por_partes.ipynb`): primero se aprenden embeddings con una tarea auxiliar (link prediction, regresión de grado/PageRank, DeepWalk, BGP2Vec) y luego un MLP independiente clasifica pares de AS.

### 1.2 Enfoque basado en GNN y representación de las relaciones

**Grafo.** El grafo es un `torch_geometric.data.Data` dirigido y homogéneo:

* **Nodos** = AS, indexados por `node_id` (entero contiguo 0..N-1) con un mapeo `asn ↔ node_id` guardado en el CSV de nodos.
* **Aristas** = adyacencias observadas en AS_PATHs. Cada enlace no dirigido se materializa como **dos aristas dirigidas** `(u→v)` y `(v→u)`. Esto es esencial porque la etiqueta es *direccional*: si `(u→v)` es C2P (`1`), entonces `(v→u)` es P2C (`2`); P2P (`0`) es simétrica. La regla de inversión `{0:0, 1:2, 2:1}` está implementada en `GNN._reverse_relationship_value` y la usan `_make_edges_bidirectional` (al cargar) y `build_tor_dataset` en `modules/gnn.py`.
* `data.x`: matriz de atributos de nodo (85 columnas con los grafos de `as-level-internet-graph`; 72 con el grafo antiguo de marzo 2026; ver §3.2).
* `data.edge_index`: aristas dirigidas `(2, E)`.
* `data.edge_label`: etiqueta `0/1/2/-1` (`torch.long`).
* `data.edge_attr`: peso normalizado de la arista (frecuencia con la que el par aparece en los AS_PATH), forma `(E, 1)`.
* `data.train_mask | val_mask | test_mask`: máscaras booleanas de split (tamaño E). El split de nodos de la regresión nodal no se guarda en `data`: lo devuelve `split_nodos` en el notebook por partes.

**Modelo.** Se sigue el patrón *encoder–decoder* para aristas:

```
h = Encoder(G, X)               # GCN / GraphSAGE / GAT  →  h ∈ R^{N×d}
logits_uv = Decoder(h_u, h_v[, w_uv])   # MLP | Bilinear | DotProduct → R^{3}
```

El *message passing* del encoder agrega información de la vecindad de cada AS (quiénes son sus vecinos, cuán centrales son, qué tipo de red declaran en PeeringDB), lo que codifica implícitamente la posición jerárquica del AS. El decoder es **asimétrico** respecto al orden `(u, v)` (concatenación `[h_u ‖ h_v]` en el MLP, forma bilineal `h_uᵀ W_c h_v` en el Bilinear), de modo que puede distinguir C2P de P2C. El `DotProductPredictor` es simétrico y se usa solo para link prediction (tarea binaria).

### 1.3 Stack tecnológico

Versiones del entorno `env312` de la máquina de desarrollo, fijadas en `requirements.txt`. La columna `env310` corresponde al entorno anterior, con el que se obtuvieron los resultados previos. La versión DGL original corría con `DGL 1.1.3 | PyTorch 2.3.0+cu121`.

| Componente | `env312` (actual) | `env310` (anterior) | Uso |
|---|---|---|---|
| Python | 3.12.3 (kernel Jupyter `env312`) | 3.10.12 | Todo |
| PyTorch | 2.8.0 (+cpu; GPU: +cu128) | 2.3.0 (+cu121 / +cpu) | Modelos, entrenamiento |
| **PyTorch Geometric** | **2.8.1** (+ `pyg_lib` 0.6.0, `torch_scatter`, `torch_sparse`, `torch_cluster`) | 2.8.0.post1 (+ `pyg_lib` 0.4.0) | `Data`, `GCNConv`, `SAGEConv`, `GATConv`, `LinkNeighborLoader`, `ClusterData`/`ClusterLoader`, `Node2Vec` |
| NetworkX | 3.7 | 3.4.2 | Pipeline legado `modules/graph.py` |
| pandas / numpy | 3.0.6 / 2.5.4 | 2.3.3 / 2.2.6 | I/O CSV, normalización |
| scikit-learn | 1.9.1 | 1.7.2 | Métricas (F1, matriz de confusión, ROC-AUC), `train_test_split`, `compute_class_weight` |
| gensim | 4.4.0 | 4.4.0 | Word2Vec para BGP2Vec (`modules/bgp2vec.py`) |
| pybgpstream | — | — | Ya no se usa aquí: la descarga de RIBs pasó a `as-level-internet-graph` |
| requests, urllib | 2.34.2 | 2.34.2 | API PeeringDB y LACNIC (`src/`) |
| PyYAML | 6.0.3 | 6.0.3 | `config.yaml` y `modules/graph.py` (legado) |
| tqdm, matplotlib, seaborn, umap-learn, powerlaw, scipy | fijadas en `requirements.txt` | — | Progreso y visualización en notebooks |

torch 2.8 es la última versión con ruedas de las cuatro extensiones de PyG: desde 2.9 no hay `pyg_lib`, y desde 2.12 tampoco `torch_cluster`.

> **Ya no se usa DGL.** Toda la parte de grafos está construida sobre PyG. Las dependencias están en `requirements.txt` (§4.2).

---

## 2. Estructura del Repositorio

```
gnn-as-relationships/
├── README.md                                   # Descripción, ubicación de datos, instalación y flujo de trabajo
├── CLAUDE.md                                   # Contexto resumido del proyecto para Claude Code
├── DOCUMENTATION.md                            # este documento
├── .gitignore                                  # ignora *.csv, *.md, *.json, *.txt, data/*, results/*; excepciones: README.md, CLAUDE.md, DOCUMENTATION.md, requirements.txt (ver §5)
├── config.yaml                                 # Configuración central: rutas, semilla, split e hiperparámetros (§4.2)
├── config_nerwito.yaml                         # Configuración anterior del equipo Windows (rutas F:/; formato incompatible)
├── requirements.txt                            # Versiones fijadas de env312 (versionado desde 2026-11-16)
│
├── AS_relationship_inference_end_to_end.ipynb  # ★ Entrenamiento end-to-end (encoder GNN + decoder de aristas)
├── AS_relationship_inference_por_partes.ipynb  # ★ Enfoque en dos etapas: embeddings + clasificador de aristas
│
├── modules/                                    # Código reutilizable importado por los notebooks
│   ├── gnn.py            # ★ Clase GNN: carga CSV→Data (PyG), splits, dataset ToR, features aleatorias, poda
│   ├── gnn_models.py     # ★ Encoders GCN/GraphSAGE/GAT (2L, 3L, *Sample2L) y decoders MLP/Bilinear/DotProduct
│   ├── experiment.py     # set_seed (reproducibilidad) y save_run (métricas, pesos e historial de cada modelo)
│   ├── metis_windows.py  # METIS para ClusterData en Windows vía metis.dll (sin conectar a los notebooks)
│   ├── graph.py          # Pipeline LEGADO: clase Graph (NetworkX) → dataset multi-grafo mensual (meta.yaml, graphs.csv)
│   ├── bgp2vec.py        # RIBCorpus (streaming de AS_PATHs) + BGP2VEC (Word2Vec skip-gram sobre rutas)
│   └── bgp2vec/          # Versión anterior de BGP2Vec (copia casi idéntica de bgp2vec.py)
│       ├── bgp2vec.py    #   - usa len(self.routes): incompatible con iteradores
│       ├── oix_utils.py  #   - parsers del formato OIX (Oregon RouteViews) del paper BGP2Vec original
│       ├── bgp2vec.word2vec  # modelo Word2Vec entrenado (binario)
│       └── ToR_bgp2vec.ipynb # notebook Keras/TensorFlow del baseline BGP2Vec + NN
│
├── scripts/                                    # Vacío: correr_experimento.py se eliminó en d1495d9
│
├── src/                                        # Utilidades de datos externos
│   ├── lacnic.py            # Descarga/parseo de delegated-lacnic-extended-latest → ASNs por país
│   ├── peeringdb.py         # Cliente paginado de la API PeeringDB (ix, netixlan, fac, netfac, net) + descarga de dump JSON
│   ├── topology_filter.py   # load_nodes_csv / load_edges_csv / filter_topology
│   ├── utils.py             # month_number_to_name (1..12 → "enero".."diciembre")
│   ├── graph_builder.py     # VACÍO (0 bytes)
│   └── __pycache__/utils.py # Archivo fuente extraviado dentro de __pycache__ (versión antigua de utils con imports de sklearn/torch)
│
├── notebooks/                                  # Análisis exploratorio y de resultados (no entrenan modelos)
│   ├── analisis_ribs_recolectadas.ipynb     # Calidad de RIBs de distintas duraciones (30 min … 24 h)
│   ├── analisis_grafo_internet.ipynb        # Grados, homofilia, simetría, implicaciones para GNN
│   ├── analysis_as_topology.ipynb           # Topología AS-level: distancias, grados, centralidad, k-core
│   ├── resultados_experimentos.ipynb        # Tablas de métricas y tiempos a partir de los metrics.json
│   ├── attribute_inspection.ipynb           # Tabla resumen de atributos, correlación Spearman, distribuciones por clase
│   ├── validar_normalizacion_atributos.ipynb# Verifica que todos los atributos estén en [0,1]
│   ├── peeringdb_coverage_bias.ipynb        # Sesgo de cobertura de PeeringDB por grado/tier
│   ├── AS_embeddings_visualization.ipynb    # t-SNE / UMAP de embeddings por caso
│   ├── tsne_embeddings.ipynb                # t-SNE comparativo (GNN vs DeepWalk vs BGP2Vec)
│   └── results/*.pdf                        # Figuras exportadas (pdb_*, tsne_embeddings)
│
└── (build/, src/, modules/graph.py, modules/bgp2vec/, notebooks/*.ipynb de análisis y notebooks/results/ se eliminaron el 2026-11-16; siguen en git)
                                                # El .tex NO está en el repo; el .out revela la estructura del paper
```

**Estado de cada componente**

| Componente | Estado | Comentario |
|---|---|---|
| `modules/gnn.py`, `modules/gnn_models.py`, `modules/experiment.py` | **Activo** | Núcleo del proyecto |
| `scripts/1..4_*.py`, `scripts/create_tor_dataset.py` | **Eliminados** (commit `4fcb754`) | La construcción del grafo pasó a `as-level-internet-graph` y el etiquetado CAIDA a `GNN.label_edges_from_caida` (hoy también eliminado, §3.2.2). Siguen en el historial de git y en `main` / `v1.0-dgl`. |
| `data/` (grafo de marzo 2026) | **Eliminado** del repo | Copia en `/media/vale/KINGSTON/TESIS/data_2026/` (ver §4.3) |
| `src/*.py` | Sin uso | Utilidades que usaban los scripts eliminados |
| `modules/bgp2vec.py` | Activo (enfoque por partes) | |
| `config.yaml` | **Activo** | Lo leen los dos notebooks principales (§4.2) |
| `config_nerwito.yaml` | **Eliminado** (2026-11-16) | Configuración del equipo Windows (`845347d`), con el formato anterior. Sigue en el historial de git |
| `scripts/correr_experimento.py` | **Eliminado** (commit `d1495d9`) | Ejecutaba los notebooks por lotes sobre un experimento. Sigue en el historial de git |
| `modules/metis_windows.py` | Sin conectar | `activar_metis_windows(dll)`; ningún notebook lo llama |
| `modules/graph.py` | **Eliminado** (2026-11-16) | Formato multi-grafo mensual; ya no lo usaban los notebooks. Sigue en el historial de git |
| `modules/bgp2vec/` | **Eliminado** (2026-11-16) | Duplicado de `modules/bgp2vec.py` + baseline Keras. Sigue en el historial de git |
| `src/`, `build/` | **Eliminados** (2026-11-16) | Pipeline de datos antiguo y artefactos LaTeX. Siguen en el historial de git |
| `src/__pycache__/utils.py` | Basura versionada | Fuente dentro de `__pycache__`; los `.pyc` se quitaron del repo en `f327a24` |

---

## 3. Arquitectura y Decisiones de Implementación

### 3.1 Modelos (`modules/gnn_models.py`)

#### 3.1.1 Decoders de arista

| Clase | Entrada | Cálculo | Salida | Simetría |
|---|---|---|---|---|
| `MLPPredictor(h_dim, n_classes, edge_dim=1, drop=0.3)` | `h_u`, `h_v`, opcional `edge_attr` | `z = [h_u ‖ h_v ‖ w_uv]` → `Linear(2h+edge_dim, h)` → ReLU → Dropout → `Linear(h, C)` | `(E, C)` (o `(E,)` si `C=1`) | Asimétrico (orden de concatenación) |
| `BilinearPredictor(h_dim, n_cls)` | `h_u`, `h_v` | `score_c = h_uᵀ W_c h_v + b_c`, con `W ∈ R^{C×F×F}` inicializado `randn` (`torch.einsum("ef,cfk,ek->ec")`) | `(E, C)` | Asimétrico (W_c no es simétrica) |
| `DotProductPredictor()` | `h_u`, `h_v` | `(h_u * h_v).sum(-1)` | `(E,)` | Simétrico → solo link prediction |

Los tres reciben `edge_index` `(2, E)` y los embeddings `h` `(N, F)` e indexan `h[edge_index[0]]`, `h[edge_index[1]]`, por lo que funcionan tanto sobre el grafo completo como sobre `edge_label_index` de un `LinkNeighborLoader` o el `edge_index` de un cluster.

> Nota: `MLPPredictor` solo concatena `edge_attr` si se le pasa (`edge_attr is not None`); si el modelo se construyó con `edge_dim=1` y el grafo no tiene `edge_feat`, la dimensión de entrada de `fc1` no coincidirá. Desde `c065a4f`, los encoders full-batch le pasan su `drop` (antes usaba siempre 0.3).

#### 3.1.2 Encoders full-batch (3 capas y 2 capas)

Todas las clases exponen la misma interfaz: `encode(x, edge_index)`, `decodeMLP(edge_index, h, edge_attr=None)`, `decodeBilinear(edge_index, h)`, `decodeDotProduct(edge_index, h)` y `forward(x, edge_index)` (= `encode` + `regressor: Linear(out, 1)`, usado para regresión nodal en el enfoque por partes). Las clases 2L no definen `regressor` (bug 4, §5.1): `node_regression_train` se lo agrega antes de entrenar.

| Clase | Capas PyG | Agregación / atención | Activación entre capas | Self-loops | Dim. de salida del embedding |
|---|---|---|---|---|---|
| `GCN3L(in, hid, out, out_feats_mlp=1, drop=0.3, edge_dim=0)` | `GCNConv×3` | Normalización simétrica de GCN | ReLU + Dropout (no tras la última) | `add_self_loops=True` (por defecto en `GCNConv`) | `out` |
| `GraphSAGE3L(...)` | `SAGEConv×3` (`aggr='mean'`) | Media de vecinos + proyección propia | ReLU + Dropout | No (SAGE no requiere) | `out` |
| `GAT3L(..., num_heads=4)` | `GATConv×3` | Atención multi-cabeza; `concat=True` concatena cabezas tras cada capa | ReLU + Dropout | `add_self_loops=True` (por defecto) | `out × num_heads` |
| `GCN2L`, `GraphSAGE2L`, `GAT2L` | Igual con 2 capas | Igual | Igual | Igual | `out` (GAT: `out×heads`) |

Detalles relevantes:

* `hidden_feats` es el ancho intermedio; en las GAT, la capa `k+1` recibe `hidden × num_heads`.
* Los decoders MLP/Bilinear se instancian dentro del encoder con `out_feats` (o `out_feats × num_heads` en GAT) y `out_feats_mlp` = nº de clases.
* En los notebooks, GCN y GraphSAGE usan `hidden_dim`/`out_dim` (256/128) y GAT usa `gat_hidden_dim`/`gat_out_dim` (64/32) con `num_heads` cabezas (hoy 1). La dimensión de embedding es `out_dim` (128), o `gat_out_dim × num_heads` (32) en GAT.

#### 3.1.3 Encoders con muestreo (`GCNSampler2L`, `GraphSAGESample2L`, `GATSample2L`)

Diseñados para entrenamiento mini-batch. En PyG tanto `LinkNeighborLoader` (neighbor sampling) como `ClusterLoader` (ClusterGCN) entregan un `Batch` con `x` y `edge_index` locales, por lo que `encode(x, edge_index)` es idéntico al de los encoders full-batch:

```python
h = relu(conv1(x, edge_index)); h = conv2(h, edge_index)
```

* No hay dropout en estas variantes; `GCNConv`/`GATConv` agregan self-loops por defecto, lo que también cubre los nodos sin vecinos de entrada de los subgrafos muestreados (el `allow_zero_in_degree` de DGL ya no es necesario).
* Exponen `decodeMLP` y `decodeBilinear` (este último desde `c065a4f`); `MLPPredictor` se crea con `edge_dim=1` por defecto.
* En `GATSample2L` las cabezas quedan concatenadas tras cada capa (`concat=True`): `conv2` recibe `hidden × heads` y el `MLPPredictor` `out × heads`.

### 3.2 Pipeline de procesamiento de datos

La construcción y el enriquecimiento del grafo se hacen en el repositorio hermano [`as-level-internet-graph`](https://github.com/valesteban/as-level-internet-graph). Este repositorio etiquetaba las aristas con CAIDA (método hoy eliminado, §3.2.2) y entrena. El pipeline anterior (`scripts/1..4_*.py`, que producía `nodes_rib_*_enriched_tesis.csv` / `edges_rib_*_enriched_tesis.csv`) se eliminó en el commit `4fcb754`. Sigue disponible en el historial de git y en `main` / `v1.0-dgl`.

```
 RouteViews / RIPE RIS (BGPStream, todos los recolectores, ventana de 15 min)
          │  as-level-internet-graph:  python as_graph.py … ;  scripts/enriquecer_nodos.py
          ▼
 GRAFO_DIR = /media/vale/KINGSTON/as-level-internet-graph/experimentos/experimento<N>/grafos/<año>/<mes>/<AAAAMMDDTHHMM>/
   ├─ as_features.csv   node_id, asn, +85 features en [0,1]   (node_id = fila)
   ├─ gnn_edges.csv     src_id, dst_id, src_asn, dst_asn, weight, weight_raw   (1 fila por enlace no dirigido)
   └─ stats.json, as_nodes.csv, as_edges.csv, as_graph.graphml, …
          │  GNN.label_edges_from_caida(gnn_edges.csv, CAIDA AAAAMM01.as-rel.txt.bz2, output_csv=…)   (eliminado)
          ▼
   ├─ gnn_edges_caida.csv     gnn_edges.csv + relationship {0,1,2,-1}   (mismas filas, mismo orden)
   └─ etiquetas_caida.json    resumen del etiquetado
          │  GNN.load_dataset(as_features.csv, gnn_edges_caida.csv)
          ▼
 torch_geometric.data.Data  x (N×85) · edge_index (2×E) · edge_label (E) · edge_attr (E×1)
          │  GNN.split_edges_classification()  /  split_edges_link_prediction()  /  split_nodos() (notebook por partes)
          ▼
 data.train_mask | val_mask | test_mask  (+ gnn.train_eids / val_eids / test_eids)
          │  notebooks (training loop)
          ▼
 métricas · matrices de confusión · embeddings (.pt) · pesos (.pth)
```

#### 3.2.1 Grafo de `as-level-internet-graph`

Cada grafo vive en su propia carpeta `GRAFO_DIR`, como se muestra arriba. El grafo cambia entre experimentos, así que los notebooks lo eligen con **una sola variable, `GRAFO_DIR`**, y derivan de ella todo lo demás. El grafo con el que se probó esta versión es `experimento7/grafos/2026/09/20260916T0800`:

* 57 de 63 recolectores (RouteViews y RIS), ventana de 15 min.
* 87 796 nodos y 668 426 enlaces.

Qué es cada experimento está en `experimento.json` / `DESCRIPCION.md` de cada carpeta y en `EXPERIMENTOS.md`, en la raíz del disco.

* **Nodos (`as_features.csv`)**: `node_id` (= número de fila), `asn` y 85 features en `[0,1]`, sin NaN:
  * Numéricas (`log1p + min-max` dentro de cada grafo): `degree`, `weighted_degree`, `transit_degree`, `pagerank`, `k_core`, `info_prefixes4`, `info_prefixes6`, `ix_count`, `fac_count`, `antiguedad_anios`, `policy_ratio`.
  * Binarias: `in_peeringdb`, `info_unicast`, `info_multicast`, `info_ipv6`, `info_never_via_route_servers`, `has_irr_as_set`.
  * One-hot con **vocabulario fijo**, con `__desconocido` y `__otro`: `info_types__*` (12), `policy_general__*` (6), `info_traffic__*` (20), `info_scope__*` (11), `policy_contracts__*` (5), `info_ratio__*` (7), `policy_locations__*` (7).

  El vocabulario fijo hace que las columnas sean las mismas en todas las fechas. Los grafos sin grado de tránsito (experimentos 1–6) tienen 84 features.
* **Aristas (`gnn_edges.csv`)**: una fila por enlace no dirigido, siempre con `src_asn < dst_asn`, sin duplicados ni bucles. `weight` es la frecuencia del par en los AS_PATH con `log1p + min-max`, y `weight_raw` es la cuenta cruda (`load_dataset` la ignora). No trae etiqueta.

El detalle de la construcción (normalización de AS_PATH, PeeringDB, features) está en el README de `as-level-internet-graph`.

#### 3.2.2 Etiquetado CAIDA (`GNN.label_edges_from_caida`)

> **`label_edges_from_caida` se eliminó de `modules/gnn.py` (cambio sin commit; sigue en `HEAD`).** Quedan `_read_caida_relationships` y `caida_file_for_graph`, que ya no se llaman desde el código, y los notebooks ya no etiquetan solos. Lo que sigue describe cómo se generaron los `gnn_edges_caida.csv` existentes.

CAIDA publica un snapshot de AS-Relationships (serial-1) el día 1 de cada mes. Se usa el del **mes del grafo**:
* `GNN.caida_file_for_graph(GRAFO_DIR, CAIDA_DIR)` lee la fecha del nombre de la carpeta (`AAAAMMDDTHHMM`) y devuelve `CAIDA_DIR/AAAAMM01.as-rel.txt.bz2`.
* Si el archivo no existe, lanza `FileNotFoundError` con la URL de descarga.
* `CAIDA_DIR` (`rutas.caida_dir`) actual: `/media/vale/KINGSTON/LAB/CAIDA/CAIDA_AS_Relationships/Serial_1`. Los `gnn_edges_caida.csv` de 2026 se etiquetaron con `/media/vale/KINGSTON/TESIS/data_2026/` (hay `20260301` y `20260901`).

`label_edges_from_caida(edges_csv, caida_file, output_csv=None)`:

1. `_read_caida_relationships`:
   * lee `.bz2`, `.gz` o texto plano;
   * `p|c|-1` (p es proveedor de c) y `a|b|0` (pares) se guardan como `{(asn_menor, asn_mayor): etiqueta vista desde asn_menor}`;
   * otros tipos de relación se ignoran.
2. Cada fila se etiqueta según su sentido `src_asn → dst_asn`. Si `src_asn > dst_asn`, la etiqueta se invierte con `_reverse_relationship_value`. Los enlaces que no están en CAIDA quedan en `-1`.
3. **No se agregan aristas ni nodos**: los enlaces de CAIDA ausentes del grafo solo se cuentan. **No se reordenan las filas**, porque el orden determina los splits.
4. Con `output_csv`:
   * escribe el CSV con todas las columnas originales más `relationship` (`int8`);
   * escribe, en la misma carpeta, `etiquetas_caida.json` con `p2p`, `c2p`, `p2c`, `sin_etiqueta`, `aristas`, `caida_total`, `caida_en_grafo` y `caida_ausentes`.

   Por convención la salida es `GRAFO_DIR/gnn_edges_caida.csv`, en el disco externo. `gnn_edges.csv` no se modifica.

Resultado para `experimento7 / 20260916T0800` con CAIDA `20260901` (525 848 relaciones):

| Medida | Valor |
|---|---|
| Enlaces etiquetados | 493 982 de 668 426 (73.9 %) |
| `0` (P2P) | 329 589 |
| `1` (C2P) | 28 715 |
| `2` (P2C) | 135 678 |
| `-1` (sin etiqueta) | 174 444 |
| Relaciones CAIDA ausentes del grafo | 31 866 |

Que haya más P2C que C2P se explica porque cada fila va del ASN menor al mayor, y el menor suele ser el proveedor. Tras la bidireccionalización, C2P y P2C quedan iguales.

#### 3.2.3 Grafo antiguo de marzo 2026

El grafo usado hasta septiembre de 2026 (RIB de 24 h del 1 de marzo, pipeline eliminado) sigue en `/media/vale/KINGSTON/TESIS/data_2026/`:
* `nodes_rib_20260301_0000_to_20260302_0000_enriched_tesis.csv`: 72 features, entre ellas `weight`, `degree_centrality`, `betweenness_centrality`, `eigenvector_centrality`, `PageRank` y one-hot de PeeringDB.
* `edges_…_enriched_tesis.csv`: ya etiquetado; 734 381 filas.

`load_dataset` lo sigue leyendo sin cambios: es la prueba de regresión de §3.2.5. Su distribución de `relationship` es `0`: 340 512 · `1`: 159 963 · `2`: 159 963 · `-1`: 73 943.

#### 3.2.4 Carga a PyG (`modules/gnn.py :: GNN`)

`load_dataset(nodes_csv, edges_csv, add_reverse_edges=True)`:

1. `_make_edges_bidirectional`: concatena el DataFrame con su copia invertida (etiqueta invertida con `_reverse_relationship_value`) y `drop_duplicates(['src_id','dst_id'], keep='first')` → la fila original prevalece. Con `gnn_edges_caida.csv` de 2026-09-16 pasa de 668 426 enlaces a **1 336 852 aristas** (87 796 nodos). Con el grafo antiguo de marzo pasa de 734 381 a 807 496 (79 504 nodos).
2. `_normalize_relationship_column`: acepta `'P2P'/'C2P'/'P2C'` o `'0'/'1'/'2'/'-1'`, fuerza `int64` y lanza `ValueError` ante valores desconocidos.
3. `Data(edge_index=torch.stack([src, dst]), num_nodes=len(df_n))` — asume que `src_id/dst_id` son índices de fila de `df_n`.
4. `x` = todas las columnas excepto `node_id, asn, country` (`float32`); `edge_label` (`long`); `edge_attr` = `weight` `(E,1)`.
5. Guarda `self.asn_to_node_id` (lo usa `AS_relationship_evaluacion_otro_grafo.ipynb`).

`load_dataset_only_cntrality_attr(...)`: igual, pero las features son solo las columnas estructurales presentes en el CSV, **estandarizadas con z-score**. En el formato nuevo son `degree`, `weighted_degree`, `transit_degree`, `pagerank` y `k_core`; en el antiguo, `weight`, `PageRank`, `degree_centrality`, `betweenness_centrality` y `eigenvector_centrality`. El z-score deja media 0 y desviación 1; las columnas constantes usan std=1. Comentario en el código: sin esto la GNN colapsa por las escalas dispares.

`_fill_labels_from_caida_stream_fast(caida_file)`, alternativa in-memory a `label_edges_from_caida` que agregaba aristas y nodos faltantes, se eliminó el 2026-11-24 porque nada la llamaba (sigue en git).

Otras utilidades: `add_random_features(dim, std, seed, mode∈{zscore,minmax,uniform})` (control experimental), `remove_low_degree_nodes(degree, iterations)` (poda con `torch_geometric.utils.subgraph` y remapeo de `asn_to_node_id` con los ids originales conservados).

#### 3.2.5 Estrategias de split

| Método | Tarea | Cómo evita fugas | Salida |
|---|---|---|---|
| `split_edges_classification(train_size=0.7, val_size=0.15, seed, balance_mode)` | Clasificación de aristas | Agrupa eids por **par no dirigido** `(min(u,v), max(u,v))` → las dos direcciones caen en el mismo split. Estratifica por *firma* de etiquetas del par (`(0,0)`, `(1,2)`, `(2,)`…) y reparte cada estrato con las proporciones dadas. `balance_mode="strict_equal"` submuestrea cada split al mínimo por clase | `edata[*_mask]`, `self.train_eids/val_eids/test_eids` |
| `split_edges_kfold(k=5, fold=0, val_size=0.15, seed=0, balance_mode="proportional", return_eids=False, store_eids=True)` | Clasificación de aristas, validación cruzada K-fold (`comunes.validacion.tipo: kfold`, §3.3.1) | Misma unidad y estratificación que `split_edges_classification`: pares no dirigidos agrupados por firma. Dentro de cada firma, los pares se barajan con `seed`, con la misma permutación para todos los folds, y se reparten con `fold_id = i % k`. **Test** = los pares del fold `fold` (≈ 1/k). **Val** = fracción `val_size` del resto, por firma y barajada con `seed + fold`, para early stopping. **Train** = lo demás. Los k test son disjuntos y su unión cubre todas las aristas etiquetadas. Valida `k ≥ 2`, `0 ≤ fold < k` y `0 ≤ val_size < 1` | Igual que `split_edges_classification` |
| `split_edges_link_prediction(train_size=0.8, val_size=0.0, seed=42, balance_mode=None, train_ratio=None)` | Link prediction | Agrupa las aristas por enlace no dirigido `{u,v}` y baraja enlaces (sin fuga entre direcciones); `train_g` solo contiene los enlaces de train; negativos uniformes `u≠v` sin arista en ninguna dirección del grafo completo, únicos, disjuntos entre splits, con dirección aleatoria y tantos como positivos. Acepta `**comunes.split` (`balance_mode` se ignora) y `train_ratio` como alias | `self.train_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g` |
| `split_nodos(num_nodes, train_size, val_size, seed)` (función del notebook por partes, celda 10) | Regresión nodal | `randperm` de nodos con `Generator` sembrado | dict `{"train", "val", "test"}` de máscaras de nodos |

`split_graph_nodes`, `split_edges_classification_leaky` y `split_edges_classification_v0` se eliminaron de `modules/gnn.py` (cambio sin commit).

**`balance_mode` en cada experimento.** `proportional` conserva la mezcla natural de clases (P2P 0,660 · C2P 0,170 · P2C 0,170) y `strict_equal` recorta cada split a igual cantidad de aristas por clase. Desde 2026-10-17 todo se corre en `proportional` y el desbalance se compensa en el entrenamiento con la cross-entropy ponderada por clase (`class_weights: true`, pesos `n / (C · conteo)` calculados solo con train).

| Período | Notebook | `balance_mode` |
|---|---|---|
| hasta 2026-04-21 | `AS_relationship_inference*.ipynb` (versiones iniciales) | no se puede determinar: no pasaban `balance_mode` |
| 2026-04-21 (`75c0c6c`) | end-to-end, casos 0 a 4 | `strict_equal` en los 5 |
| 2026-04-21 (`d8779aa`) a 2026-10-15 (`f6b24ee`) | end-to-end | **caso 0: `proportional`** (comentario «usa todos los datos; class_weights compensan»); **casos 1, 2, 3 y 4: `strict_equal`** |
| desde 2026-10-17 (`ecdf050`) | end-to-end, todos los casos | `proportional`, de `comunes.split.balance_mode` |
| desde 2026-10-21 (`903bdb7`) | por partes, Etapa 2 | `proportional` (`**P2E_SPLIT`); antes usaba `build_tor_dataset`, con split propio |

Verificado el 2026-11-24 con el disco: las 7 `config.yaml` de `ejecuciones/` dicen `proportional`, y en ninguna de las 781 corridas legibles de `resultados_gnn/experimento7_20250916T0800/` ni de `prueba e2e/` el test está balanceado. Los tests de los casos 0, 1, 2 y 4 salen con 66/17/17 % (138 810 aristas); el del caso 3 (ClusterGCN) con 41–44/28–29/28–29 % por efecto de los clusters, no de `strict_equal`. No queda en disco ningún resultado con `strict_equal`: eran del grafo antiguo de marzo y estaban en `results/`, ya eliminada.

Cifras con `experimento7 / 20260916T0800` + CAIDA `20260901`: 987 964 aristas dirigidas etiquetadas (P2P 659 178 · C2P 164 393 · P2C 164 393); con `proportional` 0.7/0.15 y `seed=42` → train 691 574 / val 148 194 / test 148 196. En link prediction, con `train_size=0.7, val_size=0.15, seed=42` (todas las aristas, también las `-1`) → 935 796 / 200 526 / 200 530 positivos, con igual número de negativos.

Cifras de K-fold con `experimento7 / 20250916T0800` (el grafo de `config.yaml`): 85 522 nodos, 1 282 010 aristas y 925 448 aristas dirigidas etiquetadas. El holdout `proportional` 0.7/0.15 con `seed=42` da train 647 790 / val 138 812 / test 138 810. `split_edges_kfold(k=5, val_size=0.15, seed=42)` da por fold train 629 282 / val 111 048 / test 185 082; el fold 0 tiene 2 aristas más en test y 2 menos en train. La proporción de clases en test es la global en todos los folds (P2P 0,660 · C2P 0,170 · P2C 0,170). Con K-fold, las proporciones efectivas son ≈ 68 / 12 / 20, porque val es el 15 % de lo que no es test.

Cifras del grafo antiguo de marzo (outputs guardados): 660 438 aristas etiquetadas; con `proportional` 0.7/0.15 → train 462 306 / val 99 064 / test 99 068; con `strict_equal` 0.8/0.15 → train 383 910 (127 970 por clase) / val 71 982 / test 23 997 (7 999 por clase).

### 3.3 Esquemas de entrenamiento

#### 3.3.0 Reproducibilidad y guardado (`modules/experiment.py`)

**Semilla.** `set_seed(seed=42, deterministic=True)`:
* fija `random`, `numpy`, `torch` (CPU y CUDA) y PyG (`torch_geometric.seed_everything`);
* con `deterministic=True`, además fija `cudnn.deterministic=True` y `cudnn.benchmark=False`, activa `torch.use_deterministic_algorithms(True, warn_only=True)` y define `CUBLAS_WORKSPACE_CONFIG`.

Los dos notebooks toman `SEED` de `comunes.seed` (42) en la celda de configuración y la usan en todos los splits (`seed=SEED`). Además, cada función de entrenamiento llama a `set_seed(seed)` al inicio, antes de crear el modelo:
* end-to-end: `entrenar_modelo`, `entrenar_modelo_sampler` y `entrenar_modelo_cluster`;
* por partes: `link_prediction_train`, `node_regression_train`, `deepwalk_train`, `entrenamiento_sin_batches` y `entrenamiento_con_batches` (este último además ordena los mini-batches con un `Generator` sembrado). `bgp2vec_train` no llama a `set_seed`: BGP2Vec usa su propia semilla (ver abajo).

Así cada modelo parte siempre del mismo estado: pesos iniciales, dropout y orden de los mini-batches de `LinkNeighborLoader`/`ClusterLoader` son los mismos. Esto vale aunque se ejecute solo esa celda o se comenten otros modelos.
* **Comprobado en CPU:** dos procesos distintos y una segunda ejecución de la celda sin repetir el split dan métricas idénticas (Caso 0 y Caso 2).
* **En GPU** algunas operaciones scatter de PyG no son deterministas: `warn_only` solo avisa, y los resultados pueden variar levemente.
* **BGP2Vec** usa `seed=7` y `workers=1` en `modules/bgp2vec.py`. gensim además depende de `PYTHONHASHSEED`.

**Guardado.** `save_run(results_dir, caso, modelo, decoder, seed=…, y_true, y_pred, class_names, state_dict, history, config, extra)` se llama después de evaluar cada modelo en test. Escribe en `results_dir/<caso>/<modelo>_<decoder>_seed<SEED>_<config_id>/`:
* `metrics.json`: accuracy, macro-F1, F1 ponderado, reporte por clase, matriz de confusión, `config`, semilla y `extra` (`clase`, `tiempo_s`, `epocas` y `mejores_val`, la mejor época de cada métrica de validación);
* `model.pth`: los pesos de la mejor época en validación;
* `history.csv`: las curvas por época.

También agrega una fila a `results_dir/resumen.csv`. Se usa en los dos notebooks:
* **End-to-end:** `results_dir = RESULTS_DIR_E2E = <experimento.salida>/<experimento.nombre>/end_to_end`, con `config = GLOBAL_CONFIG_E2E`. Los valores de `caso` son `caso0`, `caso1`, `caso2_neighbor`, `caso3_cluster` y `caso4_random`.
* **Por partes (clasificador de la Etapa 2):** `results_dir = RESULTS_DIR_P2E = …/por_etapas`, con `config = CONFIG_CLASIFICADOR`, `decoder = "<decoder del embedding>-<sin_batches|con_batches>"` y `caso` = `caso0` … `caso5`. `extra` agrega `clase_embedding`, `config_embedding`, `tiempo_embedding_s`, `tiempo_total_s` y, en link prediction, `auc_test_link_prediction`.

Repetir una corrida con la misma semilla sobrescribe su carpeta, porque el resultado es el mismo.

**Identificador de configuración.** `<config_id>` es un hash de 6 caracteres de `config` y de `extra["clase"]` (la clase del modelo, p. ej. `GAT2L`, que fija el número de capas); en el clasificador por partes también de `extra["config_embedding"]` y `extra["clase_embedding"]` (los del modelo que generó el embedding). Re-correr con la misma configuración sobrescribe la carpeta (el resultado es idéntico); cambiar hiperparámetros o capas crea una carpeta nueva al lado, así que las corridas no se pierden. Como la salida depende de `experimento.nombre` y no del grafo, para no mezclar grafos distintos hay que cambiar `experimento.nombre`. `resumen.csv` incluye `config_id`. La función es `modules.experiment.config_id`.

**Tiempos.** Cada función de entrenamiento mide su duración con `time.perf_counter`, imprime `[TIEMPO]` y devuelve `tiempo_s` y `epocas` en `extra`; `resumen.csv` los incluye como columnas `tiempo_s`, `tiempo_total_s` y `epocas`. En por partes, `imprimir_tiempo` mide además la carga y los splits de cada caso, y `tabla_tiempos` muestra por modelo `embedding_s`, `sin_batches_s`, `con_batches_s` y `total_s`. `GNN.timer` y los archivos `<embedding>.tiempo.json` ya no existen.

#### 3.3.1 End-to-end (`AS_relationship_inference_end_to_end.ipynb`)

Configuración global: `global_config_endtoend` de `config.yaml`, cargada como `GLOBAL_CONFIG_E2E` (celda 13). Valores actuales:

```yaml
lr: 0.001          hidden_dim: 256      out_dim: 128         epochs: 100      patience: 120
weight_decay: 0.0005   dropout: 0.3     num_heads: 1         scheduler: {T_0: 100, T_mult: 2}
class_weights: true    gat_hidden_dim: 64   gat_out_dim: 32
num_neighbors: 15  batch_size_caso2: 1024   num_parts: 1000  batch_size_caso3: 20
random_features: {dim: 64, std: 0.05, mode: uniform}
```

Todos los casos usan el mismo esquema, implementado en tres funciones (celdas 7–9):
* **Split:** el tipo de validación se elige con `comunes.validacion.tipo` en `config.yaml`; la función auxiliar `hacer_split(gnn, fold)` de la celda 12 llama al método que corresponde.
  * `holdout` (por defecto, y también si falta la sección): `split_edges_classification(seed=SEED, **comunes.split)`, hoy `proportional` 0.7/0.15. Es un solo split; `FOLDS = [None]`.
  * `kfold`: `split_edges_kfold(k=comunes.validacion.k, fold, seed=SEED, val_size=split.val_size, balance_mode=split.balance_mode)` para cada `fold` en `FOLDS = [0, …, k-1]`. En `comunes.split`, `train_size` no se usa en este modo.

  Cada celda de entrenamiento itera `for fold in FOLDS:`. En `kfold` rehace el split del fold y, en los Casos 2 y 3, también los loaders: `make_loader`, y `construir_loaders_cluster(g3)`, que reconstruye `ClusterData` porque este copia y permuta las máscaras. En `holdout` hace una sola iteración, idéntica a la versión anterior.

  `save_run` recibe `nombre_caso(caso, fold)`: es `<caso>` en holdout y `<caso>_kfold<K>/fold<i>` en kfold, para que los folds no se sobrescriban. En kfold, `extra` incluye además `{"validacion": "kfold", "fold": i, "k": K}` (`extra_validacion(fold)`). La celda 36 junta los folds de `resumen.csv` y muestra media ± std de accuracy, macro-F1 y F1 por clase por (caso, modelo, decoder, `config_id`). Si un fold se corrió varias veces, usa la última corrida.

  El notebook por partes siempre usa holdout: si `tipo: kfold`, solo imprime un aviso (celda 5).
* **Modelos:** `modelos` = `GCN2L`, `GraphSAGE2L`, `GAT2L` (las 3L están comentadas en la celda 15); `modelos_sampler` = `GCNSampler2L`, `GraphSAGESample2L`, `GATSample2L`. GAT usa `gat_hidden_dim`/`gat_out_dim` y `num_heads`.
* **Decoders:** cada modelo se entrena dos veces, con MLP (`edge_dim=1`, recibe el peso de la arista) y con Bilinear (`decodificar`).
* **Pérdida:** `F.cross_entropy` ponderada por clase con `N_train / (3 · n_c)`, calculado solo con train, si `class_weights: true`. No hay focal loss, label smoothing ni gradient clipping.
* **Optimización:** Adam (`lr`, `weight_decay`) + `CosineAnnealingWarmRestarts(T_0, T_mult)`, hasta `epochs` épocas; en mini-batch, el scheduler avanza una vez por época.
* **Early stopping:** por **macro-F1 de validación**, con `patience`; al terminar se restauran los pesos de la mejor época (copiados con `.detach().clone()`).

| Caso | Features de nodo | Modelos | Entrenamiento |
|---|---|---|---|
| **0** PeeringDB | 85 (`load_dataset`) | `modelos` | `entrenar_modelo` (full-batch) |
| **1** Centralidad | 5, z-score (`load_dataset_only_cntrality_attr`) | `modelos` | `entrenar_modelo` |
| **2** Neighbor sampling | 85 | `modelos_sampler` | `entrenar_modelo_sampler`: `LinkNeighborLoader(num_neighbors=[num_neighbors]*2, batch_size=batch_size_caso2)` sobre las aristas de cada split; el peso de la arista se toma de las aristas semilla |
| **3** ClusterGCN | 85 | `modelos_sampler` | `entrenar_modelo_cluster`: `ClusterData(g, num_parts)` + `ClusterLoader(batch_size=batch_size_caso3)` (50 batches por época); la pérdida usa las aristas de `train_mask` dentro de cada subgrafo. Las aristas de test que cruzan clusters de batches distintos no se evalúan (se informa el porcentaje evaluado) |
| **4** Features aleatorias (control) | 64 uniformes (`add_random_features(**random_features, seed=SEED)`) | `modelos` | `entrenar_modelo`, ahora con validación |

Evaluación (`evaluar_test`, `evaluar_test_sampler`, `evaluar_test_cluster`): `argmax` sobre las aristas de test, `classification_report`, matrices de confusión (absoluta y en %) y curvas de loss/accuracy/macro-F1 (`plot_training_evolution`). Después, `save_run` (§3.3.0). La celda de comparación completa ya no existe.

Resultados registrados en los outputs de la **versión anterior** del notebook (modelos 3L, focal loss, `strict_equal`, grafo antiguo de marzo; test). El notebook actual ya no los contiene:

| Caso | Modelo | Accuracy | Macro-F1 | Tiempo |
|---|---|---|---|---|
| 0 | GraphSAGE3L + MLP | 94.41 % | 93.61 % (baseline mayoritaria 51.56 % / 22.68 %) | 44 min (CPU) |
| 1 | GCN3L + Bilinear | 86.38 % | 86.34 % | 18 min |
| 2 | SAGE-Sampler / GAT-Sampler | 91.04 % / 88.54 % | 91.05 % / 88.53 % | 3.9 / 7.8 min |
| 3 | GCN / SAGE / GAT-Sampler (ClusterGCN) | 89.73 / 89.24 / 86.73 % | 86.88 / 87.20 / 84.63 % | 16–19 s |
| 4 | GCN / SAGE / GAT (features aleatorias) | 70.09 / 85.34 / 51.59 % | 70.12 / 85.60 / 42.20 % | 12 / 18 / 8 min |
| Comp. | GCN-2L + MLP | 91.4 % | 91.4 % | 918 s |

#### 3.3.2 Por partes (`AS_relationship_inference_por_partes.ipynb`)

Configuración: `global_config_por_partes` de `config.yaml`, cargada como `GLOBAL_CONFIG_P2E` (celda 6), con sus subsecciones `clasificador` (`CONFIG_CLASIFICADOR`), `deepwalk` y `bgp2vec`. Valores actuales de la Etapa 1: `lr 0.0005, hidden 256, out 128, dropout 0.3, epochs 5, patience 80, weight_decay 0.0005, num_heads 1, gat_hidden_dim 64, gat_out_dim 32, scheduler {T_0: 100, T_mult: 2}`. Los modelos son los de `modelos` (`GCN2L`, `GraphSAGE2L`, `GAT2L`; las 3L están comentadas en la celda 13). Todos los splits usan `comunes.split` (0.7/0.15).

Cada caso hace las dos etapas seguidas: genera los embeddings y enseguida entrena con ellos el clasificador.

**Etapa 1: embeddings.** No se guardan en disco (`save_embeddings` está comentado). Las curvas ROC de link prediction se guardan en `RESULTS_DIR_P2E/Caso{N}/roc_<modelo>_<decoder>.png`.

| Caso | Tarea auxiliar | Función / modelos | Pérdida / selección del mejor estado |
|---|---|---|---|
| 0 | Link prediction con features PeeringDB (`split_edges_link_prediction(seed, **comunes.split)`) | `link_prediction_train`; `modelos` × decoders {MLP, DotProduct}; MLP con `edge_dim=0` (los negativos no tienen peso) | `BCEWithLogits(pos ∪ neg)`; early stopping por AUC de validación. El encoder solo ve `train_g`; el embedding final es `encode(train_g)` |
| 1 | Link prediction con features de centralidad (`load_dataset_only_cntrality_attr`) | idem | idem |
| 2 | Link prediction sobre el grafo externo 2022 (`InternetGNNData2022/caida/caida_pyg.pt`, junto a `rutas.internetgraph_file`; `caida.bin` es binario DGL y debe convertirse una vez) | idem | idem |
| 3 | Regresión del **grado de salida** (`log1p` + min-max); features PeeringDB sin las columnas de centralidad (`quitar_atributos(…, COLS_CENTRALIDAD)`); split de nodos `split_nodos` | `node_regression_train`; `forward()` con un `regressor` que la función agrega a los modelos 2L | `nn.L1Loss` (MAE) y `r2_score`; early stopping por MAE de validación |
| 4 | Regresión de **PageRank** (`calcular_pagerank`: iteración de potencia con `index_add_`, `damp=0.85`, 20 iteraciones; `log` + min-max) | idem | idem |
| 5.1 | DeepWalk: `torch_geometric.nn.Node2Vec` con la sección `deepwalk` (`embedding_dim 32, walk_length 40, context_size 2, walks_per_node 1, p = q = 1, num_negative_samples 5`), `SparseAdam(lr=0.01)`, batch 128, `epochs` | `deepwalk_train` | Pérdida propia de Node2Vec |
| 5.2 | BGP2Vec: `BGP2VEC` (Word2Vec skip-gram) sobre las rutas de `rutas.rib_file_bgp2vec` (`RutasBGP`, hasta `max_rutas` = 27 M), con la sección `bgp2vec` (`embedding_dim 32, window 2, negative 5, epochs 1`) | `bgp2vec_train`; guarda `RESULTS_DIR_P2E/caso5/bgp2vec.word2vec` | — |

En BGP2Vec los embeddings se alinean con los `node_id` del grafo mediante `gnn.asn_to_node_id`; los ASN sin rutas en el RIB quedan con vector 0 (se informa la cobertura).

**Etapa 2: clasificador de aristas** (`EdgeClassifier`, celdas 15–18):

* **Aristas:** las mismas train/val/test que en el end-to-end: `split_edges_classification(seed=SEED, **comunes.split)` sobre el mismo grafo, extraídas con `aristas_clasificacion(gnn)`. Ya no se usa el dataset ToR (`build_tor_dataset` sigue en `modules/gnn.py`, pero ningún notebook lo llama).
* **Vector de arista:** `features_arista(h, edge_index)` = `[e_u ‖ e_v ‖ e_u − e_v ‖ e_u ⊙ e_v]` (`4·emb_dim`), con los embeddings fijos de la Etapa 1. Hoy tiene un bug: `e_v` también se toma de `edge_index[0]` (bug 22, §5.1).
* **Modelo:** `Linear(4d, hidden) → BatchNorm → ReLU → Dropout → Linear(hidden, hidden/2) → BatchNorm → ReLU → Dropout → Linear(hidden/2, 3)`, con `hidden` y `dropout` de `clasificador`.
* **Entrenamiento:** dos variantes por embedding:
  * `entrenamiento_sin_batches`: todas las aristas de train en cada época;
  * `entrenamiento_con_batches`: mini-batches de `batch_size` (4096) aristas.

  Ambas usan cross-entropy ponderada por clase (`pesos_clase`, si `clasificador.class_weights`) con `label_smoothing`, Adam (`lr`, `weight_decay`) + `CosineAnnealingWarmRestarts`, hasta `epochs` épocas, y early stopping por macro-F1 de validación con `patience`. La validación se hace con `evaluar_aristas`, por bloques de 65 536 aristas.
* **Evaluación:** `evaluar_test_clasificador` (reporte y matrices de confusión) y `save_run(RESULTS_DIR_P2E, …)` (§3.3.0). En los Casos 0–2 esto está escrito dentro del bucle de la celda; en los Casos 3–5 lo hace `clasificar_relaciones`.

### 3.4 Decisiones de diseño y su justificación

| Decisión | Dónde | Justificación técnica |
|---|---|---|
| Grafo dirigido con ambas direcciones y etiqueta invertida | `GNN._make_edges_bidirectional`, `add_atributes_edges` | La relación es direccional; duplicar permite que un decoder asimétrico aprenda C2P y P2C como clases distintas y que el message passing fluya en ambos sentidos. |
| Split por par no dirigido + estratificación por firma | `split_edges_classification` | Si `(u→v)` está en train y `(v→u)` en test, la etiqueta de test se deduce trivialmente. Agrupar por par elimina esa fuga; la firma preserva la proporción de pares P2P vs P2C/C2P en cada split. |
| Validación cruzada K-fold por par no dirigido, opcional | `split_edges_kfold`, `comunes.validacion` | Un holdout da un solo número por modelo, sin una medida de su variabilidad. Con K folds se reportan media ± std y se pueden comparar los modelos fold a fold con pruebas pareadas (t pareado, Wilcoxon). La unidad del fold es el par no dirigido, por la misma razón de fuga que en el holdout. `validacion` es una sección aparte de `split` porque `split` se pasa como `**kwargs` a los métodos de split y una clave extra los rompería. **Límites:** los folds aleatorios por enlace no miden generalización temporal (entrenar con un mes y evaluar con otro) ni a AS no vistos (folds disjuntos por AS); la cobertura de cada fold sigue limitada por las etiquetas CAIDA. Con ~185 k aristas de test por fold, el error por muestreo del test es pequeño: la varianza entre folds refleja sobre todo el entrenamiento. Costo ≈ K × el holdout. |
| `balance_mode="proportional"` + pesos de clase | Todos los casos (`comunes.split`, `class_weights`) | Se usan todas las aristas etiquetadas y la cross-entropy se pondera por `N_train / (3 · n_c)` para que P2P (≈ 67 % de las aristas dirigidas etiquetadas) no domine. `strict_equal` (submuestrear al mínimo por clase) sigue disponible en `split_edges_classification`, pero ya no se usa. |
| `focal_loss` con `γ=2` | Versión anterior del notebook end-to-end (ya no se usa) | Reducía la contribución de ejemplos fáciles (mayoría de P2P) y enfatizaba C2P/P2C difíciles. |
| Normalización `log1p + minmax` en todo atributo numérico | `as-level-internet-graph` (`features.py`); antes `4_add_atributes.py` | Grados, centralidades y prefijos siguen distribuciones de cola pesada (power-law); `log1p` comprime la cola y `minmax` lleva todo a `[0,1]` para convivir con los one-hot. Se valida con `_check_range` y en `validar_normalizacion_atributos.ipynb`. |
| z-score para features de centralidad | `load_dataset_only_cntrality_attr` | Escalas muy distintas (PageRank ~1e-5 vs grado ~1) hacían colapsar el entrenamiento (comentario en el código). |
| `add_self_loop` en GCN/GAT, no en SAGE | `encode()` | `GraphConv`/`GATConv` fallan con nodos de in-degree 0 y se benefician de incluir la propia representación; `SAGEConv` ya concatena la proyección del nodo. |
| Self-loops por defecto en `GCNConv`/`GATConv` | Todos los encoders | Cubren los nodos sin vecinos de entrada de los subgrafos muestreados (equivale al `allow_zero_in_degree` de DGL). |
| Decoder Bilinear con `W ∈ R^{C×F×F}` | `BilinearPredictor` | Modela explícitamente la interacción `h_uᵀ W_c h_v` por clase; asimetría natural para C2P/P2C. Es más costoso (`C·F²` parámetros: 3·128² ≈ 49 k) que el MLP. |
| `edge_feat` (peso) en el MLP decoder | `MLPPredictor(edge_dim=1)` | La frecuencia de un par en los AS_PATH correlaciona con la relación (proveedores aparecen más); se inyecta al decoder porque los encoders DGL usados no consumen features de arista. |
| Features aleatorias (Caso 4) | `add_random_features` | Control experimental: mide cuánto aporta la estructura del grafo por sí sola frente a los atributos (GraphSAGE alcanza 85 % solo con estructura). |
| Betweenness aproximada (`k=100`) | `4_add_atributes.py` (grafo antiguo) | Betweenness exacta es O(N·E) ≈ 80 k × 730 k: inviable; el muestreo con semilla fija es reproducible. El formato nuevo no incluye betweenness. |
| Early stopping por macro-F1 en validación | Todos los casos end-to-end y el clasificador por partes | Con clases desbalanceadas el accuracy es engañoso; macro-F1 penaliza el colapso a P2P. |
| Entrenamiento transductivo (aristas de val/test presentes como estructura) | Todos los casos end-to-end | Las aristas de val/test siguen en el grafo durante el message passing (solo se oculta la etiqueta). Es el escenario real (la topología es observable; la relación no) pero debe explicitarse al reportar resultados. |

---

## 4. Guía de Ejecución y Pipeline (Step-by-Step)

### 4.1 Requisitos de entorno

* **SO**: Linux (desarrollado en Ubuntu, kernel 6.8). También se preparó un equipo Windows 10 con GPU (RTX 3060 Ti, 8 GB) con los datos en `F:\`: Python 3.13, torch 2.11+cu128, PyG 2.8 sin extensiones opcionales (ver el README, *En Windows con GPU*). Allí `pyg_lib`/`torch_sparse`/`torch_cluster` no están instaladas, así que pueden fallar `LinkNeighborLoader`, `ClusterData` y `Node2Vec`, y el Caso 2 del por partes necesita `caida_pyg.pt` (conversión desde `caida.bin` con DGL).
* **Python 3.12** (venv `env312`, kernel Jupyter `env312`). El entorno anterior, `env310`, usaba Python 3.10.12.
* **CPU/GPU**: el código selecciona `cuda` si está disponible; los outputs guardados muestran ejecución en **CPU** (Caso 0 ≈ 44 min por modelo). Con GPU (CUDA 12.1) los tiempos bajan un orden de magnitud.
* **RAM**: el grafo de 2026-09-16 (88 k nodos, 1,34 M aristas dirigidas, 85 features) cabe en pocos GB para los encoders GCN/SAGE. Los GAT full-batch con `hidden=256` se quedan sin memoria en la máquina de desarrollo (15 GB, sin GPU): para pruebas use dims pequeñas (`hidden=32`, `out=16`).
* Acceso a red solo para descargar el `as-rel` de CAIDA (`publicdata.caida.org`). La construcción del grafo (BGPStream, PeeringDB) ocurre en `as-level-internet-graph`.

### 4.2 Instalación del entorno

Las dependencias, con versiones fijadas, están en `requirements.txt`: torch 2.8.0 CPU, PyG 2.8 con sus cuatro extensiones (`LinkNeighborLoader`, `ClusterData`/METIS y `Node2Vec` las necesitan) y el resto de las librerías. El archivo trae los índices de PyTorch (`--extra-index-url`) y de PyG (`--find-links`).

```bash
# 1. Clonar y crear entorno
git clone <url-del-repo> gnn-as-relationships
cd gnn-as-relationships
python3.12 -m venv env312
#    En Ubuntu sin el paquete python3.12-venv (falta ensurepip):
#    python3.12 -m venv --without-pip env312 && python3.12 -m pip --python env312/bin/python install pip
source env312/bin/activate
pip install --upgrade pip

# 2. Dependencias (CPU). Para GPU con CUDA 12.8: cambiar "cpu" por "cu128"
#    en las dos URLs de requirements.txt y en torch==2.8.0+cpu
pip install -r requirements.txt

# 3. Registrar el kernel Jupyter
python -m ipykernel install --user --name env312 --display-name "Python 3.12 (env312)"
```

Los notebooks declaran el kernel genérico `python3`. En Jupyter/VS Code, elija `env312`.

`requirements.txt` está versionado desde el 2026-11-16 (`.gitignore` lo exceptúa de `*.txt`).

> **Importante sobre rutas.** Las rutas absolutas al disco externo y los parámetros de ambos notebooks se definen en **`config.yaml`** (raíz del repo). Cada notebook lo abre con `open("config.yaml")`, una ruta relativa, así que Jupyter debe abrirse desde la raíz. Hoy las rutas apuntan a `/media/vale/KINGSTON/...`; las del equipo Windows (`F:/`) quedaron en `config_nerwito.yaml`, con el formato anterior:
> * `experimento.nombre` y `experimento.salida`: los resultados van a `<salida>/<nombre>/end_to_end/` y `<salida>/<nombre>/por_etapas/`; hoy `salida = /media/vale/KINGSTON/LAB/gnn-as-relationships/resultados_gnn`.
> * `rutas.grafo_dir` (`GRAFO_DIR`): carpeta del grafo de `as-level-internet-graph`, `…/experimentos/experimento<N>/grafos/<año>/<mes>/<AAAAMMDDTHHMM>/`. **Para cambiar de grafo basta con cambiar esta clave.**
> * `rutas.caida_dir` (`CAIDA_DIR`): carpeta con los `AAAAMM01.as-rel.txt.bz2`; hoy `/media/vale/KINGSTON/LAB/CAIDA/CAIDA_AS_Relationships/Serial_1`. Los notebooks la leen, pero no la usan.
> * `rutas.rib_file_bgp2vec` (solo por partes, BGP2Vec): RIB en texto (`rib_*.txt`), que `as-level-internet-graph` no genera; hoy la de 24 h del 1 de marzo de 2026 en `TESIS/data_2026/` (marcada `FIXME`).
> * `rutas.internetgraph_file`: el grafo externo 2022 del Caso 2 por partes, en `/media/vale/KINGSTON/LAB/gnn-as-relationships/InternetGNNData2022/caida/caida.bin`.
>
> `config.yaml` también contiene `comunes` (`seed`, `class_names`, `split`, `validacion`, `class_weights`), `global_config_endtoend` y `global_config_por_partes` (con `clasificador`, `deepwalk` y `bgp2vec`), que los notebooks cargan como `GLOBAL_CONFIG_E2E`, `GLOBAL_CONFIG_P2E` y `CONFIG_CLASIFICADOR`. Los notebooks ya no leen variables de entorno. Los decimales se escriben con punto (`0.0005`), porque PyYAML lee `5e-4` como texto. El `split` debe cumplir `train_size + val_size < 1.0`.
>
> `comunes.validacion` elige cómo se valida el clasificador del notebook end-to-end (§3.3.1):
>
> ```yaml
> comunes:
>   validacion:
>     tipo: holdout   # holdout (por defecto) | kfold
>     k: 5            # solo con tipo: kfold
> ```
>
> * `holdout`: un solo split con `comunes.split`. Si falta la sección, se usa este modo.
> * `kfold`: `k` folds. Test = 1/k de los pares; val = `split.val_size` del resto.
>
> Un `tipo` distinto de esos dos, o `k < 2`, detiene la celda 12 con un `ValueError`. El notebook por partes ignora esta sección y avisa si vale `kfold`.
>
> La carpeta antigua `/media/vale/KINGSTON/as-level-internet-graph/grafos/` ya no existe (se borró el 2026-10-14). **No existen flags de línea de comandos** (ver §5).

### 4.3 Preparación de datos

**Paso 1: construir el grafo (en `as-level-internet-graph`).** Ver su README. Por ejemplo, un grafo del día 1 de mes, que coincide con el snapshot CAIDA:

```bash
cd ../as-level-internet-graph
python as_graph.py --date "2026-09-01 00:00" --jobs 4 --experimento nuevo --descripcion "día 1 de mes, para entrenar con CAIDA AS-Rel"
python scripts/enriquecer_nodos.py --todos "$ASGRAPH_OUT_DIR"/experimentos/experimento<N>
# → GRAFO_DIR/as_features.csv, GRAFO_DIR/gnn_edges.csv
```

**Paso 2: descargar el archivo CAIDA del mes del grafo** (serial-1, publicado el día 1):

```bash
wget -P /media/vale/KINGSTON/LAB/CAIDA/CAIDA_AS_Relationships/Serial_1/ \
     https://publicdata.caida.org/datasets/as-relationships/serial-1/20260901.as-rel.txt.bz2
```

**Paso 3: etiquetar las aristas.** `label_edges_from_caida` se eliminó de `modules/gnn.py` (cambio sin commit) y los notebooks ya no etiquetan solos. Los grafos de experimento7 ya tienen `gnn_edges_caida.csv`. Para un grafo nuevo hay que recuperar el método desde git (`git show HEAD:modules/gnn.py`). Con él:

```python
from pathlib import Path
from modules.gnn import GNN

GRAFO_DIR = Path("/media/vale/KINGSTON/as-level-internet-graph/experimentos/experimento7/grafos/2026/09/20260916T0800")
CAIDA_DIR = Path("/media/vale/KINGSTON/TESIS/data_2026")

resumen = GNN(debug=True).label_edges_from_caida(
    GRAFO_DIR / "gnn_edges.csv",
    GNN.caida_file_for_graph(GRAFO_DIR, CAIDA_DIR),   # → CAIDA_DIR/20260901.as-rel.txt.bz2
    output_csv=GRAFO_DIR / "gnn_edges_caida.csv",     # + GRAFO_DIR/etiquetas_caida.json
)
```

Para volver a etiquetar, por ejemplo con otro snapshot CAIDA, borre `gnn_edges_caida.csv` o llame al método con otro `caida_file`.

**Grafo antiguo de marzo 2026.** Los CSV `*_enriched_tesis.csv` (72 features, ya etiquetados) están en `/media/vale/KINGSTON/TESIS/data_2026/`. Los notebooks fijan los nombres `as_features.csv` y `gnn_edges_caida.csv` en la celda de configuración (celda 12 end-to-end, celda 5 por partes), así que para usarlos hay que editar `nodes_file` y `edges_file` en esa celda.

**Dataset ToR para el benchmark.** `GNN.build_tor_dataset(output_dir, **info)` exporta el split de clasificación actual (`train_mask`, `val_mask`, `test_mask`) con el formato que leen los notebooks de `BenchmarckASRelationships` (Gao, Ruan, ProbLink, TopoScope, BGP2Vec): `x_training.npy`, `x_val.npy` y `x_test.npy` (pares de ASN como texto, forma `(N, 2)`, las dos direcciones de cada enlace), `y_training.npy`, `y_val.npy` e `y_test.npy` (`0 = P2P`, `1 = C2P`, `2 = P2C`) y `tor_resumen.json` (tamaños, clases y `info`). Así el benchmark se evalúa sobre **las mismas aristas de test que la GNN**, sin fuga entre direcciones. Requiere `load_dataset` (usa `asn_to_node_id`) y un split previo.

El notebook end-to-end lo llama en la celda 18, solo en holdout, y lo guarda en `<experimento.salida>/<experimento.nombre>/tor/`; con `correr_pruebas.py` queda en `…/pruebas/<prueba>/<variante>/tor/`. Las 5 variantes de `semillas` ya lo tienen (generado el 2026-11-24 rehaciendo el split con la config de cada variante; coincide con `n_test` y las clases de su `metrics.json`). Para crearlo a mano:

```python
g = GNN(); g.load_dataset(GRAFO_DIR / "as_features.csv", GRAFO_DIR / "gnn_edges_caida.csv")
g.split_edges_classification(seed=42, **cfg["comunes"]["split"])
g.build_tor_dataset(SALIDA / NOMBRE / "tor", grafo=RUN_TAG, seed=42, split=cfg["comunes"]["split"])
```

Antes (hasta el 2026-11-24) la función hacía su propio corte 80/20 desde `gnn_edges_caida.csv`, distinto del split de la GNN. El dataset que usa hoy el benchmark (`TESIS/data_2026/{x,y}_{training,test}.npy`, CAIDA de marzo de 2026, de su propio `create_tor_dataset.py`) tiene fuga entre direcciones: el 89 % de los enlaces de test también está en train con la etiqueta invertida.

**Prueba de humo** (desde la raíz del repo):

```python
gnn = GNN(); gnn.load_dataset(str(GRAFO_DIR / "as_features.csv"), str(GRAFO_DIR / "gnn_edges_caida.csv"))
# → 87 796 nodos, 1 336 852 aristas, 85 features
gnn.split_edges_classification(0.7, 0.15, seed=42)
# → train 691 574 / val 148 194 / test 148 196
gnn.split_edges_link_prediction(train_size=0.7, val_size=0.15, seed=42)
# → positivos train 935 796 / val 200 526 / test 200 530 (y tantos negativos)
```

### 4.4 Entrenamiento y evaluación

Los training loops viven en los notebooks; ejecútelos desde la raíz del repositorio (los `import modules.…` son relativos a ella):

```bash
source env312/bin/activate
jupyter lab   # o jupyter notebook
```

**End-to-end** (`AS_relationship_inference_end_to_end.ipynb`), orden de celdas:

Antes de abrir el notebook, edite `config.yaml` (§4.2): `rutas.grafo_dir`, `experimento.nombre` y los hiperparámetros.

1. Celda 2: imports (`%autoreload` recarga `modules/` al editar).
2. Celda 5: funciones de ploteo (`plot_confusion_matrix_2`, `plot_training_evolution`).
3. Celdas 7–9: funciones de entrenamiento y evaluación (`decodificar`, `entrenar_modelo`, `evaluar_test`; `pasar_loader`, `entrenar_modelo_sampler`, `evaluar_test_sampler`; `pasar_loader_cluster`, `entrenar_modelo_cluster`, `evaluar_test_cluster`).
4. Celdas 11–13: configuración. La 12 lee `config.yaml` y define `GRAFO_DIR`, `nodes_file`, `edges_file`, `RUN_TAG` (`experimento<N>_<AAAAMMDDTHHMM>`), `SEED`, `CLASS_NAMES`, `E2E_SPLIT`, `VALIDACION`, `K`, `FOLDS` y `RESULTS_DIR_E2E`, y las funciones auxiliares `hacer_split`, `nombre_caso` y `extra_validacion`. La 13 carga `GLOBAL_CONFIG_E2E`.
5. Celda 15: `modelos` (2L; las 3L están comentadas) y `modelos_sampler`. Para cambiar arquitectura o profundidad, edite esos dicts.
6. Elegir un caso y ejecutar sus celdas en orden: Caso 0 (18, 20), Caso 1 (22–23), Caso 2 (25–27), Caso 3 (29–31), Caso 4 (33–34). La primera celda de cada caso carga el grafo, hace el split (holdout, o el primer fold en kfold) e imprime un resumen. La última entrena cada modelo con MLP y con Bilinear, evalúa en test y llama a `save_run`; en kfold, repite todo esto para cada fold.
   * **Para usar K-fold:** poner `comunes.validacion.tipo: kfold` (y `k`) en `config.yaml`, re-ejecutar la celda 12 y luego las celdas del caso. Con `tipo: holdout` todo funciona como antes. K-fold multiplica el tiempo por K: en CPU conviene empezar con los casos full-batch (0, 1, 4) y sin GAT.
   * **Celda 36:** junta los folds y muestra media ± std por caso, modelo y decoder. Si no hay corridas kfold en `resumen.csv`, solo lo avisa.
7. Salidas:
   * en pantalla: log de la época 1 y cada 50 (cada época en mini-batch), tiempo total, `classification_report`, matrices de confusión y curvas;
   * en disco: cada modelo guarda métricas, pesos e historial en `RESULTS_DIR_E2E/<caso>/<modelo>_<decoder>_seed<SEED>_<config_id>/` y una fila en `RESULTS_DIR_E2E/resumen.csv` (§3.3.0).

**Por partes** (`AS_relationship_inference_por_partes.ipynb`):

1. Celda 2: imports. Celdas 4–6: configuración (`config.yaml` → `GRAFO_DIR`, `nodes_file`, `edges_file`, `SEED`, `P2E_SPLIT`, `RESULTS_DIR_P2E`, `GLOBAL_CONFIG_P2E`, `CONFIG_CLASIFICADOR`).
2. Celdas 8–11: auxiliares (métricas de link prediction y ROC, `quitar_atributos`, `split_nodos`, `calcular_pagerank`, tiempos). Celda 13: `modelos` y `modelos_sampler`.
3. Celdas 15–18: Etapa 2 (`EdgeClassifier`, `features_arista`, `evaluar_aristas`, `entrenamiento_sin_batches`, `entrenamiento_con_batches`, `evaluar_test_clasificador`, `clasificar_relaciones`). Celdas 20–23: Etapa 1 (`link_prediction_train`, `node_regression_train`, `deepwalk_train`, `bgp2vec_train`).
4. Elegir un caso y ejecutar sus dos celdas (carga y splits, luego entrenamiento): Caso 0 (27, 29), Caso 1 (31, 33), Caso 2 (35, 37), Caso 3 (39, 41), Caso 4 (43, 45). Caso 5: carga (47), DeepWalk (49), BGP2Vec (52). Cada caso entrena los embeddings y enseguida el clasificador, y termina con una tabla de tiempos.
5. Salidas en `RESULTS_DIR_P2E`: resultados del clasificador (`save_run`), `Caso{N}/roc_*.png` y `caso5/bgp2vec.word2vec`. Los embeddings no se guardan. Los embeddings antiguos del grafo de marzo están en `/media/vale/KINGSTON/TESIS/data_2026/results/` y no son compatibles con otro grafo (cambian los nodos y la dimensión).

### 4.5 Inferencia con un modelo entrenado

No hay script de inferencia; el patrón mínimo, coherente con el código de los notebooks:

```python
import torch, csv
from modules.gnn import GNN
from modules.gnn_models import GraphSAGE3L

gnn = GNN(debug=False)
gnn.load_dataset(str(GRAFO_DIR / "as_features.csv"), str(GRAFO_DIR / "gnn_edges_caida.csv"))
g = gnn.graph
in_feats = g.x.shape[1]                             # 85

model = GraphSAGE3L(in_feats, 256, 128, out_feats_mlp=3, edge_dim=1, drop=0.3)
model.load_state_dict(torch.load("ruta/al/modelo.pth", map_location="cpu"))
model.eval()

with torch.no_grad():
    h = model.encode(g.x, g.edge_index)             # embeddings (N, 128)
    logits = model.decodeMLP(g.edge_index, h, g.edge_attr)   # (E, 3)
    pred = logits.argmax(1)                         # 0=P2P, 1=C2P, 2=P2C

# Traducir a ASNs
id_to_asn = {int(r["node_id"]): int(r["asn"]) for r in csv.DictReader(open(GRAFO_DIR / "as_features.csv"))}
u, v = g.edge_index
for eid in range(5):
    print(id_to_asn[int(u[eid])], "->", id_to_asn[int(v[eid])], ["P2P","C2P","P2C"][int(pred[eid])])
```

Las aristas con `edge_label == -1` (174 444 enlaces, ≈26 %, en 2026-09-16) son exactamente las que el modelo puede inferir sin etiqueta de referencia. Para pares de AS **no** presentes en el grafo, use el clasificador del enfoque por partes (`features_arista` + `EdgeClassifier`, celdas 15–16 del notebook), siempre que ambos AS tengan embedding, o el estilo de `_fill_labels_from_caida_stream_fast` para agregar nodos con `feat=0`.

### 4.6 Artefactos generados

| Ruta | Contenido |
|---|---|
| `GRAFO_DIR/as_features.csv`, `gnn_edges.csv` | Grafo (generado por `as-level-internet-graph`) |
| `GRAFO_DIR/gnn_edges_caida.csv`, `etiquetas_caida.json` | Aristas etiquetadas y resumen (generados por `GNN.label_edges_from_caida`, hoy eliminado) |
| `CAIDA_DIR/AAAAMM01.as-rel.txt.bz2` | CAIDA AS-Relationships (descarga manual) |
| `/media/vale/KINGSTON/TESIS/data_2026/` | Grafo antiguo de marzo (`*_enriched_tesis.csv`), `rib_*.txt`, dataset ToR antiguo (`*.npy`) |
| `<salida>/<nombre>/end_to_end/<caso>/<modelo>_<decoder>_seed<SEED>_<config_id>/` | `metrics.json`, `model.pth`, `history.csv` de cada modelo end-to-end (`save_run`). Con `comunes.validacion.tipo: kfold`, `<caso>` es `<caso>_kfold<K>/fold<i>` (una subcarpeta por fold) y `metrics.json → extra` incluye `validacion`, `fold` y `k` |
| `<salida>/<nombre>/end_to_end/resumen.csv` | Una fila por corrida: fecha, caso, modelo, decoder, semilla, `config_id`, accuracy, macro-F1, `tiempo_s`, `tiempo_total_s`, épocas, carpeta |
| `<salida>/<nombre>/por_etapas/<caso>/<modelo>_<decoder>-<modo>_seed<SEED>_<config_id>/`, `por_etapas/resumen.csv` | Lo mismo para el clasificador de la Etapa 2, por embedding y variante (`sin_batches` / `con_batches`) |
| `<salida>/<nombre>/por_etapas/Caso{0..2}/roc_<modelo>_<decoder>.png` | Curvas ROC y distribución de scores de link prediction |
| `<salida>/<nombre>/por_etapas/caso5/bgp2vec.word2vec` | Modelo Word2Vec de BGP2Vec |
| `notebooks/results/*.pdf` | Figuras de análisis. **Eliminadas** del repo el 2026-11-16 junto con los notebooks de análisis (siguen en git) |
| `notebooks/resultados_experimentos.ipynb` | Tablas de accuracy, F1, métricas por clase (P2P/C2P/P2C) y tiempos a partir de los `metrics.json`. Hoy no corre: lee `rutas.experimento_dir`, que ya no está en `config.yaml`, y busca los resultados en `GRAFO_DIR/results/…`, no en `<salida>/<nombre>/` |

`<salida>` y `<nombre>` son `experimento.salida` y `experimento.nombre` de `config.yaml`.

### 4.7 Ejecución por lotes

`scripts/correr_experimento.py`, que ejecutaba los dos notebooks sobre todos los grafos de un experimento, se eliminó en el commit `d1495d9`. Sigue en el historial de git. Los notebooks actuales no tienen las etiquetas de celda por caso que usaba ni leen variables de entorno, así que no son compatibles con él sin cambios.

#### Correr varias pruebas (`scripts/correr_pruebas.py`)

Automatiza pruebas que varían hiperparámetros, semillas o el split (holdout con distintas proporciones o K-fold con `split_edges_kfold`) sobre los dos notebooks, sin modificarlos. Las pruebas se describen en un YAML (ejemplo en `pruebas.yaml`, en la raíz):

```yaml
base: config.yaml              # config de partida (ruta relativa al archivo de pruebas)
salida: /media/vale/KINGSTON/LAB/gnn-as-relationships/ejecuciones

pruebas:
  - nombre: semillas
    notebooks: [end_to_end, por_partes]
    barrido:                   # producto cartesiano: una variante por combinación
      comunes.seed: [42, 43, 44, 45, 46]

  - nombre: splits
    notebooks: [end_to_end]
    barrido:
      comunes.split:
        - {train_size: 0.7, val_size: 0.15, balance_mode: proportional}
        - {train_size: 0.6, val_size: 0.2,  balance_mode: proportional}
      comunes.validacion:
        - {tipo: holdout}
        - {tipo: kfold, k: 5}

  - nombre: lr_hidden
    notebooks: [end_to_end]
    cambios:                   # se aplican a todas las variantes de la prueba
      global_config_endtoend.epochs: 200
    barrido:
      global_config_endtoend.lr: [0.001, 0.0005]
      global_config_endtoend.hidden_dim: [128, 256]
```

- Las claves van con puntos sobre `config.yaml` y deben existir en la base (un error de tipeo se detecta antes de entrenar). El valor puede ser un escalar o un dict completo (`comunes.split`, `comunes.validacion`).
- Antes de ejecutar se valida todo: `train_size + val_size < 1` (holdout y por partes), `val_size` en [0, 1), `validacion.tipo ∈ {holdout, kfold}`, `k >= 2` y `balance_mode`. Fuera de `--dry-run` también se comprueba que existan `as_features.csv` y `gnn_edges_caida.csv` de cada `rutas.grafo_dir` (p. ej. que el disco externo esté montado).
- Con `kfold` solo se usa `val_size` (test = 1/k): las variantes que solo difieren en `train_size` se ejecutan una vez (aviso en la salida).
- El notebook por partes solo implementa holdout: en las variantes `kfold` no se ejecuta y queda como `omitido`.

Uso, desde la raíz y con el venv del notebook:

```bash
env312/bin/python scripts/correr_pruebas.py pruebas.yaml --dry-run    # lista variantes, cambios y avisos
env312/bin/python scripts/correr_pruebas.py pruebas.yaml              # corre todo
env312/bin/python scripts/correr_pruebas.py pruebas.yaml --solo splits
env312/bin/python scripts/correr_pruebas.py pruebas.yaml --reanudar   # salta lo que ya terminó ok
```

Por cada variante y notebook, el script:

1. escribe `<salida>/<prueba>/<variante>/config.yaml` con los cambios y con `experimento.nombre = <nombre base>/pruebas/<prueba>/<variante>`. Así cada variante tiene sus propios `end_to_end/` y `por_etapas/`. Es necesario porque `config_id` solo depende de `global_config_*`: dos variantes que cambian `comunes.split` con la misma semilla se pisarían en la misma carpeta. Si la prueba fija `experimento.nombre`, se respeta;
2. ejecuta el notebook completo en un kernel nuevo (`nbclient`), cambiando solo en memoria la línea `CONFIG_FILE = ...`. El `.ipynb` del repositorio no se modifica;
3. guarda el notebook ejecutado en `<salida>/<prueba>/<variante>/<notebook>.ipynb`, también si falla. Un error (p. ej. falta de memoria) se registra y se sigue con la siguiente variante;
4. agrega una fila a `<salida>/estado.csv`: fecha, prueba, variante, notebook, estado (`ok` / `error` / `omitido`), duración, error, config y carpeta de resultados.

Al terminar junta los `resumen.csv` de las variantes `ok` y muestra media ± std de accuracy y macro-F1 por prueba, variante sin la semilla, enfoque, validación, caso, modelo y decoder. Los folds de K-fold se agrupan igual que las semillas (`caso0_kfold5/fold<i>` → `caso0`), así que `n` = semillas × folds. La tabla se guarda en `<salida>/resumen_pruebas.csv`.

Limitaciones:
- cada variante corre el notebook completo (todos los casos y modelos), y K-fold multiplica el tiempo por k. En la máquina de desarrollo (15 GB, sin GPU) GAT full-batch con dimensiones grandes se queda sin memoria: esa ejecución queda como `error`;
- el `save_run` del caso 0 por partes está comentado, así que ese caso no aparece en la tabla;
- la semilla también cambia el split (`seed=SEED`): la std mezcla la variación del split y la de la inicialización.

El Caso 2 por partes necesita `InternetGNNData2022/caida/caida_pyg.pt`, que hoy no existe: hay que convertir `caida.bin` desde un entorno con DGL (§3.3.2).

---

## 5. Propuestas de Mejora y Deuda Técnica

### 5.1 Análisis crítico del código actual

**Bugs y defectos confirmados por lectura del código**

| # | Ubicación | Problema | Efecto |
|---|---|---|---|
| 1 | `modules/gnn.py` | *(Resuelto en parte el 2026-11-24: se eliminó la primera definición.)* La clase `GNN` estaba definida **dos veces**; la segunda sombreaba a la primera (que inicializaba `train_eids/val_eids/test_eids`). Siguen los imports duplicados (`torch`, `random`, `defaultdict`, `Counter`). | Los atributos `*_eids` solo existen tras llamar a `split_edges_classification`; código confuso. |
| 2 | `modules/gnn.py` (`split_edges_link_prediction`, `split_edges_classification_leaky`, `_v0`) | *(Resuelto)* Buscaban `edata["Relationship"]`, pero `load_dataset` escribe `edge_label`. | `_leaky` y `_v0` se eliminaron, y `split_edges_link_prediction` ahora copia `edge_label` (o `label`/`relationship`/`Relationship`) a los subgrafos positivos. |
| 3 | `modules/gnn.py` (`split_edges_link_prediction`) | En link prediction se eliminan las aristas de test de `train_g`, pero como el grafo es bidireccional **la arista inversa de cada positivo de test sigue en `train_g`** (y probablemente en `train_pos_g`). | Fuga de información: los AUC de los Casos 0–2 del enfoque por partes están sobreestimados. **Corregido**: el split ahora es por enlace no dirigido (los resultados previos se obtuvieron con la versión con fuga). |
| 4 | `modules/gnn_models.py:195-201, 229-235, 263-269` | `GCN2L`, `GraphSAGE2L` y `GAT2L` definen `forward()` usando `self.regressor`, que **no existe** en las clases 2L. | `AttributeError` al usar modelos 2L para regresión nodal. El notebook por partes lo evita: `node_regression_train` asigna `model.regressor` antes de entrenar. |
| 5 | `modules/gnn_models.py` (`GAT2L`) | *(Corregido en `c065a4f`)* `GAT2L.Bilinear = BilinearPredictor(out_feats, …)` mientras el embedding tiene `out_feats × num_heads` dims. | `decodeBilinear` fallaba por tamaño en GAT2L. Ahora usa `out_feats × num_heads`. |
| 6 | `modules/gnn_models.py` (`GATSample2L`) | *(Corregido en la migración)* La versión DGL no aplanaba las cabezas entre `conv1` y `conv2` y el `MLPPredictor` recibía una dimensión inconsistente. | En PyG `conv2` y el MLP usan `hidden×heads` / `out×heads`. |
| 7 | `modules/graph.py:371-380` | En `label_edges_caida`, tras asignar `label = 2` para `(src→dst)`, la comprobación `label == -1` para `(dst→src)` ya es falsa. | **Ambas direcciones quedan P2C**; el commit "Corregir etiquetas P2C/C2P invertidas" no lo resolvió. (Pipeline legado.) |
| 8 | `modules/graph.py:430-471` | `only_degree_features_nodes` contiene código muerto tras `f.close()` que escribe en el archivo cerrado. | `ValueError: I/O operation on closed file`. |
| 9 | `modules/graph.py:190` | `create_graph_from_caida` ignora `filename_out` y concatena `self.data_path + "edges.csv"` sin `os.path.join`. | Ruta incorrecta si `data_path` no termina en `/`. |
| 10 | `scripts/4_add_atributes.py` (`__main__`) | *(Script eliminado en `4fcb754`)* *(Corregido)* `caida_relationships_file` (2026) se sobrescribía con la ruta CAIDA de 2024 antes de usarse y `peeringdb_file` se definía sin usarse. | Hasta la corrección, el etiquetado del RIB 2026 usaba relaciones CAIDA de marzo 2024; los CSV de `data/` pueden haberse generado así. |
| 11 | `scripts/2_create_edges_from_ribs.py:92` *(eliminado en `4fcb754`)* | `sorted(dict_nodes.keys())` ordena **strings**, no enteros (`"10" < "9"`). | `node_id` no sigue orden numérico de ASN (inofensivo, pero difiere de `3_filter_by_country.py`, que sí usa `key=int`). |
| 12 | `scripts/1_ribs_extraction.py` *(eliminado en `4fcb754`)* | `argparse` importado y no usado; `count_records` se incrementa antes del chequeo `rec is None`; `strftime('%s')` no portable. | Configuración no parametrizable; conteo desviado en 1. |
| 13 | `modules/bgp2vec.py:86-90` | `asn2idx`/`idx2asn` usan la API de gensim 3 (`wv.vocab`, `wv.index2word`). | Rotos con gensim 4.x (instalado 4.4.0). |
| 14 | `modules/bgp2vec/bgp2vec.py:50` | `total_examples=len(self.routes)` falla con iteradores (`RIBCorpus`, `islice`). | Copia obsoleta; debe eliminarse. |
| 15 | `.gitignore` | Ignora `*.md`, `*.csv`, `*.json`, `*.txt`, `*.png`; solo `README.md` está exceptuado. Hoy (cambio sin commit) también vuelve a ignorar `DOCUMENTATION.md`. | Cualquier documentación o dato nuevo queda invisible para git. `DOCUMENTATION.md` y `CLAUDE.md` siguen versionados porque ya estaban en el índice. `requirements.txt` cae en `*.txt` y no está en el repo. |
| 16 | `src/__pycache__/utils.py`, `src/graph_builder.py` | Fuente dentro de `__pycache__` y módulo vacío. Los `.pyc` versionados se quitaron en `f327a24`. | Ruido en el repo; riesgo de importar la versión equivocada de `utils`. |
| 17 | `README.md` | *(Corregido)* Citaba `3_export_to_dgl_graph.py` (no existe) y omitía `4_add_atributes.py`. | Onboarding incorrecto hasta la corrección. |
| 18 | `notebooks/*.ipynb` (análisis), `modules/bgp2vec/ToR_bgp2vec.ipynb` | Leen sus rutas de claves de `config.yaml` que el archivo actual ya no tiene: `rutas.grafo_antiguo_dir` (`analisis_grafo_internet`, `analisis_ribs_recolectadas`, `AS_embeddings_visualization`, `attribute_inspection`, `peeringdb_coverage_bias`, `tsne_embeddings`, `validar_normalizacion_atributos`), `rutas.as_graph_dir` (`analysis_as_topology`), `rutas.experimento_dir` (`resultados_experimentos`) y `rutas.tesis_path` (`ToR_bgp2vec`). Salvo `analysis_as_topology` y `resultados_experimentos`, además leen el formato antiguo (`nodes_*/edges_*_enriched_tesis.csv`, `rib_*.txt`). | `KeyError` al cargar la configuración. Solo se migraron los dos notebooks de entrenamiento. |
| 19 | Notebook end-to-end, celdas de entrenamiento | *(Corregido)* El "mejor modelo" se guardaba con `best_w = model.state_dict()` (o `{k: v.cpu() …}` en CPU), que no copia: apunta a los pesos vivos. | Hasta la corrección, test se evaluaba con los pesos de la **última** época, no con los de la mejor en validación. Las cifras guardadas en los notebooks están afectadas. Ahora se copia con `.detach().clone()`. |
| 20 | `AS_relationship_inference_por_partes.ipynb`, celda 93 | *(Corregido)* `save_dir` concatenaba `results/EnfoquePorPartes/{CASO}/` tres veces. | `savefig` fallaba por una carpeta inexistente. Ahora se guarda `CasoN/confusion_matrix_<embedding>.png`. (La celda ya no existe: la Etapa 2 se reescribió.) |
| 21 | `AS_relationship_inference_end_to_end.ipynb`, celda 24 (Caso 1 Bilinear) | *(Corregido)* El bloque de diagnóstico del final, fuera del bucle, usaba `hist`, `model_f1_macro` y `decodeMLP`, que vienen del Caso 0. | Solo funcionaba si el Caso 0 se había corrido antes en la misma sesión y fallaba con `NameError` al ejecutar el caso aislado (`scripts/correr_experimento.py`). Ahora usa `history`, `f1_macro` y `decodeBilinear` de la propia celda. (La celda ya no existe: el notebook se reescribió en `c065a4f`.) |
| 22 | `AS_relationship_inference_por_partes.ipynb`, celda 16 (`features_arista`) | `nodo_dst = embeddings[edge_index[0]]`: el embedding del destino se toma de la fila de origen. | El vector de arista queda `[e_u ‖ e_u ‖ 0 ‖ e_u ⊙ e_u]`: el clasificador de la Etapa 2 solo ve el nodo origen, así que la predicción de `u→v` depende únicamente de `e_u`. Afecta a todos los resultados de la Etapa 2. |

**Problemas metodológicos**

* Los casos con muestreo (2 y 3) usan ahora los mismos `epochs` y `patience` que el full-batch, pero una época de mini-batch hace muchas actualizaciones de pesos y una de full-batch solo una: el presupuesto de entrenamiento sigue sin ser equivalente. ClusterGCN, con `num_parts=1000` y `batch_size_caso3=20`, procesa 50 subgrafos por época; las aristas que cruzan subgrafos no participan ni se evalúan.
* `BilinearPredictor.W` se inicializa con `torch.randn` sin escalado (`1/√F`), lo que produce logits de gran magnitud al inicio. Antes lo mitigaba `grad_clip=2.0`, que ya no se usa.
* El vocabulario de BGP2Vec proviene de un RIB de 24 h del 1 de marzo de 2026 (`rutas.rib_file_bgp2vec`, marcado `FIXME`), distinto del grafo en uso; `as-level-internet-graph` no genera RIBs en texto. Además, `epochs: 1`.
* Los valores actuales de `config.yaml` (`epochs: 100` end-to-end, `epochs: 5` en la Etapa 1 y en el clasificador por partes) son de prueba: los resultados con esa configuración no son definitivos.
* **Cobertura CAIDA parcial.** Con el grafo 2026-09-16 y CAIDA `20260901`, el 26 % de los enlaces (174 444) queda sin etiqueta (`-1`) y fuera de los splits de clasificación. Además, el grafo es del día 16 y el snapshot CAIDA del día 1. Para que coincidan, construya el grafo el día 1 del mes (§4.3).
* Las cifras de la tabla de §3.3.1 (72 features, tiempos, métricas) corresponden a los outputs guardados con el grafo antiguo de marzo. Con los grafos nuevos hay que volver a correr los casos.
* Entrenamiento transductivo: aristas de val/test participan en el message passing (ver §3.4). Es defendible, pero debe declararse y, idealmente, contrastarse con un escenario inductivo (`remove_edges` de test antes de `encode`).

**Calidad general**

* El entrenamiento ya está en funciones, pero hay varias copias casi idénticas del mismo bucle (`entrenar_modelo`, `entrenar_modelo_sampler`, `entrenar_modelo_cluster`, `link_prediction_train`, `node_regression_train`, `entrenamiento_sin_batches`, `entrenamiento_con_batches`), definidas en los notebooks y no en `modules/`. `decodificar` y `plot_training_evolution` están duplicadas entre los dos notebooks.
* Seis clases de encoder casi idénticas; ausencia de tipado en `modules/`, docstrings parciales y mezcla español/inglés.
* La configuración ya está centralizada en `config.yaml`, pero algunas claves no se usan (`n_neighbors` y `batch_size` en end-to-end; `cases` y `label_smooth` en por partes; `comunes.class_weights`). Las salidas son mayormente por `print`. La semilla ya está unificada (`SEED` + `set_seed`, §3.3.0), pero en GPU la reproducibilidad no es exacta.
* Cero tests automatizados.

### 5.2 Oportunidades de optimización

**Rendimiento de entrenamiento**

* Mover el training loop a `modules/train.py` con una función `train_edge_classifier(model, g, cfg, decoder="mlp")` y un CLI (`python -m modules.train --case 0 --model sage --decoder mlp --device cuda`). Elimina las copias y hace reproducibles los experimentos.
* Ejecutar en GPU y usar `torch.autocast` (AMP) para el decoder y las capas lineales; considerar `torch.compile` en PyTorch ≥ 2.3.
* Evaluar en validación cada *k* épocas en lugar de cada época (hoy cada época implica dos `encode` completos).
* `split_edges_classification` construye `pair2eids` con bucles Python sobre 660 k aristas; vectorizar con `torch.minimum/maximum` + `torch.unique(return_inverse=True)`.

**Grafos a gran escala y memoria**

* Para RIBs completos (varios colectores, IPv6) el grafo puede superar 100 k nodos y varios millones de aristas: guardar el `Data` con `torch.save` (o un `InMemoryDataset` de PyG) para evitar re-parsear CSV (hoy `load_dataset` reconstruye el grafo en cada notebook).
* En `1_ribs_extraction.py`, reemplazar el `set` de AS_PATHs completos por hashing (`hash(path)`) o por deduplicación posterior con `sort -u`, y escribir con buffer.
* `NeighborSampler`: usar `num_workers > 0`, `use_uva=True` con GPU, y `fanouts` asimétricos (`[10, 25]`); igualar el presupuesto de épocas al full-batch antes de comparar.
* ClusterGCN: reducir `batch_size` de clusters (p. ej. 20–50 de 1000) para obtener verdadero mini-batching y ruido de gradiente.
* Betweenness: paralelizar con `networkx` sobre subconjuntos o usar `igraph`/`graph-tool`, que son 10–100× más rápidos.

### 5.3 Buenas prácticas recomendadas

1. **Corrección inmediata de los bugs 1, 4, 7, 15 y 22** (tabla anterior). Para el 15: añadir `!DOCUMENTATION.md` y `!CLAUDE.md` junto a `!README.md` (o dejar de ignorar `*.md`).
2. **Tests con `pytest`** (carpeta `tests/`), como mínimo:
   * Inversión de etiquetas (`_reverse_relationship_value`, `add_atributes_edges`): para toda arista `(u→v)=1` existe `(v→u)=2` y viceversa.
   * `split_edges_classification`: ningún par no dirigido cruza splits; sumas de máscaras = aristas etiquetadas; `strict_equal` produce clases iguales.
   * `_log_minmax_scale` y `_check_range`: rangos, columnas constantes, NaN.
   * Forma de salida de cada encoder/decoder con un grafo sintético de 10 nodos (detectaría los bugs 4 y 5).
   * `prune_leaves` y `remove_low_degree_nodes` sobre un grafo de juguete.
3. **Configuración declarativa**: `argparse` en los scripts (`--data-path`, `--rib`, `--year`, `--month`, `--caida`, `--country`) y un `config.yaml`/Hydra para hiperparámetros; eliminar rutas absolutas y usar `pathlib`.
4. **Tipado estático** (`mypy --strict` progresivo) y `ruff`/`black` con `pre-commit`; añadir `from __future__ import annotations` donde se usa `int | None`.
5. **Logging** (`logging` con niveles y `RichHandler`/archivo) en lugar de `print`; registrar métricas por época en CSV/TensorBoard o MLflow y persistir `GLOBAL_CONFIG` junto a cada modelo.
6. **Reproducibilidad**: *(hecho: `modules/experiment.py :: set_seed`)*. Pendiente: guardar los `eids` de cada split a disco.
7. **Estructura de paquete**: `pyproject.toml` con `packages = ["modules", "src"]`, `requirements.txt` fijado, borrar `__pycache__` del índice (`git rm -r --cached '**/__pycache__'`), eliminar `src/graph_builder.py` y `modules/bgp2vec/`, o moverlos a `legacy/`.
8. **Refactor de modelos**: una clase `GNNEncoder(conv_type, num_layers, ...)` que construya la pila con `nn.ModuleList`, más decoders independientes; `BilinearPredictor` con inicialización `xavier_uniform_` escalada.
9. **Documentar el protocolo de evaluación** (transductivo, balanceo, semilla, épocas) en el README y en los reportes de resultados.

### 5.4 Ideas para futuras características y experimentos

* **Escenario inductivo**: eliminar las aristas de test del grafo antes de `encode` (como en link prediction, pero corrigiendo el bug 3) y reportar ambos regímenes.
* **Encoders que consumen features de arista** (`EGATConv`, `EdgeConv`, `GINEConv`) para explotar `weight` en el message passing, no solo en el decoder.
* **GNN dirigidas** (Dir-GNN, MagNet) o grafo heterogéneo con dos tipos de arista (`observed`, `reverse`) para modelar explícitamente la asimetría en lugar de duplicar aristas.
* **Semi-supervisión / pseudo-etiquetado** con las ~74 k aristas `-1`: el modelo ya las predice; usar predicciones de alta confianza como etiquetas adicionales y validar con AS-Rank/ProbLink/TopoScope.
* **Evaluación estratificada**: por tier (transit-free vs stub), por región (Chile/LATAM vs global), por grado del par y por cobertura de PeeringDB (`peeringdb_coverage_bias.ipynb` ya identifica el sesgo).
* **Dimensión temporal**: `modules/graph.py` fue pensado para 12 snapshots mensuales; reactivar esa idea con un `Batch` de PyG por mes o con GNN temporales (EvolveGCN, TGN) para detectar cambios de relación.
* **Combinar embeddings**: concatenar embeddings estructurales (DeepWalk/BGP2Vec) con los de la GNN, o usarlos como `feat` inicial en lugar de PeeringDB para nodos sin cobertura.
* **Calibración y explicabilidad**: temperatura sobre los logits, `GNNExplainer` para identificar qué vecinos/atributos determinan una predicción C2P.
* **Baselines clásicos**: Gao, AS-Rank, ProbLink y un `XGBoost` sobre features de par (`[x_u ‖ x_v ‖ grado, peso]`) para cuantificar la ganancia real de la GNN.
* **Ablaciones sistemáticas**: PeeringDB vs centralidades vs ambos, `edge_feat` on/off, profundidad 1–4, `num_heads`, pesos de clase, `balance_mode`, automatizadas sobre las funciones de entrenamiento y `config.yaml`.

---

## 6. Pendientes

Estado al 2026-11-24. Cada punto dice qué falta, por qué importa y dónde está. Lo urgente va primero.

### 6.1 Datos y entorno

| # | Pendiente | Por qué | Dónde |
|---|---|---|---|
| 1 | **Respaldar el disco `KINGSTON`** y revisarlo con `sudo smartctl -a /dev/sda` | El sector 783 281 152 no se puede leer (error de E/S repetible). Afecta a `resultados_gnn/prueba e2e/end_to_end/caso2_neighbor/`, una prueba vieja; lo de `experimento7_20250916T0800/` se lee bien. Copiar `resultados_gnn/` y `ejecuciones/` (≈ 4,2 GB) | `/media/vale/KINGSTON/LAB/gnn-as-relationships/` |
| 2 | Convertir `caida.bin` (DGL) a `caida_pyg.pt` | Sin ese archivo el notebook por partes lanza `FileNotFoundError` en el caso 2 y no corren los casos 3, 4, 5.1 y 5.2 | `InternetGNNData2022/caida/` · por partes, celda 36 |
| 3 | Arreglar los lanzadores de `env312/bin/` (`jupyter`, `jupyter-lab`, …) | Apuntan a la ruta anterior del repositorio (`…/LABORATORIO/gnn-as-relationships/`). Mientras tanto: `env312/bin/python -m jupyterlab`. El kernel `env312` se volvió a registrar el 2026-11-13 | `env312/bin/` |
| 4 | RIB de BGP2Vec de la misma fecha que el grafo | `rutas.rib_file_bgp2vec` usa marzo de 2026 (`FIXME`) para un grafo de septiembre de 2025 | `config.yaml` |
| 5 | CAIDA de septiembre de 2019 a 2024 | Necesario para etiquetar los grafos 2018–2024 y extender la validación temporal. Hay que recuperar `label_edges_from_caida` (en git, `d1495d9`) | `CAIDA_AS_Relationships/Serial_1/` |

### 6.2 Barrido de `pruebas.yaml`

| # | Pendiente | Por qué | Dónde |
|---|---|---|---|
| 6 | Completar `splits` 0,7/0,15 · K-fold 5 | Se cortó el 2026-11-18 a las 23:43 con 236 de 240 modelos (faltan GAT2L, GCN3L, GraphSAGE3L y GAT3L con Bilinear, fold 4, caso 4). No quedó en `estado.csv`, así que `--reanudar` la repite entera (≈ 8–9 h) | `ejecuciones/estado.csv` |
| 7 | Correr las 8 ejecuciones que no empezaron | `splits` K-fold 10 (0,7/0,15), y holdout, K-fold 5 y K-fold 10 (0,6/0,2); las 4 variantes de `lr_hidden` | `pruebas.yaml` |
| 8 | Marcar `ok` (o quitar `por_partes` de `semillas`) en las semillas 45 y 46 | Siguen en `error` por el caso 2; con `--reanudar` se vuelven a correr (≈ 2,6 h cada una) sin resultado nuevo mientras falte `caida_pyg.pt` | `ejecuciones/estado.csv` |
| 9 | Generar `resumen_pruebas.csv` | Se escribe al terminar todas las pruebas y la última sesión se cortó antes | `ejecuciones/` |
| 10 | Revisar el diseño de `pruebas.yaml` | El holdout 0,7/0,15 de `splits` repite exactamente `semillas__seed-42`. K-fold con semilla fija mide solo la varianza de la partición (≈ 4 veces menor que la de semillas): conviene agregar `comunes.seed` al barrido. `hidden_dim` no cambia los GAT (usan `gat_hidden_dim`): en `lr_hidden` hay 32 corridas repetidas | `pruebas.yaml` |

### 6.3 Código que afecta resultados

| # | Pendiente | Por qué | Dónde |
|---|---|---|---|
| 11 | Corregir `features_arista` y repetir la Etapa 2 de por partes | `nodo_dst = embeddings[edge_index[0]]` toma otra vez el nodo de origen: el clasificador nunca ve el destino. Afecta a todas las cifras por partes | por partes, celda 16 |
| 12 | Descomentar el `save_run` del caso 0 por partes | Hoy el clasificador se entrena pero no se guarda; sus métricas solo están en el texto de los notebooks ejecutados | por partes, celda 30 |
| 13 | Unificar `Caso0/Caso1/Caso2` con `caso0/caso1/caso2` | La Etapa 1 guarda con mayúscula y `save_run` con minúscula. En Windows es la misma carpeta; en Linux, dos | por partes, celdas 30, 34 y 38 |
| 14 | Evaluar ClusterGCN sobre todo el test e igualar el presupuesto del caso 2 | El caso 3 evalúa ≈ 1/3 del test; el caso 2 corre 10 épocas de mini-batch frente a 300 full-batch | end-to-end, celdas 8, 9 y 31 |
| 15 | Sincronizar el notebook end-to-end con la copia de Windows | La copia que corre en Windows importa `modules/metis_windows.py` en la celda 2 (sin eso el caso 3 no corre allí); la del repositorio no la tiene | end-to-end, celda 2 |

### 6.4 Benchmark (`BenchmarckASRelationships`)

| # | Pendiente | Por qué | Dónde |
|---|---|---|---|
| 16 | Que los notebooks del benchmark lean `…/pruebas/semillas/<variante>/tor/` | Hoy leen `TESIS/data_2026/{x,y}_{training,test}.npy`, de CAIDA de marzo de 2026, con fuga entre direcciones (89 % de los enlaces de test también en train) y otro test que la GNN. El ToR nuevo (§4.3) usa el mismo split que la GNN | `BenchmarckASRelationships/*/*.ipynb` (`PATH`, `DATA`) |
| 17 | Rutas BGP del 2025-09-16 para Gao, Ruan, ProbLink y TopoScope | Infieren desde rutas en texto y hoy usan el RIB de marzo de 2026. La carpeta del grafo solo tiene aristas agregadas (`parciales/`), no los AS_PATH | `as-level-internet-graph` |
| 18 | Exportar ToR también en K-fold | `build_tor_dataset` solo se llama en holdout; para comparar fold a fold habría que exportar `tor/fold<i>/` | end-to-end, celdas 18 y 20 |

### 6.5 Setting inductivo (plan, sin implementar)

| # | Pendiente | Detalle |
|---|---|---|
| 19 | Split por AS nuevos | `GNN.split_nodes_inductive(frac_nodos_test, frac_nodos_val, frac_nodos_train, frac_etiquetas, seed)` con `msg_train_mask`/`msg_val_mask`; `comunes.inductivo` en `config.yaml`; 4 líneas en `entrenar_modelo`; casos 2 y 3 omitidos; pruebas `inductivo_etiquetas` e `inductivo_estructura` |
| 20 | Entrenar con un grafo y testear con otro de otra fecha (2025 ↔ 2026) | `comunes.temporal.grafo_test_dir`, `GNN.mascaras_temporales` (subconjuntos `todos`, `as_nuevos`, `enlaces_nuevos`, `conocidos`, `cambio_relacion`) y la línea base de persistencia. El 77 % de los enlaces de 2026 ya existía en 2025. `AS_relationship_evaluacion_otro_grafo.ipynb` (ignorado por git) ya hace una evaluación entre grafos sin guardar resultados |

### 6.6 Documentación y repositorio

| # | Pendiente | Detalle |
|---|---|---|
| 21 | Actualizar §2 de este documento | El árbol y la tabla de estado todavía listan `modules/graph.py`, `modules/bgp2vec/`, `src/`, `config_nerwito.yaml` y `scripts/` vacío; `metis_windows.py` aparece "sin conectar" aunque la copia de Windows lo usa |
| 22 | Prueba de humo de `CLAUDE.md` | Las cifras son del grafo 2026-09-16; `config.yaml` usa el de 2025-09-16 (85 522 nodos; holdout 647 790 / 138 812 / 138 810) |
| 23 | Commit y push | Los cambios del 2026-11-24 (limpieza de funciones sin uso, `build_tor_dataset` nuevo, esta sección) están sin commit; el commit `a008fb4` (`.gitignore`) no está en GitHub |
| 24 | `balance_mode` | Nada pendiente de código: todo corre en `proportional`. `strict_equal` sigue disponible en `split_edges_classification` y `split_edges_kfold`, pero ningún experimento lo usa desde 2026-10-17 |
