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

Este repositorio es el código de una tesis (NIC Chile Research Labs / DCC, Universidad de Chile; ver `build/comment.cut`) que aborda el problema como **clasificación de aristas multiclase sobre el grafo AS-level de Internet**:

1. Se construye la topología a partir de tablas de enrutamiento BGP (RIBs de RouteViews / RIPE RIS, vía BGPStream). Desde octubre de 2026 esto lo hace el repositorio hermano [`as-level-internet-graph`](https://github.com/valesteban/as-level-internet-graph).
2. Cada AS (nodo) se enriquece con atributos estructurales (grados, PageRank, k-core) y de negocio (PeeringDB). También lo hace `as-level-internet-graph`.
3. Las aristas se etiquetan en este repositorio con el dataset de referencia **CAIDA AS-Relationships** (`as-rel.txt.bz2`), mediante `GNN.label_edges_from_caida`.
4. Una GNN aprende *embeddings* de los AS y un *decoder* de aristas predice la clase P2P / C2P / P2C.

Se exploran dos familias de experimentos, cada una en su notebook:

* **End-to-end** (`AS_relationship_inference_end_to_end.ipynb`): el encoder GNN y el decoder de aristas se entrenan conjuntamente con la pérdida de clasificación.
* **Por partes** (`AS_relationship_inference_por_partes.ipynb`): primero se aprenden embeddings con una tarea auxiliar (link prediction, regresión de grado/PageRank, DeepWalk, BGP2Vec) y luego un MLP independiente clasifica pares de AS.

### 1.2 Enfoque basado en GNN y representación de las relaciones

**Grafo.** El grafo es un `torch_geometric.data.Data` dirigido y homogéneo:

* **Nodos** = AS, indexados por `node_id` (entero contiguo 0..N-1) con un mapeo `asn ↔ node_id` guardado en el CSV de nodos.
* **Aristas** = adyacencias observadas en AS_PATHs. Cada enlace no dirigido se materializa como **dos aristas dirigidas** `(u→v)` y `(v→u)`. Esto es esencial porque la etiqueta es *direccional*: si `(u→v)` es C2P (`1`), entonces `(v→u)` es P2C (`2`); P2P (`0`) es simétrica. La regla de inversión `{0:0, 1:2, 2:1}` está implementada en `GNN._reverse_relationship_value` y la usan `_make_edges_bidirectional` (al cargar) y `label_edges_from_caida` (al etiquetar) en `modules/gnn.py`.
* `data.x`: matriz de atributos de nodo (85 columnas con los grafos de `as-level-internet-graph`; 72 con el grafo antiguo de marzo 2026; ver §3.2).
* `data.edge_index`: aristas dirigidas `(2, E)`.
* `data.edge_label`: etiqueta `0/1/2/-1` (`torch.long`).
* `data.edge_attr`: peso normalizado de la arista (frecuencia con la que el par aparece en los AS_PATH), forma `(E, 1)`.
* `data.train_mask | val_mask | test_mask`: máscaras booleanas de split (tamaño E); `data.node_train_mask | node_test_mask` para el split de nodos.

**Modelo.** Se sigue el patrón *encoder–decoder* para aristas:

```
h = Encoder(G, X)               # GCN / GraphSAGE / GAT  →  h ∈ R^{N×d}
logits_uv = Decoder(h_u, h_v[, w_uv])   # MLP | Bilinear | DotProduct → R^{3}
```

El *message passing* del encoder agrega información de la vecindad de cada AS (quiénes son sus vecinos, cuán centrales son, qué tipo de red declaran en PeeringDB), lo que codifica implícitamente la posición jerárquica del AS. El decoder es **asimétrico** respecto al orden `(u, v)` (concatenación `[h_u ‖ h_v]` en el MLP, forma bilineal `h_uᵀ W_c h_v` en el Bilinear), de modo que puede distinguir C2P de P2C. El `DotProductPredictor` es simétrico y se usa solo para link prediction (tarea binaria).

### 1.3 Stack tecnológico

Versiones verificadas en el entorno `env310` de la máquina de desarrollo (la versión DGL original corría con `DGL 1.1.3 | PyTorch 2.3.0+cu121`):

| Componente | Versión / detalle | Uso |
|---|---|---|
| Python | 3.10.12 (kernel Jupyter `env310`) | Todo |
| PyTorch | 2.3.0 (+cu121) | Modelos, entrenamiento |
| **PyTorch Geometric** | **2.8.0** (+ `pyg_lib`, `torch_scatter`, `torch_sparse`, `torch_cluster`) | `Data`, `GCNConv`, `SAGEConv`, `GATConv`, `LinkNeighborLoader`, `ClusterData`/`ClusterLoader`, `Node2Vec` |
| NetworkX | (sin fijar) | Pipeline legado `modules/graph.py` |
| pandas / numpy | 2.3.3 / 2.2.6 | I/O CSV, normalización |
| scikit-learn | 1.7.2 | Métricas (F1, matriz de confusión, ROC-AUC), `train_test_split`, `compute_class_weight` |
| gensim | 4.4.0 | Word2Vec para BGP2Vec (`modules/bgp2vec.py`) |
| pybgpstream | 2.0.4 (`_pybgpstream`) | Ya no se usa aquí: la descarga de RIBs pasó a `as-level-internet-graph` |
| requests, urllib | 2.25.1 | API PeeringDB y LACNIC (`src/`) |
| PyYAML | 5.4.1 | Solo `modules/graph.py` (legado) |
| tqdm, matplotlib, seaborn, umap-learn, scipy | — | Progreso y visualización en notebooks |

> **Ya no se usa DGL.** Toda la parte de grafos está construida sobre PyG. No existe `requirements.txt` ni `environment.yml`; en §4.1 se propone uno.

---

## 2. Estructura del Repositorio

```
gnn-as-relationships/
├── README.md                                   # Descripción, ubicación de datos, instalación y flujo de trabajo
├── CLAUDE.md                                   # Contexto resumido del proyecto para Claude Code
├── DOCUMENTATION.md                            # este documento
├── .gitignore                                  # ignora *.csv, *.md, *.json, *.txt, data/*, results/* (ver §5)
│
├── AS_relationship_inference_end_to_end.ipynb  # ★ Entrenamiento end-to-end (encoder GNN + decoder de aristas)
├── AS_relationship_inference_por_partes.ipynb  # ★ Enfoque en dos etapas: embeddings + clasificador ANN
│
├── modules/                                    # Código reutilizable importado por los notebooks
│   ├── gnn.py            # ★ Clase GNN: etiquetado CAIDA de gnn_edges.csv, carga CSV→Data (PyG), splits, features aleatorias, poda
│   ├── gnn_models.py     # ★ Encoders GCN/GraphSAGE/GAT (2L, 3L, *Sample2L) y decoders MLP/Bilinear/DotProduct
│   ├── experiment.py     # set_seed (reproducibilidad) y save_run (métricas, pesos e historial de cada modelo)
│   ├── graph.py          # Pipeline LEGADO: clase Graph (NetworkX) → dataset multi-grafo mensual (meta.yaml, graphs.csv)
│   ├── bgp2vec.py        # RIBCorpus (streaming de AS_PATHs) + BGP2VEC (Word2Vec skip-gram sobre rutas)
│   └── bgp2vec/          # Versión anterior de BGP2Vec (copia casi idéntica de bgp2vec.py)
│       ├── bgp2vec.py    #   - usa len(self.routes): incompatible con iteradores
│       ├── oix_utils.py  #   - parsers del formato OIX (Oregon RouteViews) del paper BGP2Vec original
│       ├── bgp2vec.word2vec  # modelo Word2Vec entrenado (binario)
│       └── ToR_bgp2vec.ipynb # notebook Keras/TensorFlow del baseline BGP2Vec + NN
│
├── scripts/
│   └── correr_experimento.py   # Corre ambos notebooks sobre todos los grafos de un experimento (§4.7)
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
│   ├── attribute_inspection.ipynb           # Tabla resumen de atributos, correlación Spearman, distribuciones por clase
│   ├── validar_normalizacion_atributos.ipynb# Verifica que todos los atributos estén en [0,1]
│   ├── peeringdb_coverage_bias.ipynb        # Sesgo de cobertura de PeeringDB por grado/tier
│   ├── AS_embeddings_visualization.ipynb    # t-SNE / UMAP de embeddings por caso
│   ├── tsne_embeddings.ipynb                # t-SNE comparativo (GNN vs DeepWalk vs BGP2Vec)
│   └── results/*.pdf                        # Figuras exportadas (pdb_*, tsne_embeddings)
│
└── build/                                      # Artefactos de compilación LaTeX de un paper (paper.aux/.log/.out…)
                                                # El .tex NO está en el repo; el .out revela la estructura del paper
```

**Estado de cada componente**

| Componente | Estado | Comentario |
|---|---|---|
| `modules/gnn.py`, `modules/gnn_models.py`, `modules/experiment.py` | **Activo** | Núcleo del proyecto |
| `scripts/1..4_*.py`, `scripts/create_tor_dataset.py` | **Eliminados** (commit `4fcb754`) | La construcción del grafo pasó a `as-level-internet-graph` y el etiquetado CAIDA a `GNN.label_edges_from_caida`. Siguen en el historial de git y en `main` / `v1.0-dgl`. |
| `data/` (grafo de marzo 2026) | **Eliminado** del repo | Copia en `/media/vale/KINGSTON/TESIS/data_2026/` (ver §4.3) |
| `src/*.py` | Sin uso | Utilidades que usaban los scripts eliminados |
| `modules/bgp2vec.py` | Activo (enfoque por partes) | |
| `scripts/correr_experimento.py` | **Activo** | Ejecución por lotes de los notebooks sobre un experimento (§4.7) |
| `modules/graph.py` | **Legado** | Formato multi-grafo mensual; ya no lo usan los notebooks. Contiene bugs (§5.1) |
| `modules/bgp2vec/` | Legado | Duplicado de `modules/bgp2vec.py` + baseline Keras |
| `src/graph_builder.py` | Vacío | |
| `__pycache__/` (3 carpetas) | Basura versionada | Deben salir del repo |

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

> Nota: `MLPPredictor` solo concatena `edge_attr` si se le pasa (`edge_attr is not None`); si el modelo se construyó con `edge_dim=1` y el grafo no tiene `edge_feat`, la dimensión de entrada de `fc1` no coincidirá.

#### 3.1.2 Encoders full-batch (3 capas y 2 capas)

Todas las clases exponen la misma interfaz: `encode(x, edge_index)`, `decodeMLP(edge_index, h, edge_attr=None)`, `decodeBilinear(edge_index, h)`, `decodeDotProduct(edge_index, h)` y `forward(x, edge_index)` (= `encode` + `regressor: Linear(out, 1)`, usado para regresión nodal en el enfoque por partes).

| Clase | Capas PyG | Agregación / atención | Activación entre capas | Self-loops | Dim. de salida del embedding |
|---|---|---|---|---|---|
| `GCN3L(in, hid, out, out_feats_mlp=1, drop=0.3, edge_dim=0)` | `GCNConv×3` | Normalización simétrica de GCN | ReLU + Dropout (no tras la última) | `add_self_loops=True` (por defecto en `GCNConv`) | `out` |
| `GraphSAGE3L(...)` | `SAGEConv×3` (`aggr='mean'`) | Media de vecinos + proyección propia | ReLU + Dropout | No (SAGE no requiere) | `out` |
| `GAT3L(..., num_heads=4)` | `GATConv×3` | Atención multi-cabeza; `concat=True` concatena cabezas tras cada capa | ReLU + Dropout | `add_self_loops=True` (por defecto) | `out × num_heads` |
| `GCN2L`, `GraphSAGE2L`, `GAT2L` | Igual con 2 capas | Igual | Igual | Igual | `out` (GAT: `out×heads`) |

Detalles relevantes:

* `hidden_feats` es el ancho intermedio; en las GAT, la capa `k+1` recibe `hidden × num_heads`.
* Los decoders MLP/Bilinear se instancian dentro del encoder con `out_feats` (o `out_feats × num_heads` en GAT) y `out_feats_mlp` = nº de clases.
* La dimensión de embedding para el enfoque por partes es `out_feats` (128 en `GLOBAL_CONFIG`), o 256 en GAT con 2 cabezas.

#### 3.1.3 Encoders con muestreo (`GCNSampler2L`, `GraphSAGESample2L`, `GATSample2L`)

Diseñados para entrenamiento mini-batch. En PyG tanto `LinkNeighborLoader` (neighbor sampling) como `ClusterLoader` (ClusterGCN) entregan un `Batch` con `x` y `edge_index` locales, por lo que `encode(x, edge_index)` es idéntico al de los encoders full-batch:

```python
h = relu(conv1(x, edge_index)); h = conv2(h, edge_index)
```

* No hay dropout en estas variantes; `GCNConv`/`GATConv` agregan self-loops por defecto, lo que también cubre los nodos sin vecinos de entrada de los subgrafos muestreados (el `allow_zero_in_degree` de DGL ya no es necesario).
* Solo exponen `decodeMLP`; `MLPPredictor` se crea con `edge_dim=1` por defecto.
* En `GATSample2L` las cabezas quedan concatenadas tras cada capa (`concat=True`): `conv2` recibe `hidden × heads` y el `MLPPredictor` `out × heads`.

### 3.2 Pipeline de procesamiento de datos

La construcción y el enriquecimiento del grafo se hacen en el repositorio hermano [`as-level-internet-graph`](https://github.com/valesteban/as-level-internet-graph). Este repositorio solo etiqueta las aristas con CAIDA y entrena. El pipeline anterior (`scripts/1..4_*.py`, que producía `nodes_rib_*_enriched_tesis.csv` / `edges_rib_*_enriched_tesis.csv`) se eliminó en el commit `4fcb754`. Sigue disponible en el historial de git y en `main` / `v1.0-dgl`.

```
 RouteViews / RIPE RIS (BGPStream, todos los recolectores, ventana de 15 min)
          │  as-level-internet-graph:  python as_graph.py … ;  scripts/enriquecer_nodos.py
          ▼
 GRAFO_DIR = /media/vale/KINGSTON/as-level-internet-graph/experimentos/experimento<N>/grafos/<año>/<mes>/<AAAAMMDDTHHMM>/
   ├─ as_features.csv   node_id, asn, +85 features en [0,1]   (node_id = fila)
   ├─ gnn_edges.csv     src_id, dst_id, src_asn, dst_asn, weight, weight_raw   (1 fila por enlace no dirigido)
   └─ stats.json, as_nodes.csv, as_edges.csv, as_graph.graphml, …
          │  GNN.label_edges_from_caida(gnn_edges.csv, CAIDA AAAAMM01.as-rel.txt.bz2, output_csv=…)
          ▼
   ├─ gnn_edges_caida.csv     gnn_edges.csv + relationship {0,1,2,-1}   (mismas filas, mismo orden)
   └─ etiquetas_caida.json    resumen del etiquetado
          │  GNN.load_dataset(as_features.csv, gnn_edges_caida.csv)
          ▼
 torch_geometric.data.Data  x (N×85) · edge_index (2×E) · edge_label (E) · edge_attr (E×1)
          │  GNN.split_edges_classification()  /  split_edges_link_prediction()  /  split_graph_nodes()
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

CAIDA publica un snapshot de AS-Relationships (serial-1) el día 1 de cada mes. Se usa el del **mes del grafo**:
* `GNN.caida_file_for_graph(GRAFO_DIR, CAIDA_DIR)` lee la fecha del nombre de la carpeta (`AAAAMMDDTHHMM`) y devuelve `CAIDA_DIR/AAAAMM01.as-rel.txt.bz2`.
* Si el archivo no existe, lanza `FileNotFoundError` con la URL de descarga.
* `CAIDA_DIR` actual: `/media/vale/KINGSTON/TESIS/data_2026/` (hay `20260301` y `20260901`).

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

   Por convención la salida es `GRAFO_DIR/gnn_edges_caida.csv`, en el disco externo. `gnn_edges.csv` no se modifica. Los notebooks etiquetan automáticamente si `gnn_edges_caida.csv` no existe.

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
5. Guarda `self.asn_to_node_id` para `_fill_labels_from_caida_stream_fast`.

`load_dataset_only_cntrality_attr(...)`: igual, pero las features son solo las columnas estructurales presentes en el CSV, **estandarizadas con z-score**. En el formato nuevo son `degree`, `weighted_degree`, `transit_degree`, `pagerank` y `k_core`; en el antiguo, `weight`, `PageRank`, `degree_centrality`, `betweenness_centrality` y `eigenvector_centrality`. El z-score deja media 0 y desviación 1; las columnas constantes usan std=1. Comentario en el código: sin esto la GNN colapsa por las escalas dispares.

`_fill_labels_from_caida_stream_fast(caida_file)`: alternativa in-memory a `label_edges_from_caida`, que los notebooks no usan. Recorre el `.bz2/.gz/.txt` de CAIDA, etiqueta aristas existentes (`eid_map` dict `(u,v)→eid`), agrega aristas faltantes y crea nodos nuevos con `feat = 0` para ASNs ausentes.

Otras utilidades: `add_random_features(dim, std, seed, mode∈{zscore,minmax,uniform})` (control experimental), `remove_low_degree_nodes(degree, iterations)` (poda con `torch_geometric.utils.subgraph` y remapeo de `asn_to_node_id` con los ids originales conservados).

#### 3.2.5 Estrategias de split

| Método | Tarea | Cómo evita fugas | Salida |
|---|---|---|---|
| `split_edges_classification(train_size=0.7, val_size=0.15, seed, balance_mode)` | Clasificación de aristas | Agrupa eids por **par no dirigido** `(min(u,v), max(u,v))` → las dos direcciones caen en el mismo split. Estratifica por *firma* de etiquetas del par (`(0,0)`, `(1,2)`, `(2,)`…) y reparte cada estrato con las proporciones dadas. `balance_mode="strict_equal"` submuestrea cada split al mínimo por clase | `edata[*_mask]`, `self.train_eids/val_eids/test_eids` |
| `split_edges_link_prediction(train_ratio=0.8, seed=42)` | Link prediction | Baraja eids; `train_g = remove_edges(g, test_eids)`; subgrafos `train_pos_g/test_pos_g`; negativos por muestreo uniforme de pares sin arista (`g.has_edges_between`, hasta 3 rondas de sobre-muestreo) | `self.train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g` |
| `split_graph_nodes(train_size=0.8)` | Regresión nodal | `randperm` de nodos | `ndata['train_mask'|'test_mask']` |
| `split_edges_classification_leaky`, `split_edges_classification_v0` | Legado | Requieren `edata['Relationship']` (no lo produce `load_dataset`) | — |

Cifras con `experimento7 / 20260916T0800` + CAIDA `20260901`: 987 964 aristas dirigidas etiquetadas (P2P 659 178 · C2P 164 393 · P2C 164 393); con `proportional` 0.7/0.15 y `seed=42` → train 691 574 / val 148 194 / test 148 196.

Cifras del grafo antiguo de marzo (outputs guardados): 660 438 aristas etiquetadas; con `proportional` 0.7/0.15 → train 462 306 / val 99 064 / test 99 068; con `strict_equal` 0.8/0.15 → train 383 910 (127 970 por clase) / val 71 982 / test 23 997 (7 999 por clase).

### 3.3 Esquemas de entrenamiento

#### 3.3.0 Reproducibilidad y guardado (`modules/experiment.py`)

**Semilla.** `set_seed(seed=42, deterministic=True)`:
* fija `random`, `numpy`, `torch` (CPU y CUDA) y PyG (`torch_geometric.seed_everything`);
* con `deterministic=True`, además fija `cudnn.deterministic=True` y `cudnn.benchmark=False`, activa `torch.use_deterministic_algorithms(True, warn_only=True)` y define `CUBLAS_WORKSPACE_CONFIG`.

Los dos notebooks definen `SEED = 42` en la celda de configuración y la usan en todos los splits (`seed=SEED`). Además llaman a `set_seed(SEED)`:
* al inicio de cada iteración del bucle de modelos, es decir, antes de crear cada modelo;
* en por partes, también antes de `split_graph_nodes`, de `Node2Vec` y del clasificador ANN.

Así cada modelo parte siempre del mismo estado: pesos iniciales, dropout y orden de los mini-batches de `LinkNeighborLoader`/`ClusterLoader` son los mismos. Esto vale aunque se ejecute solo esa celda o se comenten otros modelos.
* **Comprobado en CPU:** dos procesos distintos y una segunda ejecución de la celda sin repetir el split dan métricas idénticas (Caso 0 y Caso 2).
* **En GPU** algunas operaciones scatter de PyG no son deterministas: `warn_only` solo avisa, y los resultados pueden variar levemente.
* **BGP2Vec** usa `seed=7` y `workers=1` en `modules/bgp2vec.py`. gensim además depende de `PYTHONHASHSEED`.

**Guardado (solo end-to-end).** `save_run(RESULTS_DIR, caso, modelo, decoder, seed=…, y_true, y_pred, class_names, state_dict, history, config, extra)` se llama después de evaluar cada modelo en test. Escribe en `RESULTS_DIR/<caso>/<modelo>_<decoder>_seed<SEED>/`:
* `metrics.json`: accuracy, macro-F1, F1 ponderado, reporte por clase, matriz de confusión, `GLOBAL_CONFIG`, semilla, `RUN_TAG`, tiempo, épocas y mejor métrica de validación;
* `model.pth`: los pesos de la mejor época en validación;
* `history.csv`: las curvas por época.

También agrega una fila a `RESULTS_DIR/resumen.csv`. `RESULTS_DIR = GRAFO_DIR/results/end_to_end`. Los valores de `caso` son `caso0`, `caso1`, `caso2_neighbor`, `caso3_cluster`, `caso4_random` y `comparacion`. La comparación guarda solo métricas, más `tabla_comparacion_seed<SEED>.csv`. Repetir una corrida con la misma semilla sobrescribe su carpeta, porque el resultado es el mismo.

**Identificador de configuración.** `<config_id>` es un hash de 6 caracteres de `config` (`GLOBAL_CONFIG`) y de `extra["clase"]` (la clase del modelo, p. ej. `GAT3L`, que fija el número de capas); en el clasificador ANN también de `extra["config_embedding"]` y `extra["clase_embedding"]` (los del GNN que generó el embedding). Re-correr con la misma configuración sobrescribe la carpeta (el resultado es idéntico); cambiar hiperparámetros o capas crea una carpeta nueva al lado, así que las corridas no se pierden. Otro grafo nunca choca: cada uno escribe en su `GRAFO_DIR/results/`. Las carpetas antiguas sin hash siguen siendo legibles. `resumen.csv` y `metricas.csv` incluyen `config_id` (y `metricas.csv` también `clase`). La función es `modules.experiment.config_id`.

**Tiempos.** Las celdas miden el entrenamiento con `GNN.timer(nombre)` (o `time.perf_counter`) y pasan `tiempo_s` y `epocas` en `extra`; `resumen.csv` y `metricas.csv` los incluyen como columnas `tiempo_s`, `tiempo_total_s` y `epocas`. En por partes, cada modelo que genera embeddings (Caso0–Caso4, DeepWalk, BGP2Vec) guarda su duración en `<embedding>.tiempo.json` con `GNN.save_tiempo_embedding`; el clasificador ANN la lee con `GNN.load_tiempo_embedding` y reporta `tiempo_s` (solo ANN), `tiempo_embedding_s` y `tiempo_total_s` (suma), para comparar con el tiempo end-to-end. Si el embedding es antiguo y no tiene `.tiempo.json`, `tiempo_embedding_s` queda vacío.

#### 3.3.1 End-to-end (`AS_relationship_inference_end_to_end.ipynb`)

Configuración global (celda 7):

```python
GLOBAL_CONFIG = {
  'lr': 0.001, 'hidden_dim': 256, 'out_dim': 128, 'epochs': 500, 'patience': 120,
  'weight_decay': 5e-4, 'dropout': 0.3, 'label_smooth': 0.05, 'grad_clip': 2.0,
  'num_heads': 2, 'batch_size': 1024, 'num_neighbors': 15, 'focal_gamma': 2.0,
}
```

Pérdida auxiliar definida en el notebook:

```python
def focal_loss(logits, targets, alpha=None, gamma=2.0):   # CE ponderada × (1 - p_t)^gamma
    ce = F.cross_entropy(logits, targets, weight=alpha, reduction='none')
    with torch.no_grad(): p_t = softmax(logits).gather(targets); w = (1 - p_t) ** gamma
    return (w * ce).mean()
```

| Caso | Features de nodo | Split | Modelos | Decoder / pérdida | Optimización |
|---|---|---|---|---|---|
| **0** PeeringDB | 72 (`load_dataset`) | `proportional` 0.7/0.15, seed 42 | `GraphSAGE3L`, `GAT3L` (GCN comentado) | MLP (`edge_dim=1`) + `focal_loss(alpha=class_weights, γ=2)`; Bilinear + CE con `label_smoothing=0.05` (código comentado en la versión actual) | Adam (wd 5e-4) + `CosineAnnealingWarmRestarts(T_0=100, T_mult=2)`; `clip_grad_norm_(2.0)`; early stopping por **macro-F1 de validación** (desempate por loss), `patience=120` |
| **1** Centralidad | 5 (z-score) | `strict_equal` 0.8/0.15 | `GraphSAGE3L`, `GAT3L` | Bilinear + CE ponderada + label smoothing (`T_0=50`); MLP + focal loss (`T_0=100`) | Igual |
| **2** NeighborSampling | 72 | `strict_equal` 0.8/0.15 | `GraphSAGESample2L`, `GATSample2L` (heads=4 por defecto) | MLP + `focal_loss` sin pesos; validación con CE | `NeighborSampler([15,15], prefetch_node_feats=['feat'])` + `as_edge_prediction_sampler(prefetch_labels=['label'])`, `DataLoader` sobre `train_eids`, batch 1024; Adam + `ReduceLROnPlateau(patience=8, factor=0.5)`; **15 épocas fijas**; early stopping por loss val |
| **3** ClusterGCN | 72 | `strict_equal` | `GCNSampler2L`, `GraphSAGESample2L`, `GATSample2L` | MLP + `focal_loss` sobre `subg.edata['train_mask']` | `ClusterGCNSampler(g, num_parts=1000, prefetch_edata=['label', *masks])`, batch 1024 clusters (⇒ 1 batch por época), 15 épocas |
| **4** Features aleatorias (control) | 64 uniformes `U[0,1]` (`add_random_features(mode="uniform", seed=42)`) | `strict_equal` | `GCN3L`, `GraphSAGE3L`, `GAT3L` | MLP + CE sin pesos | Adam; **early stopping sobre el test set** (no hay val) |
| Comparación | 72 (último `gnn` cargado) | — | 2L/3L × GCN/SAGE/GAT × {MLP, Bilinear} | CE ponderada + label smoothing | `_train_one`; la celda quedó interrumpida (`KeyboardInterrupt`) tras GCN-2L+MLP |

Evaluación: `load_state_dict(best_w)`, `argmax` sobre `test_mask`, `confusion_matrix`, `classification_report`, baseline de clase mayoritaria y curvas loss/accuracy.

Resultados registrados en los outputs del notebook (test):

| Caso | Modelo | Accuracy | Macro-F1 | Tiempo |
|---|---|---|---|---|
| 0 | GraphSAGE3L + MLP | 94.41 % | 93.61 % (baseline mayoritaria 51.56 % / 22.68 %) | 44 min (CPU) |
| 1 | GCN3L + Bilinear | 86.38 % | 86.34 % | 18 min |
| 2 | SAGE-Sampler / GAT-Sampler | 91.04 % / 88.54 % | 91.05 % / 88.53 % | 3.9 / 7.8 min |
| 3 | GCN / SAGE / GAT-Sampler (ClusterGCN) | 89.73 / 89.24 / 86.73 % | 86.88 / 87.20 / 84.63 % | 16–19 s |
| 4 | GCN / SAGE / GAT (features aleatorias) | 70.09 / 85.34 / 51.59 % | 70.12 / 85.60 / 42.20 % | 12 / 18 / 8 min |
| Comp. | GCN-2L + MLP | 91.4 % | 91.4 % | 918 s |

#### 3.3.2 Por partes (`AS_relationship_inference_por_partes.ipynb`)

`GLOBAL_CONFIG` propio: `lr 5e-4, hidden 256, out 128, dropout 0.3, epochs 500, patience 80, weight_decay 5e-4, label_smooth 0.05`.

**Etapa A — embeddings** (guardados en `<data_path>/results/EnfoquePorPartes/Caso{N}/`):

| Caso | Tarea auxiliar | Modelos | Pérdida / métrica | Artefactos |
|---|---|---|---|---|
| 0 | Link prediction con features PeeringDB (`split_edges_link_prediction(0.8)`) | `GCN3L`, `GraphSAGE3L`, `GAT3L` | `BCEWithLogits(pos ∪ neg)`; AUC; umbral óptimo por distancia a (0,1) en ROC | `embeddings_ribs_{DotProduct\|MLP}_{modelo}_mis_attr_marzo.pt`, `model_emb_*.pth`, `roc_*.png` |
| 1 | Link prediction con features de grado | idem | idem | `…_grado_attr_febrero.pt` |
| 2 | Link prediction sobre grafo externo `InternetGNNData2022/caida/caida_pyg.pt` (`torch.load`; el `caida.bin` original es binario DGL y debe convertirse una vez) | idem | idem | `…_2022_attr.pt` |
| 3 | Regresión de **out-degree** (`log1p + minmax`) con `split_graph_nodes(0.8)` + 20 % de train como val | `forward()` (regressor) | `nn.L1Loss` (MAE), `r2_score` | `embeddings_ribs_{modelo}_out_degree.pt` |
| 4 | Regresión de **PageRank** calculado con message passing explícito (`index_add_` sobre `edge_index`, `DAMP=0.85, K=20`), `log1p + minmax` | idem | MAE | `embeddings_ribs_{modelo}_pagerank_norm.pt` |
| 5.1 | `torch_geometric.nn.Node2Vec(edge_index, embedding_dim=32, walk_length=40, context_size=2, p=q=1, num_negative_samples=5)` (DeepWalk), `SparseAdam(lr=0.01)`, 50 épocas, batch 128 | — | Pérdida propia de Node2Vec | `embeddings_deepWalk_marzo_2026.pt` |
| 5.2 | `BGP2VEC` (Word2Vec skip-gram, `vector_size=32, window=2, negative=5, min_count=1`) sobre `RIBCorpus` (hasta 5 M rutas del RIB de 12 h) | — | — | `bgp2vec.word2vec` |

**Etapa B — clasificador de aristas** (`EdgeClassifierANN`):

* Dataset ToR (`x_training.npy`, `x_test.npy`, `y_*.npy`; pares de ASN como strings). La celda 71 usa `TOR_PATH`:
  * por defecto, el del grafo, `data_path/results/EnfoquePorPartes/ToR/`, creado por `GNN.build_tor_dataset` con los enlaces etiquetados del grafo y un split por enlace no dirigido;
  * si no existe, el antiguo de marzo 2026 (`TESIS/data_2026/ToR_data/tor_data_marzo/`), que generaba `scripts/create_tor_dataset.py` desde todo el archivo CAIDA, con un split 80/20 sin separar direcciones.
* Mapeo ASN → índice de embedding (CSV de nodos para GNN/DeepWalk, `wv.key_to_index` para BGP2Vec); pares con algún ASN fuera del vocabulario se descartan (se reporta la cobertura).
* Vector de arista `[e_u ‖ e_v ‖ e_u − e_v ‖ e_u ⊙ e_v]` (`4·emb_dim`), construido vectorizadamente en `build_edge_features`.
* Modelo: `Linear(4d,256) → BatchNorm → ReLU → Dropout(0.3) → Linear(256,128) → BatchNorm → ReLU → Dropout → Linear(128,3)`.
* Entrenamiento: `CrossEntropyLoss(weight=compute_class_weight('balanced'))`, Adam `lr=1e-3`, `ReduceLROnPlateau`, batch 512, 200 épocas, `patience=50`, val = 20 % estratificado del train.
* Evaluación: accuracy, `classification_report`, matrices de confusión (guardadas como PNG), `ann_con_batches_{embeddings_name}.pth`.

### 3.4 Decisiones de diseño y su justificación

| Decisión | Dónde | Justificación técnica |
|---|---|---|
| Grafo dirigido con ambas direcciones y etiqueta invertida | `GNN._make_edges_bidirectional`, `add_atributes_edges` | La relación es direccional; duplicar permite que un decoder asimétrico aprenda C2P y P2C como clases distintas y que el message passing fluya en ambos sentidos. |
| Split por par no dirigido + estratificación por firma | `split_edges_classification` | Si `(u→v)` está en train y `(v→u)` en test, la etiqueta de test se deduce trivialmente. Agrupar por par elimina esa fuga; la firma preserva la proporción de pares P2P vs P2C/C2P en cada split. |
| `balance_mode="strict_equal"` | Casos 1–4 | P2P ≈ 51 % de las aristas; igualar clases evita que el modelo colapse a la mayoritaria y hace comparables los casos. Se combina con `class_weights` cuando se usa `proportional`. |
| `focal_loss` con `γ=2` y pesos de clase | Casos 0–3 | Reduce la contribución de ejemplos fáciles (mayoría de P2P) y enfatiza C2P/P2C difíciles. |
| Normalización `log1p + minmax` en todo atributo numérico | `as-level-internet-graph` (`features.py`); antes `4_add_atributes.py` | Grados, centralidades y prefijos siguen distribuciones de cola pesada (power-law); `log1p` comprime la cola y `minmax` lleva todo a `[0,1]` para convivir con los one-hot. Se valida con `_check_range` y en `validar_normalizacion_atributos.ipynb`. |
| z-score para features de centralidad | `load_dataset_only_cntrality_attr` | Escalas muy distintas (PageRank ~1e-5 vs grado ~1) hacían colapsar el entrenamiento (comentario en el código). |
| `add_self_loop` en GCN/GAT, no en SAGE | `encode()` | `GraphConv`/`GATConv` fallan con nodos de in-degree 0 y se benefician de incluir la propia representación; `SAGEConv` ya concatena la proyección del nodo. |
| Self-loops por defecto en `GCNConv`/`GATConv` | Todos los encoders | Cubren los nodos sin vecinos de entrada de los subgrafos muestreados (equivale al `allow_zero_in_degree` de DGL). |
| Decoder Bilinear con `W ∈ R^{C×F×F}` | `BilinearPredictor` | Modela explícitamente la interacción `h_uᵀ W_c h_v` por clase; asimetría natural para C2P/P2C. Es más costoso (`C·F²` parámetros: 3·128² ≈ 49 k) que el MLP. |
| `edge_feat` (peso) en el MLP decoder | `MLPPredictor(edge_dim=1)` | La frecuencia de un par en los AS_PATH correlaciona con la relación (proveedores aparecen más); se inyecta al decoder porque los encoders DGL usados no consumen features de arista. |
| Features aleatorias (Caso 4) | `add_random_features` | Control experimental: mide cuánto aporta la estructura del grafo por sí sola frente a los atributos (GraphSAGE alcanza 85 % solo con estructura). |
| Betweenness aproximada (`k=100`) | `4_add_atributes.py` (grafo antiguo) | Betweenness exacta es O(N·E) ≈ 80 k × 730 k: inviable; el muestreo con semilla fija es reproducible. El formato nuevo no incluye betweenness. |
| Early stopping por macro-F1 en validación | Casos 0–1 | Con clases desbalanceadas el accuracy es engañoso; macro-F1 penaliza el colapso a P2P. |
| Entrenamiento transductivo (aristas de val/test presentes como estructura) | Todos los casos end-to-end | Las aristas de val/test siguen en el grafo durante el message passing (solo se oculta la etiqueta). Es el escenario real (la topología es observable; la relación no) pero debe explicitarse al reportar resultados. |

---

## 4. Guía de Ejecución y Pipeline (Step-by-Step)

### 4.1 Requisitos de entorno

* **SO**: Linux (desarrollado en Ubuntu, kernel 6.8).
* **Python 3.10** (los notebooks usan el kernel `env310`, Python 3.10.12).
* **CPU/GPU**: el código selecciona `cuda` si está disponible; los outputs guardados muestran ejecución en **CPU** (Caso 0 ≈ 44 min por modelo). Con GPU (CUDA 12.1) los tiempos bajan un orden de magnitud.
* **RAM**: el grafo de 2026-09-16 (88 k nodos, 1,34 M aristas dirigidas, 85 features) cabe en pocos GB para los encoders GCN/SAGE. Los GAT full-batch con `hidden=256` se quedan sin memoria en la máquina de desarrollo (15 GB, sin GPU): para pruebas use dims pequeñas (`hidden=32`, `out=16`).
* Acceso a red solo para descargar el `as-rel` de CAIDA (`publicdata.caida.org`). La construcción del grafo (BGPStream, PeeringDB) ocurre en `as-level-internet-graph`.

### 4.2 Instalación del entorno

No existe archivo de dependencias; los comandos de instalación están comentados en la primera celda del notebook end-to-end. Secuencia recomendada:

```bash
# 1. Clonar y crear entorno
git clone <url-del-repo> gnn-as-relationships
cd gnn-as-relationships
python3.10 -m venv env310
source env310/bin/activate
pip install --upgrade pip

# 2. PyTorch 2.3.0 (CUDA 12.1). Para CPU: --index-url https://download.pytorch.org/whl/cpu
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# 3. PyTorch Geometric + extensiones (LinkNeighborLoader, ClusterData/METIS y Node2Vec las necesitan)
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
#    (CPU: -f https://data.pyg.org/whl/torch-2.3.0+cpu.html)

# 4. Resto de dependencias
pip install "pandas>=2.0" "numpy<3" networkx scikit-learn scipy matplotlib seaborn tqdm \
            gensim==4.4.0 requests pyyaml umap-learn jupyter ipykernel

# 5. Registrar el kernel que esperan los notebooks
python -m ipykernel install --user --name env310 --display-name env310
```

`requirements.txt` propuesto (a añadir al repo):

```
torch==2.3.0
torch_geometric==2.8.0
pyg_lib
torch_scatter
torch_sparse
torch_cluster
pandas==2.3.3
numpy==2.2.6
networkx
scikit-learn==1.7.2
scipy
matplotlib
seaborn
tqdm
gensim==4.4.0
requests
pyyaml
umap-learn
jupyter
```

> **Importante sobre rutas.** Los notebooks tienen rutas absolutas a un disco externo. Todas se definen en su celda de configuración (celda 6 del end-to-end y celda 3 del por partes):
> * `GRAFO_DIR`: carpeta del grafo de `as-level-internet-graph`, `/media/vale/KINGSTON/as-level-internet-graph/experimentos/experimento<N>/grafos/<año>/<mes>/<AAAAMMDDTHHMM>/`. **Para cambiar de grafo basta con cambiar esta variable.**
> * `CAIDA_DIR`: carpeta con los `AAAAMM01.as-rel.txt.bz2`; hoy `/media/vale/KINGSTON/TESIS/data_2026/`.
> * `RIBS_DIR` (solo por partes, BGP2Vec): RIBs en texto (`rib_*.txt`), que `as-level-internet-graph` no genera; hoy las de marzo en `/media/vale/KINGSTON/TESIS/data_2026/`.
> * `TESIS_PATH`: el dataset ToR y el grafo externo 2022 del enfoque por partes siguen en `/media/vale/KINGSTON/TESIS/`.
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
wget -P /media/vale/KINGSTON/TESIS/data_2026/ \
     https://publicdata.caida.org/datasets/as-relationships/serial-1/20260901.as-rel.txt.bz2
```

**Paso 3: etiquetar las aristas.** Los notebooks lo hacen solos en la celda de configuración si `gnn_edges_caida.csv` no existe. A mano:

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

**Grafo antiguo de marzo 2026.** Los CSV `*_enriched_tesis.csv` (72 features, ya etiquetados) están en `/media/vale/KINGSTON/TESIS/data_2026/`. Para usarlos, descomente las líneas correspondientes en la celda de configuración del notebook end-to-end.

**Dataset ToR (solo enfoque por partes).** Se crea por grafo:

```python
GNN().build_tor_dataset(GRAFO_DIR / "gnn_edges_caida.csv", GRAFO_DIR / "results/EnfoquePorPartes/ToR", seed=42)
```

`scripts/correr_experimento.py` lo hace solo (§4.7). El dataset antiguo de marzo sigue en `TESIS/data_2026/ToR_data/tor_data_marzo/`.

**Prueba de humo** (desde la raíz del repo):

```python
gnn = GNN(); gnn.load_dataset(str(GRAFO_DIR / "as_features.csv"), str(GRAFO_DIR / "gnn_edges_caida.csv"))
# → 87 796 nodos, 1 336 852 aristas, 85 features
gnn.split_edges_classification(0.7, 0.15, seed=42)
# → train 691 574 / val 148 194 / test 148 196
```

### 4.4 Entrenamiento y evaluación

Los training loops viven en los notebooks; ejecútelos desde la raíz del repositorio (los `import modules.…` son relativos a ella):

```bash
source env310/bin/activate
jupyter lab   # o jupyter notebook
```

**End-to-end** (`AS_relationship_inference_end_to_end.ipynb`), orden de celdas:

1. Celda 2: imports (`%autoreload` recarga `modules/` al editar).
2. Celda 4: funciones de ploteo.
3. Celda 6: **editar `GRAFO_DIR`** (y `CAIDA_DIR` si cambia). Esta celda:
   * define `nodes_file`, `edges_file`, `RUN_TAG` (`experimento<N>_<AAAAMMDDTHHMM>`) y `RESULTS_DIR = GRAFO_DIR/results/end_to_end`;
   * define `SEED = 42` (cámbiela para probar otra semilla) y llama a `set_seed(SEED)`;
   * etiqueta con CAIDA si `gnn_edges_caida.csv` no existe.
4. Celda 7: `GLOBAL_CONFIG` (hiperparámetros, ver §3.3.1). Celda 8: `focal_loss`.
5. Elegir un caso y ejecutar sus celdas en orden:
   * Caso 0 (celdas 11–17): carga + split → `models = {...}` → entrenamiento MLP. Para el decoder Bilinear descomente la celda 15.
   * Caso 1 (19–26), Caso 2 (28–31), Caso 3 (33–36), Caso 4 (38–39), comparación completa (43).
   * Cada celda de entrenamiento itera sobre el dict `models` (`'GraphSAGE': GraphSAGE3L`, …); para cambiar arquitectura o profundidad edite ese dict (las variantes 2L están comentadas).
6. Salidas:
   * en pantalla: log cada 10 épocas, tiempo total, `classification_report`, matrices de confusión, baseline y curvas;
   * en disco: cada modelo guarda métricas, pesos e historial en `RESULTS_DIR/<caso>/<modelo>_<decoder>_seed<SEED>_<config_id>/` y una fila en `RESULTS_DIR/resumen.csv` (§3.3.0). El Caso 1 Bilinear (celda 24) además guarda curvas PNG y predicciones por arista en `RESULTS_DIR/plots/` y `RESULTS_DIR/predictions/`.

**Por partes** (`AS_relationship_inference_por_partes.ipynb`):

1. Celdas 1–7: imports; rutas en la celda 3 (**editar `GRAFO_DIR`**; de ella salen `nodes_file`, `edges_file` y `data_path = GRAFO_DIR/`; además `CAIDA_DIR`, `RIBS_DIR`, `ribs_file`); helpers; `GLOBAL_CONFIG`.
2. Etapa A: ejecutar el caso deseado (Caso 0 → celdas 9–14; Caso 3 → 29–33; Caso 4 → 35–41; DeepWalk → 44–49; BGP2Vec → 51–60). Cada uno guarda embeddings/modelos en `GRAFO_DIR/results/EnfoquePorPartes/Caso{N}/`, en la carpeta del grafo. Los embeddings antiguos del grafo de marzo están en `/media/vale/KINGSTON/TESIS/data_2026/results/` y no son compatibles con otro grafo (cambian los nodos y la dimensión).
3. Etapa B (celdas 62–94): en la celda 67 descomente la ruta del embedding a usar y fije `attr`/`CASO`; después ejecute carga del ToR (73), mapeo y features (75–76), split y pesos (78), modelo (80), entrenamiento con batches (90) y evaluación (92–94).

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

Las aristas con `edge_label == -1` (174 444 enlaces, ≈26 %, en 2026-09-16) son exactamente las que el modelo puede inferir sin etiqueta de referencia. Para pares de AS **no** presentes en el grafo, use el enfoque por partes (`build_edge_features` + `EdgeClassifierANN`) o `_fill_labels_from_caida_stream_fast`-style para agregar nodos con `feat=0`.

### 4.6 Artefactos generados

| Ruta | Contenido |
|---|---|
| `GRAFO_DIR/as_features.csv`, `gnn_edges.csv` | Grafo (generado por `as-level-internet-graph`) |
| `GRAFO_DIR/gnn_edges_caida.csv`, `etiquetas_caida.json` | Aristas etiquetadas y resumen (`GNN.label_edges_from_caida`) |
| `CAIDA_DIR/AAAAMM01.as-rel.txt.bz2` | CAIDA AS-Relationships (descarga manual) |
| `/media/vale/KINGSTON/TESIS/data_2026/` | Grafo antiguo de marzo (`*_enriched_tesis.csv`), `rib_*.txt`, dataset ToR (`*.npy`) |
| `GRAFO_DIR/results/end_to_end/<caso>/<modelo>_<decoder>_seed<SEED>_<config_id>/` | `metrics.json`, `model.pth`, `history.csv` de cada modelo end-to-end (`save_run`) |
| `GRAFO_DIR/results/end_to_end/resumen.csv` | Una fila por corrida: fecha, caso, modelo, decoder, semilla, accuracy, macro-F1, tiempo_s, tiempo_total_s, épocas, carpeta |
| `GRAFO_DIR/results/end_to_end/plots/`, `predictions/` | Curvas y predicciones del Caso 1 Bilinear (celda 24) |
| `GRAFO_DIR/results/EnfoquePorPartes/ToR/` | Dataset ToR del grafo (`build_tor_dataset`) + `tor_resumen.json` |
| `GRAFO_DIR/results/EnfoquePorPartes/clasificador_ann/`, `resumen.csv` | Métricas del clasificador ANN por embedding (`save_run`, celda 92) |
| `GRAFO_DIR/results/ejecucion/`, `<experimento>/resultados_gnn/` | Ejecución por lotes (§4.7) |
| `GRAFO_DIR/results/EnfoquePorPartes/Caso{0..5}/` | `embeddings_*.pt`, `model_*.pth`, `bgp2vec.word2vec`, `roc_*.png`, `confusion_matrix.png`, `ann_con_batches_*.pth` |
| `notebooks/results/*.pdf` | Figuras de análisis |
| `notebooks/resultados_experimentos.ipynb` | Tablas de accuracy, F1, métricas por clase (P2P/C2P/P2C) y tiempos de todos los modelos de ambos enfoques; lee los `metrics.json` de un experimento (parámetros `EXPERIMENTO`, `GRAFOS`, `RAPIDO`, `SEED`) y puede exportar las tablas a `<experimento>/resultados_gnn/tablas/` |

### 4.7 Correr un experimento completo (`scripts/correr_experimento.py`)

El script ejecuta los dos notebooks sobre **todos los grafos de un experimento** de `as-level-internet-graph`, sin abrir Jupyter:

```bash
cd gnn-as-relationships
EXP=/media/vale/KINGSTON/as-level-internet-graph/experimentos/experimento7

env310/bin/python scripts/correr_experimento.py $EXP                    # todo, en todos los grafos
env310/bin/python scripts/correr_experimento.py $EXP --rapido           # prueba de humo (minutos por grafo)
env310/bin/python scripts/correr_experimento.py $EXP --grafos 2025 2026 --notebooks end_to_end --casos-e2e caso0 caso1
env310/bin/python scripts/correr_experimento.py $EXP --reanudar         # retoma: salta los casos terminados
env310/bin/python scripts/correr_experimento.py $EXP --solo-preparar    # solo CAIDA + etiquetado + dataset ToR
env310/bin/python scripts/correr_experimento.py $EXP --config '{"epochs": 100, "hidden_dim": 64}'
```

**Qué hace con cada grafo** (`grafos/<año>/<mes>/<AAAAMMDDTHHMM>/`):

1. **Preparación.**
   * Omite el grafo si no tiene `as_features.csv` y `gnn_edges.csv`. En experimento7, solo 2018–2026 están enriquecidos.
   * Descarga el CAIDA del mes a `--caida-dir` (por defecto `/media/vale/KINGSTON/TESIS/data_2026`) si falta. Con `--sin-descarga`, en cambio, omite el grafo.
   * Etiqueta las aristas (`gnn_edges_caida.csv`).
   * Crea el **dataset ToR del grafo** con `GNN.build_tor_dataset`, en `results/EnfoquePorPartes/ToR/`:
     * usa los enlaces etiquetados del propio grafo, así que las etiquetas son del mismo mes;
     * hace un split 80/20 por enlace no dirigido, estratificado y con `SEED`, de modo que no hay fuga entre direcciones;
     * cada enlace aporta las dos direcciones.
2. **End-to-end.** Ejecuta `caso0`, `caso1`, `caso2`, `caso3`, `caso4` y `comparacion`.
3. **Por partes.**
   * Ejecuta `caso0`, `caso1`, `caso2`, `caso3`, `caso4`, `deepwalk` y `bgp2vec`.
   * Después ejecuta `clasificador` **una vez por cada embedding** generado en ese grafo: Casos 0, 1, 3 y 4, DeepWalk y BGP2Vec. Excluye el Caso 2, que es del grafo externo 2022 y cuyos nodos no corresponden.
   * El Caso 2 y BGP2Vec no dependen del grafo, pero se ejecutan igual si están seleccionados.

**Cómo ejecuta.**
* Cada caso corre en un **kernel nuevo**, solo con las celdas cuyas etiquetas de metadatos (`metadata.tags`) son `config` o el nombre del caso. Para cambiar qué celdas forman un caso, edite esas etiquetas en Jupyter (*View → Cell Toolbar → Tags*).
* Los parámetros llegan a la celda de configuración por variables de entorno: `GRAFO_DIR`, `CAIDA_DIR`, `SEED`, `RESULTS_ROOT`, `GNN_CONFIG`, `TOR_PATH` y `EMBEDDINGS_PATH`. Sin ellas, los notebooks usan sus valores por defecto, así que siguen funcionando igual en Jupyter.
* `GNN_CONFIG` (`--config`) se mezcla con `GLOBAL_CONFIG`. Además acepta claves para épocas que estaban fijas: `epochs_muestreo` (Casos 2–3 end-to-end, 15), `deepwalk_epochs` (50), `ann_epochs` (200), `bgp2vec_max_routes` (5 000 000) y `bgp2vec_count_routes` (True).
* Un caso que falla, por ejemplo por un error o porque el kernel muere sin memoria, queda registrado y el script **sigue con el siguiente**.
* `--rapido` usa hiperparámetros mínimos y escribe todo en `GRAFO_DIR/prueba_rapida/` y `<experimento>/resultados_gnn_prueba/`, sin tocar los resultados reales.

**Salidas.**

| Ruta | Contenido |
|---|---|
| `GRAFO_DIR/results/end_to_end/…`, `GRAFO_DIR/results/EnfoquePorPartes/…` | Lo que guarda cada notebook (§4.6) |
| `GRAFO_DIR/results/EnfoquePorPartes/clasificador_ann/<embedding>_ann_seed<SEED>_<config_id>/` | Métricas, pesos e historial del clasificador ANN de cada embedding (`save_run`) |
| `GRAFO_DIR/results/ejecucion/<notebook>_<caso>[_<embedding>].ipynb` | Notebook ejecutado de cada caso, con sus salidas y el error si falló |
| `GRAFO_DIR/results/ejecucion/*.ok` | Marcador de caso terminado (lo usa `--reanudar`) |
| `<experimento>/resultados_gnn/ejecucion_<fecha>.csv` | Estado (`ok` / `error` / `omitido` / `ya hecho`), duración y error de cada caso |
| `<experimento>/resultados_gnn/metricas.csv` | Todas las métricas de test de todos los grafos (una fila por `metrics.json`, con `tiempo_s`, `tiempo_total_s` y `epocas`) para comparar entre años |
| `<embedding>.tiempo.json` (junto a cada `.pt` / `.word2vec`) | Duración del entrenamiento del modelo que generó el embedding (`GNN.save_tiempo_embedding`) |

**Advertencias.**
* Con la configuración por defecto de los notebooks (500 épocas, `hidden=256`), un grafo completo tarda **muchas horas** en CPU. Los GAT full-batch pueden quedarse sin memoria en una máquina de 15 GB; en ese caso el caso se registra como `error` y la corrida continúa.
* Use `--grafos` y `--casos-*` para acotar, y `--reanudar` para retomar.
* El Caso 2 por partes necesita `InternetGNNData2022/caida/caida_pyg.pt`, que hoy no existe (hay que convertir `caida.bin` desde un entorno con DGL; §3.3.2). Mientras tanto, ese caso termina en `error`.

---

## 5. Propuestas de Mejora y Deuda Técnica

### 5.1 Análisis crítico del código actual

**Bugs y defectos confirmados por lectura del código**

| # | Ubicación | Problema | Efecto |
|---|---|---|---|
| 1 | `modules/gnn.py:17-35` | La clase `GNN` está definida **dos veces**; la segunda sombrea a la primera (que inicializaba `train_eids/val_eids/test_eids`). Imports duplicados (`torch`, `random`, `defaultdict`, `Counter`). | Los atributos `*_eids` solo existen tras llamar a `split_edges_classification`; código confuso. |
| 2 | `modules/gnn.py:560-563`, `616`, `759` | `split_edges_link_prediction`, `split_edges_classification_leaky` y `_v0` buscan `edata["Relationship"]`, pero `load_dataset` escribe `edata["label"]`. | En link prediction las etiquetas nunca se copian a `train_pos_g/test_pos_g`; los otros dos métodos lanzan `KeyError`. |
| 3 | `modules/gnn.py:547` | En link prediction se eliminan las aristas de test de `train_g`, pero como el grafo es bidireccional **la arista inversa de cada positivo de test sigue en `train_g`** (y probablemente en `train_pos_g`). | Fuga de información: los AUC de los Casos 0–2 del enfoque por partes están sobreestimados. |
| 4 | `modules/gnn_models.py:200-206, 233-239, 267-273` | `GCN2L`, `GraphSAGE2L` y `GAT2L` definen `forward()` usando `self.regressor`, que **no existe** en las clases 2L. | `AttributeError` al usar modelos 2L para regresión nodal. |
| 5 | `modules/gnn_models.py:248` | `GAT2L.Bilinear = BilinearPredictor(out_feats, …)` mientras el embedding tiene `out_feats × num_heads` dims. | `decodeBilinear` falla por tamaño en GAT2L (afecta la celda de comparación). |
| 6 | `modules/gnn_models.py` (`GATSample2L`) | *(Corregido en la migración)* La versión DGL no aplanaba las cabezas entre `conv1` y `conv2` y el `MLPPredictor` recibía una dimensión inconsistente. | En PyG `conv2` y el MLP usan `hidden×heads` / `out×heads`. |
| 7 | `modules/graph.py:371-380` | En `label_edges_caida`, tras asignar `label = 2` para `(src→dst)`, la comprobación `label == -1` para `(dst→src)` ya es falsa. | **Ambas direcciones quedan P2C**; el commit "Corregir etiquetas P2C/C2P invertidas" no lo resolvió. (Pipeline legado.) |
| 8 | `modules/graph.py:430-471` | `only_degree_features_nodes` contiene código muerto tras `f.close()` que escribe en el archivo cerrado. | `ValueError: I/O operation on closed file`. |
| 9 | `modules/graph.py:190` | `create_graph_from_caida` ignora `filename_out` y concatena `self.data_path + "edges.csv"` sin `os.path.join`. | Ruta incorrecta si `data_path` no termina en `/`. |
| 10 | `scripts/4_add_atributes.py` (`__main__`) | *(Script eliminado en `4fcb754`)* *(Corregido)* `caida_relationships_file` (2026) se sobrescribía con la ruta CAIDA de 2024 antes de usarse y `peeringdb_file` se definía sin usarse. | Hasta la corrección, el etiquetado del RIB 2026 usaba relaciones CAIDA de marzo 2024; los CSV de `data/` pueden haberse generado así. |
| 11 | `scripts/2_create_edges_from_ribs.py:92` *(eliminado en `4fcb754`)* | `sorted(dict_nodes.keys())` ordena **strings**, no enteros (`"10" < "9"`). | `node_id` no sigue orden numérico de ASN (inofensivo, pero difiere de `3_filter_by_country.py`, que sí usa `key=int`). |
| 12 | `scripts/1_ribs_extraction.py` *(eliminado en `4fcb754`)* | `argparse` importado y no usado; `count_records` se incrementa antes del chequeo `rec is None`; `strftime('%s')` no portable. | Configuración no parametrizable; conteo desviado en 1. |
| 13 | `modules/bgp2vec.py:86-90` | `asn2idx`/`idx2asn` usan la API de gensim 3 (`wv.vocab`, `wv.index2word`). | Rotos con gensim 4.x (instalado 4.4.0). |
| 14 | `modules/bgp2vec/bgp2vec.py:50` | `total_examples=len(self.routes)` falla con iteradores (`RIBCorpus`, `islice`). | Copia obsoleta; debe eliminarse. |
| 15 | `.gitignore` | Ignora `*.md`, `*.csv`, `*.json`, `*.txt`, `*.png`; `README.md` y `data/*.csv` fueron añadidos a la fuerza. | Cualquier documentación o dato nuevo queda invisible para git (este archivo requiere la excepción `!DOCUMENTATION.md`). |
| 16 | `src/__pycache__/utils.py`, `**/__pycache__/*.pyc`, `src/graph_builder.py` | Fuente dentro de `__pycache__`, bytecode versionado, módulo vacío. | Ruido en el repo; riesgo de importar la versión equivocada de `utils`. |
| 17 | `README.md` | *(Corregido)* Citaba `3_export_to_dgl_graph.py` (no existe) y omitía `4_add_atributes.py`. | Onboarding incorrecto hasta la corrección. |
| 18 | `notebooks/*.ipynb` (análisis) | `analisis_grafo_internet`, `attribute_inspection`, `tsne_embeddings`, `AS_embeddings_visualization`, `peeringdb_coverage_bias`, `validar_normalizacion_atributos`, `analisis_ribs_recolectadas` y `analysis_as_topology` siguen apuntando a la carpeta borrada `as-level-internet-graph/grafos/`. Salvo `analysis_as_topology` (lee `as_edges.csv`, así que solo hay que cambiarle la ruta), además leen el formato antiguo (`nodes_*/edges_*_enriched_tesis.csv`, `rib_*.txt`). | No corren hasta migrarlos a `experimentos/experimento<N>/…` y al formato nuevo. Solo se migraron los dos notebooks de entrenamiento. |
| 19 | Notebook end-to-end, celdas de entrenamiento | *(Corregido)* El "mejor modelo" se guardaba con `best_w = model.state_dict()` (o `{k: v.cpu() …}` en CPU), que no copia: apunta a los pesos vivos. | Hasta la corrección, test se evaluaba con los pesos de la **última** época, no con los de la mejor en validación. Las cifras guardadas en los notebooks están afectadas. Ahora se copia con `.detach().clone()`. |
| 20 | `AS_relationship_inference_por_partes.ipynb`, celda 93 | *(Corregido)* `save_dir` concatenaba `results/EnfoquePorPartes/{CASO}/` tres veces. | `savefig` fallaba por una carpeta inexistente. Ahora se guarda `CasoN/confusion_matrix_<embedding>.png`. |
| 21 | `AS_relationship_inference_end_to_end.ipynb`, celda 24 (Caso 1 Bilinear) | *(Corregido)* El bloque de diagnóstico del final, fuera del bucle, usaba `hist`, `model_f1_macro` y `decodeMLP`, que vienen del Caso 0. | Solo funcionaba si el Caso 0 se había corrido antes en la misma sesión y fallaba con `NameError` al ejecutar el caso aislado (`scripts/correr_experimento.py`). Ahora usa `history`, `f1_macro` y `decodeBilinear` de la propia celda. |

**Problemas metodológicos**

* **Caso 4** (features aleatorias) hace early stopping y selección de pesos con el **test set** (no hay `val_mask` en esa celda): las métricas reportadas son optimistas.
* Los casos con muestreo (2 y 3) entrenan **15 épocas fijas** frente a 500 con paciencia 120 en full-batch; la comparación entre paradigmas no es equitativa. Con `batch_size=1024` y `num_parts=1000`, ClusterGCN procesa **todos los clusters en un solo batch** por época (equivale casi a full-batch con aristas inter-cluster eliminadas).
* `BilinearPredictor.W` se inicializa con `torch.randn` sin escalado (`1/√F`), lo que produce logits de gran magnitud al inicio (el `grad_clip=2.0` lo mitiga).
* La celda de "Comparación completa" (12 combinaciones) quedó interrumpida; no hay tabla final de resultados.
* El vocabulario de BGP2Vec proviene de un RIB de 12 h del 1 de marzo de 2026 (`RIBS_DIR`), distinto del grafo en uso; `as-level-internet-graph` no genera RIBs en texto. Además, `epochs=1`.
* **Cobertura CAIDA parcial.** Con el grafo 2026-09-16 y CAIDA `20260901`, el 26 % de los enlaces (174 444) queda sin etiqueta (`-1`) y fuera de los splits de clasificación. Además, el grafo es del día 16 y el snapshot CAIDA del día 1. Para que coincidan, construya el grafo el día 1 del mes (§4.3).
* Las cifras de la tabla de §3.3.1 (72 features, tiempos, métricas) corresponden a los outputs guardados con el grafo antiguo de marzo. Con los grafos nuevos hay que volver a correr los casos.
* Entrenamiento transductivo: aristas de val/test participan en el message passing (ver §3.4). Es defendible, pero debe declararse y, idealmente, contrastarse con un escenario inductivo (`remove_edges` de test antes de `encode`).

**Calidad general**

* Lógica de entrenamiento duplicada ~8 veces entre celdas (solo cambian decoder/pérdida); `_train_one` de la celda 43 ya es el esqueleto correcto para factorizarla.
* Seis clases de encoder casi idénticas; ausencia de tipado en `modules/`, docstrings parciales y mezcla español/inglés.
* Configuración por edición de código; salidas mayormente por `print`. La semilla ya está unificada (`SEED` + `set_seed`, §3.3.0), pero en GPU la reproducibilidad no es exacta.
* Cero tests automatizados.

### 5.2 Oportunidades de optimización

**Rendimiento de entrenamiento**

* Mover el training loop a `modules/train.py` con una función `train_edge_classifier(model, g, cfg, decoder="mlp")` y un CLI (`python -m modules.train --case 0 --model sage --decoder mlp --device cuda`). Elimina las copias y hace reproducibles los experimentos.
* Ejecutar en GPU y usar `torch.autocast` (AMP) para el decoder y las capas lineales; considerar `torch.compile` en PyTorch ≥ 2.3.
* Evaluar en validación cada *k* épocas en lugar de cada época (hoy cada época implica dos `encode` completos).
* `split_edges_classification` construye `pair2eids` con bucles Python sobre 660 k aristas; vectorizar con `torch.minimum/maximum` + `torch.unique(return_inverse=True)`.
* `_fill_labels_from_caida_stream_fast` construye un `dict` Python de 800 k entradas y actualiza `edata['label'][eid]` elemento a elemento; usar `g.edge_ids(u, v, return_uv=True)` vectorizado y una sola asignación con máscara.

**Grafos a gran escala y memoria**

* Para RIBs completos (varios colectores, IPv6) el grafo puede superar 100 k nodos y varios millones de aristas: guardar el `Data` con `torch.save` (o un `InMemoryDataset` de PyG) para evitar re-parsear CSV (hoy `load_dataset` reconstruye el grafo en cada notebook).
* En `1_ribs_extraction.py`, reemplazar el `set` de AS_PATHs completos por hashing (`hash(path)`) o por deduplicación posterior con `sort -u`, y escribir con buffer.
* `NeighborSampler`: usar `num_workers > 0`, `use_uva=True` con GPU, y `fanouts` asimétricos (`[10, 25]`); igualar el presupuesto de épocas al full-batch antes de comparar.
* ClusterGCN: reducir `batch_size` de clusters (p. ej. 20–50 de 1000) para obtener verdadero mini-batching y ruido de gradiente.
* Betweenness: paralelizar con `networkx` sobre subconjuntos o usar `igraph`/`graph-tool`, que son 10–100× más rápidos.

### 5.3 Buenas prácticas recomendadas

1. **Corrección inmediata de los bugs 1–5, 7, 10 y 15** (tabla anterior). Para el 15: añadir `!README.md` y `!DOCUMENTATION.md` (o dejar de ignorar `*.md`).
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
* **Ablaciones sistemáticas**: PeeringDB vs centralidades vs ambos, `edge_feat` on/off, profundidad 1–4, `num_heads`, `focal_gamma`, `balance_mode`, con la celda 43 completada y automatizada.
