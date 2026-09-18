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

1. Se construye la topología a partir de tablas de enrutamiento BGP (RIBs de RouteViews / RIPE RIS, vía BGPStream).
2. Cada AS (nodo) se enriquece con atributos estructurales (centralidades) y de negocio (PeeringDB).
3. Las aristas se etiquetan con el dataset de referencia **CAIDA AS-Relationships** (`as-rel.txt.bz2`).
4. Una GNN aprende *embeddings* de los AS y un *decoder* de aristas predice la clase P2P / C2P / P2C.

Se exploran dos familias de experimentos, cada una en su notebook:

* **End-to-end** (`AS_relationship_inference_end_to_end.ipynb`): el encoder GNN y el decoder de aristas se entrenan conjuntamente con la pérdida de clasificación.
* **Por partes** (`AS_relationship_inference_por_partes.ipynb`): primero se aprenden embeddings con una tarea auxiliar (link prediction, regresión de grado/PageRank, DeepWalk, BGP2Vec) y luego un MLP independiente clasifica pares de AS.

### 1.2 Enfoque basado en GNN y representación de las relaciones

**Grafo.** El grafo es un `torch_geometric.data.Data` dirigido y homogéneo:

* **Nodos** = AS, indexados por `node_id` (entero contiguo 0..N-1) con un mapeo `asn ↔ node_id` guardado en el CSV de nodos.
* **Aristas** = adyacencias observadas en AS_PATHs. Cada enlace no dirigido se materializa como **dos aristas dirigidas** `(u→v)` y `(v→u)`. Esto es esencial porque la etiqueta es *direccional*: si `(u→v)` es C2P (`1`), entonces `(v→u)` es P2C (`2`); P2P (`0`) es simétrica. La regla de inversión `{0:0, 1:2, 2:1}` está implementada en `GNN._reverse_relationship_value` (`modules/gnn.py:37`) y en `add_atributes_edges` (`scripts/4_add_atributes.py:355-413`).
* `data.x`: matriz de atributos de nodo (72 columnas en el dataset actual, ver §3.2).
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
| NetworkX | (sin fijar) | Centralidades en `scripts/4_add_atributes.py`; pipeline legado `modules/graph.py` |
| pandas / numpy | 2.3.3 / 2.2.6 | I/O CSV, normalización |
| scikit-learn | 1.7.2 | Métricas (F1, matriz de confusión, ROC-AUC), `train_test_split`, `compute_class_weight` |
| gensim | 4.4.0 | Word2Vec para BGP2Vec (`modules/bgp2vec.py`) |
| pybgpstream | 2.0.4 (`_pybgpstream`) | Descarga de RIBs (`scripts/1_ribs_extraction.py`) |
| requests, urllib | 2.25.1 | API PeeringDB, descargas LACNIC/CAIDA |
| PyYAML | 5.4.1 | Solo `modules/graph.py` (legado) |
| tqdm, matplotlib, seaborn, umap-learn, scipy | — | Progreso y visualización en notebooks |

> **Ya no se usa DGL.** Toda la parte de grafos está construida sobre PyG. No existe `requirements.txt` ni `environment.yml`; en §4.1 se propone uno.

---

## 2. Estructura del Repositorio

```
gnn-as-relationships/
├── README.md                                   # Descripción, ubicación de datos, instalación y pipeline
├── DOCUMENTATION.md                            # este documento
├── .gitignore                                  # ignora *.csv, *.md, *.json, *.txt, data/*, results/* (ver §5)
│
├── AS_relationship_inference_end_to_end.ipynb  # ★ Entrenamiento end-to-end (encoder GNN + decoder de aristas)
├── AS_relationship_inference_por_partes.ipynb  # ★ Enfoque en dos etapas: embeddings + clasificador ANN
│
├── modules/                                    # Código reutilizable importado por los notebooks
│   ├── gnn.py            # ★ Clase GNN: carga CSV→Data (PyG), etiquetado CAIDA, splits, features aleatorias, poda
│   ├── gnn_models.py     # ★ Encoders GCN/GraphSAGE/GAT (2L, 3L, *Sample2L) y decoders MLP/Bilinear/DotProduct
│   ├── graph.py          # Pipeline LEGADO: clase Graph (NetworkX) → dataset multi-grafo mensual (meta.yaml, graphs.csv)
│   ├── bgp2vec.py        # RIBCorpus (streaming de AS_PATHs) + BGP2VEC (Word2Vec skip-gram sobre rutas)
│   └── bgp2vec/          # Versión anterior de BGP2Vec (copia casi idéntica de bgp2vec.py)
│       ├── bgp2vec.py    #   - usa len(self.routes): incompatible con iteradores
│       ├── oix_utils.py  #   - parsers del formato OIX (Oregon RouteViews) del paper BGP2Vec original
│       ├── bgp2vec.word2vec  # modelo Word2Vec entrenado (binario)
│       └── ToR_bgp2vec.ipynb # notebook Keras/TensorFlow del baseline BGP2Vec + NN
│
├── scripts/                                    # Pipeline de datos (numerados en orden de ejecución)
│   ├── 1_ribs_extraction.py        # BGPStream → rib_<inicio>_to_<fin>.txt (un AS_PATH por línea, separador |)
│   ├── 2_create_edges_from_ribs.py # AS_PATHs → nodes_*.csv / edges_*.csv (+ versión _pruned sin hojas)
│   ├── 3_filter_by_country.py      # (opcional) subgrafo por país: LACNIC + IXPs/facilities PeeringDB + vecinos 1-hop
│   ├── 4_add_atributes.py          # Centralidades + atributos PeeringDB (nodos) y etiquetas CAIDA (aristas) → *_enriched_tesis.csv
│   └── create_tor_dataset.py       # CAIDA as-rel → dataset ToR (x_training/x_test/y_*.npy) para el enfoque por partes
│
├── src/                                        # Utilidades de datos externos
│   ├── lacnic.py            # Descarga/parseo de delegated-lacnic-extended-latest → ASNs por país
│   ├── peeringdb.py         # Cliente paginado de la API PeeringDB (ix, netixlan, fac, netfac, net) + descarga de dump JSON
│   ├── topology_filter.py   # load_nodes_csv / load_edges_csv / filter_topology
│   ├── utils.py             # month_number_to_name (1..12 → "enero".."diciembre")
│   ├── graph_builder.py     # VACÍO (0 bytes)
│   └── __pycache__/utils.py # Archivo fuente extraviado dentro de __pycache__ (versión antigua de utils con imports de sklearn/torch)
│
├── data/                                       # Dataset actual (trackeado a la fuerza pese a .gitignore; ~42 MB)
│   ├── nodes_rib_20260301_0000_to_20260302_0000_enriched_tesis.csv   # 79 504 nodos × 74 columnas
│   └── edges_rib_20260301_0000_to_20260302_0000_enriched_tesis.csv   # 734 381 aristas dirigidas × 6 columnas
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
| `modules/gnn.py`, `modules/gnn_models.py` | **Activo** | Núcleo del proyecto |
| `scripts/1..4_*.py`, `src/*.py` | **Activo** | Pipeline de datos vigente (formato `nodes_*/edges_*` plano) |
| `scripts/create_tor_dataset.py`, `modules/bgp2vec.py` | Activo (enfoque por partes) | |
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

```
 RouteViews / RIPE RIS (BGPStream)
          │  scripts/1_ribs_extraction.py
          ▼
 rib_<YYYYMMDD_HHMM>_to_<YYYYMMDD_HHMM>.txt        (AS1|AS2|...|ASn, un AS_PATH único por línea, solo IPv4)
          │  scripts/2_create_edges_from_ribs.py
          ▼
 nodes_rib_*.csv (node_id,asn,weight)   edges_rib_*.csv (src_id,dst_id,src_asn,dst_asn,weight)
 nodes_rib_*_pruned.csv / edges_rib_*_pruned.csv    (sin hojas, 3 iteraciones)
          │  scripts/3_filter_by_country.py   (opcional, p.ej. nodes_cl_rib_*.csv)
          │  scripts/4_add_atributes.py
          │      ├─ NetworkX: degree / betweenness(k=100) / eigenvector / PageRank
          │      ├─ PeeringDB dump JSON (publicdata.caida.org) → numéricos + one-hot
          │      └─ CAIDA as-rel.txt.bz2 → relationship {0,1,2,-1} + aristas inversas
          ▼
 nodes_rib_*_enriched_tesis.csv (node_id,asn,+72 feats)   edges_rib_*_enriched_tesis.csv (…,weight,relationship)
          │  modules/gnn.py :: GNN.load_dataset()
          ▼
 torch_geometric.data.Data  x (N×72) · edge_index (2×E) · edge_label (E) · edge_attr (E×1)
          │  GNN.split_edges_classification()  /  split_edges_link_prediction()  /  split_graph_nodes()
          ▼
 data.train_mask | val_mask | test_mask  (+ gnn.train_eids / val_eids / test_eids)
          │  notebooks (training loop)
          ▼
 métricas · matrices de confusión · embeddings (.pt) · pesos (.pth)
```

#### 3.2.1 Extracción de RIBs (`scripts/1_ribs_extraction.py`)

* `BGPStream()` con `add_interval_filter(base_ts, end_ts)` y `add_filter('record-type','ribs')`; **no** se filtra por colector ni proyecto, así que se combinan todos los colectores disponibles de RouteViews y RIS.
* Se descartan AS_PATHs con AS-sets/confederaciones (`'{'` o `'('`) y prefijos IPv6 (`":" in prefix`).
* Deduplicación exacta de AS_PATH mediante un `set` en memoria.
* Salida: `rib_{start}_to_{end}.txt`, un path por línea, ASNs separados por `|`.
* `__main__` itera sobre una lista de duraciones (30 min, 1 h, 2 h, 4 h, 6 h, 12 h, 24 h) desde `03/01/2026`; el notebook `analisis_ribs_recolectadas.ipynb` compara esas capturas. El dataset final usa la de 24 h.

#### 3.2.2 Construcción de nodos y aristas (`scripts/2_create_edges_from_ribs.py`)

* `dict_nodes[asn]` = nº de apariciones del ASN en todos los AS_PATH (→ columna `weight` de nodo).
* `dict_edges[(as_i, as_{i+1})]` = nº de veces que el par consecutivo aparece (→ `weight` de arista). Las aristas se guardan **en el sentido del AS_PATH** (origen → destino) y se omiten self-loops (`as1 == as2`, típico de *prepending*).
* `asn_to_id` asigna `node_id` por orden numérico del ASN (como string → orden lexicográfico; ver §5).
* `prune_leaves(...)` elimina iterativamente (3 pasadas) nodos con grado no dirigido ≤ 1 y reindexa; genera los archivos `_pruned`.

#### 3.2.3 Filtro por país (`scripts/3_filter_by_country.py`, opcional)

Conjunto semilla = ASNs del país en `delegated-lacnic-extended-latest` (estado `allocated`/`assigned`) ∪ ASNs participantes en IXPs del país (`/api/ix` → `/api/netixlan`) ∪ ASNs con presencia en facilities del país (`/api/fac` → `/api/netfac` → `/api/net`). Con `include_neighbors=True` se agregan los vecinos a 1 salto. Se conservan solo aristas con ambos extremos en el conjunto y se reindexan los `node_id`. Salida `nodes_{cc}_{stem}.csv` / `edges_{cc}_{stem}.csv`.

#### 3.2.4 Enriquecimiento (`scripts/4_add_atributes.py`)

**Nodos (`add_atributes_nodes`)** — todas las columnas numéricas quedan en `[0, 1]` mediante `_log_minmax_scale = minmax(log1p(x))` y se verifican con `_check_range`:

| Grupo | Columnas | Fuente / cálculo |
|---|---|---|
| Peso | `weight` | Apariciones en AS_PATHs, `log1p + minmax` |
| Centralidad | `degree_centrality`, `betweenness_centrality`, `eigenvector_centrality`, `PageRank` | NetworkX sobre `DiGraph` con `weight`; betweenness **aproximada** con `k=min(100, N)` muestras y `seed=42`; eigenvector con `max_iter=1000` y fallback a grafo no dirigido → ceros si no converge |
| PeeringDB numéricos | `ix_count`, `fac_count`, `info_prefixes4`, `info_prefixes6` | `net` del dump `peeringdb_2_dump_{year}_{month}_01.json`, join por `asn`, faltantes → 0 |
| PeeringDB categóricos (one-hot 0/1) | `policy_general_*` (5), `policy_locations_*` (6), `policy_contracts_*` (4), `info_traffic_*` (19), `info_scope_*` (11), `info_type_*` (11), `info_ratio_*` (7) | `pd.get_dummies`; vacío/NaN → categoría `Unknown` |

Se eliminan las columnas nuevas con varianza cero. Resultado en el dataset actual: **72 features** por nodo (`nodes_*_enriched_tesis.csv`, 74 columnas incluyendo `node_id` y `asn`). No hay columna `country` en este dataset, aunque `GNN.load_dataset` la excluiría si existiera.

**Aristas (`add_atributes_edges`)**:

1. Parseo de CAIDA: `p|c|-1` → `(p→c)=2 (P2C)` y `(c→p)=1 (C2P)`; `a|b|0` → `0` en ambos sentidos. Deduplicación y merge `left` sobre `(src_asn, dst_asn)`; sin coincidencia → `-1`.
2. **Garantía de arista inversa**: para cada arista con etiqueta conocida se genera la inversa con etiqueta invertida; al deduplicar se prioriza (a) etiquetas conocidas sobre `-1` y (b) filas originales sobre generadas. Por eso el CSV tiene 734 381 filas y no un número par.
3. `weight` de arista → `log1p + minmax` (`float32`).

Distribución de `relationship` en `data/edges_*.csv`: `0`: 340 512 · `1`: 159 963 · `2`: 159 963 · `-1`: 73 943 (≈10 % sin etiqueta).

#### 3.2.5 Carga a PyG (`modules/gnn.py :: GNN`)

`load_dataset(nodes_csv, edges_csv, add_reverse_edges=True)`:

1. `_make_edges_bidirectional`: concatena el DataFrame con su copia invertida (etiqueta invertida con `_reverse_relationship_value`) y `drop_duplicates(['src_id','dst_id'], keep='first')` → la fila original prevalece. Con el dataset actual pasa de 734 381 a **807 496 aristas** (79 504 nodos).
2. `_normalize_relationship_column`: acepta `'P2P'/'C2P'/'P2C'` o `'0'/'1'/'2'/'-1'`, fuerza `int64` y lanza `ValueError` ante valores desconocidos.
3. `Data(edge_index=torch.stack([src, dst]), num_nodes=len(df_n))` — asume que `src_id/dst_id` son índices de fila de `df_n`.
4. `x` = todas las columnas excepto `node_id, asn, country` (`float32`); `edge_label` (`long`); `edge_attr` = `weight` `(E,1)`.
5. Guarda `self.asn_to_node_id` para `_fill_labels_from_caida_stream_fast`.

`load_dataset_only_cntrality_attr(...)`: igual pero `feat` = `[weight, PageRank, degree_centrality, betweenness_centrality, eigenvector_centrality]` **estandarizadas con z-score** (media 0, desviación 1; columnas constantes → std=1). Comentario en el código: sin esto la GNN colapsa por las escalas dispares.

`_fill_labels_from_caida_stream_fast(caida_file)`: alternativa in-memory al paso 4 de scripts: recorre el `.bz2/.gz/.txt` de CAIDA, etiqueta aristas existentes (`eid_map` dict `(u,v)→eid`), agrega aristas faltantes y crea nodos nuevos con `feat = 0` para ASNs ausentes.

Otras utilidades: `add_random_features(dim, std, seed, mode∈{zscore,minmax,uniform})` (control experimental), `remove_low_degree_nodes(degree, iterations)` (poda con `torch_geometric.utils.subgraph` y remapeo de `asn_to_node_id` con los ids originales conservados).

#### 3.2.6 Estrategias de split

| Método | Tarea | Cómo evita fugas | Salida |
|---|---|---|---|
| `split_edges_classification(train_size=0.7, val_size=0.15, seed, balance_mode)` | Clasificación de aristas | Agrupa eids por **par no dirigido** `(min(u,v), max(u,v))` → las dos direcciones caen en el mismo split. Estratifica por *firma* de etiquetas del par (`(0,0)`, `(1,2)`, `(2,)`…) y reparte cada estrato con las proporciones dadas. `balance_mode="strict_equal"` submuestrea cada split al mínimo por clase | `edata[*_mask]`, `self.train_eids/val_eids/test_eids` |
| `split_edges_link_prediction(train_ratio=0.8, seed=42)` | Link prediction | Baraja eids; `train_g = remove_edges(g, test_eids)`; subgrafos `train_pos_g/test_pos_g`; negativos por muestreo uniforme de pares sin arista (`g.has_edges_between`, hasta 3 rondas de sobre-muestreo) | `self.train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g` |
| `split_graph_nodes(train_size=0.8)` | Regresión nodal | `randperm` de nodos | `ndata['train_mask'|'test_mask']` |
| `split_edges_classification_leaky`, `split_edges_classification_v0` | Legado | Requieren `edata['Relationship']` (no lo produce `load_dataset`) | — |

Cifras del dataset actual (outputs guardados): 660 438 aristas etiquetadas; con `proportional` 0.7/0.15 → train 462 306 / val 99 064 / test 99 068; con `strict_equal` 0.8/0.15 → train 383 910 (127 970 por clase) / val 71 982 / test 23 997 (7 999 por clase).

### 3.3 Esquemas de entrenamiento

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

* Dataset ToR de `scripts/create_tor_dataset.py` (`x_training.npy`, `x_test.npy`, `y_*.npy`; pares de ASN como strings, split 80/20 con `train_test_split`).
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
| Normalización `log1p + minmax` en todo atributo numérico | `4_add_atributes.py` | Grados, centralidades y prefijos siguen distribuciones de cola pesada (power-law); `log1p` comprime la cola y `minmax` lleva todo a `[0,1]` para convivir con los one-hot. Se valida con `_check_range` y en `validar_normalizacion_atributos.ipynb`. |
| z-score para features de centralidad | `load_dataset_only_cntrality_attr` | Escalas muy distintas (PageRank ~1e-5 vs grado ~1) hacían colapsar el entrenamiento (comentario en el código). |
| `add_self_loop` en GCN/GAT, no en SAGE | `encode()` | `GraphConv`/`GATConv` fallan con nodos de in-degree 0 y se benefician de incluir la propia representación; `SAGEConv` ya concatena la proyección del nodo. |
| Self-loops por defecto en `GCNConv`/`GATConv` | Todos los encoders | Cubren los nodos sin vecinos de entrada de los subgrafos muestreados (equivale al `allow_zero_in_degree` de DGL). |
| Decoder Bilinear con `W ∈ R^{C×F×F}` | `BilinearPredictor` | Modela explícitamente la interacción `h_uᵀ W_c h_v` por clase; asimetría natural para C2P/P2C. Es más costoso (`C·F²` parámetros: 3·128² ≈ 49 k) que el MLP. |
| `edge_feat` (peso) en el MLP decoder | `MLPPredictor(edge_dim=1)` | La frecuencia de un par en los AS_PATH correlaciona con la relación (proveedores aparecen más); se inyecta al decoder porque los encoders DGL usados no consumen features de arista. |
| Features aleatorias (Caso 4) | `add_random_features` | Control experimental: mide cuánto aporta la estructura del grafo por sí sola frente a los atributos (GraphSAGE alcanza 85 % solo con estructura). |
| Betweenness aproximada (`k=100`) | `4_add_atributes.py` | Betweenness exacta es O(N·E) ≈ 80 k × 730 k: inviable; el muestreo con semilla fija es reproducible. |
| Early stopping por macro-F1 en validación | Casos 0–1 | Con clases desbalanceadas el accuracy es engañoso; macro-F1 penaliza el colapso a P2P. |
| Entrenamiento transductivo (aristas de val/test presentes como estructura) | Todos los casos end-to-end | Las aristas de val/test siguen en el grafo durante el message passing (solo se oculta la etiqueta). Es el escenario real (la topología es observable; la relación no) pero debe explicitarse al reportar resultados. |

---

## 4. Guía de Ejecución y Pipeline (Step-by-Step)

### 4.1 Requisitos de entorno

* **SO**: Linux (desarrollado en Ubuntu, kernel 6.8). `1_ribs_extraction.py` usa `strftime('%s')`, que no es portable a Windows.
* **Python 3.10** (los notebooks usan el kernel `env310`, Python 3.10.12).
* **CPU/GPU**: el código selecciona `cuda` si está disponible; los outputs guardados muestran ejecución en **CPU** (Caso 0 ≈ 44 min por modelo). Con GPU (CUDA 12.1) los tiempos bajan un orden de magnitud.
* **RAM**: el grafo completo (80 k nodos, 807 k aristas, 72 features) cabe holgadamente en < 4 GB; los pasos pesados son `4_add_atributes.py` (NetworkX) y la deduplicación de AS_PATHs en `1_ribs_extraction.py` (`set` en memoria: un RIB de 24 h supera los 27 M de rutas según comentarios del notebook).
* **libBGPStream** instalada en el sistema (dependencia nativa de `pybgpstream`).
* Acceso a red para: BGPStream (RouteViews/RIS), `publicdata.caida.org` (PeeringDB dumps y `as-rel`), `ftp.lacnic.net`, `peeringdb.com/api`.

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

# 5. Solo para extraer RIBs (requiere libbgpstream en el sistema: https://bgpstream.caida.org/docs/install)
pip install pybgpstream==2.0.4

# 6. Registrar el kernel que esperan los notebooks
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
pybgpstream==2.0.4   # opcional
jupyter
```

> **Importante sobre rutas.** Todos los scripts y notebooks tienen rutas absolutas hard-coded a un disco externo. Los archivos de los grafos (`rib_*.txt`, `nodes_*.csv`, `edges_*.csv` y sus versiones `_enriched_tesis`) se leen y escriben en **`/media/vale/KINGSTON/as-level-internet-graph/grafos/`** (variables `data_path` / `output_dir` / `DATA_PATH` / `BASE`). El dataset ToR y los archivos CAIDA del enfoque por partes siguen en `/media/vale/KINGSTON/TESIS/` (`TESIS_PATH`, `PATH` en `create_tor_dataset.py`). Si su disco está en otra ruta, edite esas variables en el bloque `if __name__ == '__main__':` de cada script o en la celda de configuración del notebook. **No existen flags de línea de comandos** (ver §5).
>
> Ese directorio es además la salida del proyecto hermano [`as-level-internet-graph`](https://github.com/valesteban/as-level-internet-graph), que genera `grafos/<año>/<mes>/<fecha>/as_edges.csv` con columnas `as_a,as_b,weight`. Ese formato **no** es el que consume `GNN.load_dataset` (`nodes_*.csv` / `edges_*.csv` con `node_id`/`src_id`/`dst_id`); para usar uno de esos grafos hay que convertirlo primero al formato de `scripts/2_create_edges_from_ribs.py` y luego pasar por `scripts/4_add_atributes.py`.

### 4.3 Preparación de datos

En lo que sigue `<DATA>` = `/media/vale/KINGSTON/as-level-internet-graph/grafos/` (valor actual en los scripts; edítelo si cambia).

**Paso 1 — Extraer AS_PATHs de RIBs**

```bash
# Editar en scripts/1_ribs_extraction.py:
#   start = "03/01/2026"                    # MM/DD/YYYY
#   list_durations = [86400]                 # segundos; dejar solo la duración deseada
#   data_path = "<DATA>/"
python3 scripts/1_ribs_extraction.py
# → <DATA>/rib_20260301_0000_to_20260302_0000.txt
```

Parámetros de la función `ribs_download(start_date, duration, data_path)`: fecha `MM/DD/YYYY`, duración en segundos, carpeta de salida. Emite progreso cada `LOG_EVERY_N = 1_000_000` registros.

**Paso 2 — Nodos y aristas**

```bash
# Editar en scripts/2_create_edges_from_ribs.py:
#   list_ribs_files = ["rib_20260301_0000_to_20260302_0000.txt"]
#   data_path = "<DATA>/"
python3 scripts/2_create_edges_from_ribs.py
# → nodes_rib_…csv, edges_rib_…csv, nodes_rib_…_pruned.csv, edges_rib_…_pruned.csv
```

**Paso 3 (opcional) — Subgrafo por país**

```bash
# Editar en scripts/3_filter_by_country.py: data_path, country_code="CL", list_ribs_files
# Descarga automáticamente delegated-lacnic-extended-latest si no existe (src/lacnic.py)
python3 scripts/3_filter_by_country.py
# → nodes_cl_rib_…csv, edges_cl_rib_…csv
```

Parámetros de `filter_country_topology(...)`: `country_code`, `include_ixp_participants`, `include_facility_asns`, `include_neighbors` (todos `True` por defecto en `__main__`). Las consultas a PeeringDB son paginadas (`limit=250`) y pueden tardar varios minutos.

**Paso 4 — Enriquecer nodos y etiquetar aristas**

Descargas previas (manuales):

```bash
# CAIDA AS Relationships (serial-1), mismo mes del RIB
wget -P <DATA>/ https://publicdata.caida.org/datasets/as-relationships/serial-1/20260301.as-rel.txt.bz2
# El dump de PeeringDB se descarga solo si falta:
#   https://publicdata.caida.org/datasets/peeringdb/2026/03/peeringdb_2_dump_2026_03_01.json
```

```bash
# Editar en scripts/4_add_atributes.py (__main__):
#   output_dir = "<DATA>/"
#   nodes_csv  = output_dir + "nodes_rib_20260301_0000_to_20260302_0000.csv"   # o la versión _pruned / _cl_
#   edges_csv  = output_dir + "edges_rib_20260301_0000_to_20260302_0000.csv"
#   year, month = "2026", "03"
#   caida_relationships_file = output_dir + "20260301.as-rel.txt.bz2"
python3 scripts/4_add_atributes.py
# → nodes_rib_…_enriched_tesis.csv, edges_rib_…_enriched_tesis.csv
```

Funciones públicas: `add_atributes_nodes(nodes_csv, edges_csv, year, month, output_dir)` y `add_atributes_edges(edges_csv, caida_rel_path=None, output_dir=None)` (si no encuentra el archivo CAIDA asigna `-1` a todo).

**Paso 5 (solo enfoque por partes) — Dataset ToR**

```bash
# Editar PATH, DATA, ToR_CSV y MES en scripts/create_tor_dataset.py; crear <DATA>/ToR_data/tor_data_marzo/
python3 scripts/create_tor_dataset.py
# → x_training.npy, x_test.npy, y_training.npy, y_test.npy  (TEST_SIZE = 0.2)
```

Los dos CSV `*_enriched_tesis.csv` incluidos en `data/` son el resultado de los pasos 1–4 para el RIB de 24 h del 1 de marzo de 2026 y permiten ejecutar los notebooks sin rehacer el pipeline (ajustando `data_path`).

### 4.4 Entrenamiento y evaluación

Los training loops viven en los notebooks; ejecútelos desde la raíz del repositorio (los `import modules.…` son relativos a ella):

```bash
source env310/bin/activate
jupyter lab   # o jupyter notebook
```

**End-to-end** (`AS_relationship_inference_end_to_end.ipynb`), orden de celdas:

1. Celda 2: imports (`%autoreload` recarga `modules/` al editar).
2. Celda 4: funciones de ploteo.
3. Celda 6: **editar `data_path`**, `nodes_file`, `edges_file`.
4. Celda 7: `GLOBAL_CONFIG` (hiperparámetros, ver §3.3.1). Celda 8: `focal_loss`.
5. Elegir un caso y ejecutar sus celdas en orden:
   * Caso 0 (celdas 11–17): carga + split → `models = {...}` → entrenamiento MLP. Para el decoder Bilinear descomente la celda 15.
   * Caso 1 (19–26), Caso 2 (28–31), Caso 3 (33–36), Caso 4 (38–39), comparación completa (43).
   * Cada celda de entrenamiento itera sobre el dict `models` (`'GraphSAGE': GraphSAGE3L`, …); para cambiar arquitectura o profundidad edite ese dict (las variantes 2L están comentadas).
6. Salidas en pantalla: log cada 10 épocas, tiempo total, `classification_report`, matrices de confusión, baseline y curvas. `output_root = ./results/end_to_end` se crea pero en la versión actual no se escriben archivos.

**Por partes** (`AS_relationship_inference_por_partes.ipynb`):

1. Celdas 1–7: imports, rutas (`data_path`, `nodes_file`, `edges_file`, `ribs_file`), helpers, `GLOBAL_CONFIG`.
2. Etapa A: ejecutar el caso deseado (Caso 0 → celdas 9–14; Caso 3 → 29–33; Caso 4 → 35–41; DeepWalk → 44–49; BGP2Vec → 51–60). Cada uno guarda embeddings/modelos en `<data_path>/results/EnfoquePorPartes/Caso{N}/`.
3. Etapa B (celdas 62–94): en la celda 67 descomente la ruta del embedding a usar y fije `attr`/`CASO`; después ejecute carga del ToR (73), mapeo y features (75–76), split y pesos (78), modelo (80), entrenamiento con batches (90) y evaluación (92–94).

### 4.5 Inferencia con un modelo entrenado

No hay script de inferencia; el patrón mínimo, coherente con el código de los notebooks:

```python
import torch, csv
from modules.gnn import GNN
from modules.gnn_models import GraphSAGE3L

gnn = GNN(debug=False)
gnn.load_dataset("data/nodes_rib_…_enriched_tesis.csv", "data/edges_rib_…_enriched_tesis.csv")
g = gnn.graph
in_feats = g.x.shape[1]                             # 72

model = GraphSAGE3L(in_feats, 256, 128, out_feats_mlp=3, edge_dim=1, drop=0.3)
model.load_state_dict(torch.load("ruta/al/modelo.pth", map_location="cpu"))
model.eval()

with torch.no_grad():
    h = model.encode(g.x, g.edge_index)             # embeddings (N, 128)
    logits = model.decodeMLP(g.edge_index, h, g.edge_attr)   # (E, 3)
    pred = logits.argmax(1)                         # 0=P2P, 1=C2P, 2=P2C

# Traducir a ASNs
id_to_asn = {int(r["node_id"]): int(r["asn"]) for r in csv.DictReader(open("data/nodes_rib_…_enriched_tesis.csv"))}
u, v = g.edge_index
for eid in range(5):
    print(id_to_asn[int(u[eid])], "->", id_to_asn[int(v[eid])], ["P2P","C2P","P2C"][int(pred[eid])])
```

Las aristas con `edge_label == -1` (≈74 k en el CSV, ~10 %) son exactamente las que el modelo puede inferir sin etiqueta de referencia. Para pares de AS **no** presentes en el grafo, use el enfoque por partes (`build_edge_features` + `EdgeClassifierANN`) o `_fill_labels_from_caida_stream_fast`-style para agregar nodos con `feat=0`.

### 4.6 Artefactos generados

| Ruta | Contenido |
|---|---|
| `<DATA>/rib_*.txt`, `nodes_*.csv`, `edges_*.csv` | Pipeline de datos |
| `<DATA>/peeringdb_2_dump_YYYY_MM_01.json`, `*.as-rel.txt.bz2`, `delegated-lacnic-extended-latest` | Fuentes externas descargadas |
| `<DATA>/ToR_data/tor_data_<mes>/*.npy` | Dataset ToR |
| `<DATA>/results/EnfoquePorPartes/Caso{0..5}/` | `embeddings_*.pt`, `model_*.pth`, `bgp2vec.word2vec`, `roc_*.png`, `confusion_matrix.png`, `ann_con_batches_*.pth` |
| `notebooks/results/*.pdf` | Figuras de análisis |

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
| 10 | `scripts/4_add_atributes.py` (`__main__`) | *(Corregido)* `caida_relationships_file` (2026) se sobrescribía con la ruta CAIDA de 2024 antes de usarse y `peeringdb_file` se definía sin usarse. | Hasta la corrección, el etiquetado del RIB 2026 usaba relaciones CAIDA de marzo 2024; los CSV de `data/` pueden haberse generado así. |
| 11 | `scripts/2_create_edges_from_ribs.py:92` | `sorted(dict_nodes.keys())` ordena **strings**, no enteros (`"10" < "9"`). | `node_id` no sigue orden numérico de ASN (inofensivo, pero difiere de `3_filter_by_country.py`, que sí usa `key=int`). |
| 12 | `scripts/1_ribs_extraction.py` | `argparse` importado y no usado; `count_records` se incrementa antes del chequeo `rec is None`; `strftime('%s')` no portable. | Configuración no parametrizable; conteo desviado en 1. |
| 13 | `modules/bgp2vec.py:86-90` | `asn2idx`/`idx2asn` usan la API de gensim 3 (`wv.vocab`, `wv.index2word`). | Rotos con gensim 4.x (instalado 4.4.0). |
| 14 | `modules/bgp2vec/bgp2vec.py:50` | `total_examples=len(self.routes)` falla con iteradores (`RIBCorpus`, `islice`). | Copia obsoleta; debe eliminarse. |
| 15 | `.gitignore` | Ignora `*.md`, `*.csv`, `*.json`, `*.txt`, `*.png`; `README.md` y `data/*.csv` fueron añadidos a la fuerza. | Cualquier documentación o dato nuevo queda invisible para git (este archivo requiere la excepción `!DOCUMENTATION.md`). |
| 16 | `src/__pycache__/utils.py`, `**/__pycache__/*.pyc`, `src/graph_builder.py` | Fuente dentro de `__pycache__`, bytecode versionado, módulo vacío. | Ruido en el repo; riesgo de importar la versión equivocada de `utils`. |
| 17 | `README.md` | *(Corregido)* Citaba `3_export_to_dgl_graph.py` (no existe) y omitía `4_add_atributes.py`. | Onboarding incorrecto hasta la corrección. |

**Problemas metodológicos**

* **Caso 4** (features aleatorias) hace early stopping y selección de pesos con el **test set** (no hay `val_mask` en esa celda): las métricas reportadas son optimistas.
* Los casos con muestreo (2 y 3) entrenan **15 épocas fijas** frente a 500 con paciencia 120 en full-batch; la comparación entre paradigmas no es equitativa. Con `batch_size=1024` y `num_parts=1000`, ClusterGCN procesa **todos los clusters en un solo batch** por época (equivale casi a full-batch con aristas inter-cluster eliminadas).
* `BilinearPredictor.W` se inicializa con `torch.randn` sin escalado (`1/√F`), lo que produce logits de gran magnitud al inicio (el `grad_clip=2.0` lo mitiga).
* La celda de "Comparación completa" (12 combinaciones) quedó interrumpida; no hay tabla final de resultados.
* El vocabulario de BGP2Vec proviene de un RIB de 12 h distinto del grafo (24 h), y `epochs=1`.
* Entrenamiento transductivo: aristas de val/test participan en el message passing (ver §3.4). Es defendible, pero debe declararse y, idealmente, contrastarse con un escenario inductivo (`remove_edges` de test antes de `encode`).

**Calidad general**

* Lógica de entrenamiento duplicada ~8 veces entre celdas (solo cambian decoder/pérdida); `_train_one` de la celda 43 ya es el esqueleto correcto para factorizarla.
* Seis clases de encoder casi idénticas; ausencia de tipado en `modules/`, docstrings parciales y mezcla español/inglés.
* Configuración por edición de código; salidas por `print`; sin semilla global unificada (`torch_geometric.seed_everything` solo en link prediction; `torch.backends.cudnn.deterministic` nunca se fija).
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
6. **Reproducibilidad**: función `set_seed(seed)` que fije `random`, `numpy`, `torch`, `torch.cuda`, `torch_geometric.seed_everything` y `cudnn.deterministic`; guardar los `eids` de cada split a disco.
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
