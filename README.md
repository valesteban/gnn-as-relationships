# gnn-as-relationships

Inferencia del tipo de relación entre Sistemas Autónomos de Internet (**P2P / C2P / P2C**) con Graph Neural Networks (PyTorch Geometric) sobre el grafo AS-level construido desde RIBs BGP y enriquecido con PeeringDB (repositorio [`as-level-internet-graph`](https://github.com/valesteban/as-level-internet-graph)), etiquetado aquí con CAIDA AS-Relationships.

Stack: Python 3.12 · PyTorch 2.8 · **PyTorch Geometric 2.8** · pandas · scikit-learn · gensim (BGP2Vec).

> **Versión nueva con PyTorch Geometric** (rama `migracion-pyg`). La versión original con DGL 1.1.3 está en la rama principal `main`, congelada en el tag `v1.0-dgl`.
>
> Documentación técnica completa (arquitectura, pipeline, decisiones de diseño, deuda técnica): **[DOCUMENTATION.md](DOCUMENTATION.md)**.

## Ubicación de los datos

Los grafos los construye el repositorio hermano [`as-level-internet-graph`](https://github.com/valesteban/as-level-internet-graph) (BGPStream + PeeringDB). Quedan en el disco externo, una carpeta por grafo:

```
/media/vale/KINGSTON/as-level-internet-graph/experimentos/experimento<N>/grafos/<año>/<mes>/<AAAAMMDDTHHMM>/
```

Los notebooks eligen el grafo con **una sola clave de `config.yaml`, `rutas.grafo_dir`** (variable `GRAFO_DIR`), y de ella derivan el resto:

| Archivo en `GRAFO_DIR` | Qué es |
|---|---|
| `as_features.csv` | Nodos: `node_id` (= fila), `asn` y 85 features en [0,1] (PeeringDB, grados, PageRank, k-core). Lo genera `as-level-internet-graph`. |
| `gnn_edges.csv` | Aristas: `src_id, dst_id, src_asn, dst_asn, weight, weight_raw`, una fila por enlace no dirigido y sin etiqueta. Lo genera `as-level-internet-graph`. |
| `gnn_edges_caida.csv` | `gnn_edges.csv` + `relationship` (0/1/2/-1), en el mismo orden. Lo generaba `GNN.label_edges_from_caida` (este repo; hoy eliminado, ver *Preparar un grafo*) y es la **entrada de los notebooks**. |
| `etiquetas_caida.json` | Resumen del etiquetado: conteos por clase, cobertura CAIDA. |

Otros datos externos:
* CAIDA AS-Relationships (`AAAAMM01.as-rel.txt.bz2`): `rutas.caida_dir`, hoy `/media/vale/KINGSTON/LAB/CAIDA/CAIDA_AS_Relationships/Serial_1`. `/media/vale/KINGSTON/TESIS/data_2026/` tiene además `20260301` y `20260901`.
* En `/media/vale/KINGSTON/TESIS/data_2026/`:
  * `rib_*.txt`: RIBs en texto para BGP2Vec (`rutas.rib_file_bgp2vec`);
  * el grafo antiguo del 1 de marzo de 2026 (`*_enriched_tesis.csv`, 72 features, ya etiquetado);
  * el dataset ToR antiguo (`*.npy`).
* El grafo externo 2022: `rutas.internetgraph_file`, hoy `/media/vale/KINGSTON/LAB/gnn-as-relationships/InternetGNNData2022/caida/caida.bin`.

## Configuración (`config.yaml`)

Las rutas y los parámetros con que se corren los notebooks viven en **un solo archivo, `config.yaml`** (raíz del repo). Los dos notebooks principales lo leen en sus celdas de configuración (celdas 11–13 del end-to-end, 4–6 del por partes), así que para cambiar de equipo, de disco o de grafo basta editar el YAML (hoy los datos están en `/media/vale/KINGSTON/`) y reiniciar el kernel:

```python
CONFIG_FILE = "config.yaml"          # ruta relativa: abrir Jupyter desde la raíz del repo
with open(CONFIG_FILE, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
GLOBAL_CONFIG_E2E = dict(cfg["global_config_endtoend"])   # por partes: GLOBAL_CONFIG_P2E = dict(cfg["global_config_por_partes"])
```

| Sección del YAML | Contenido |
|---|---|
| `experimento` | `nombre` de la corrida y carpeta `salida`: el end-to-end escribe en `<salida>/<nombre>/end_to_end/` y el por partes en `<salida>/<nombre>/por_etapas/` |
| `rutas` | `grafo_dir`, `caida_dir`, `internetgraph_file` (grafo 2022), `rib_file_bgp2vec` (RIB para BGP2Vec) |
| `comunes` | `seed`, `class_names`, `split` (`train_size`, `val_size`, `balance_mode`; lo usan todos los splits de ambos notebooks), `class_weights` |
| `global_config_endtoend` | Hiperparámetros del end-to-end (`GLOBAL_CONFIG_E2E`): `lr`, `hidden_dim`, `out_dim`, `epochs`, `patience`, `weight_decay`, `dropout`, `num_heads`, `scheduler`, `class_weights`, `gat_hidden_dim`/`gat_out_dim` (GAT full-batch); `num_neighbors` y `batch_size_caso2` (Caso 2), `num_parts` y `batch_size_caso3` (Caso 3), `random_features` (Caso 4) |
| `global_config_por_partes` | Hiperparámetros de la Etapa 1 (`GLOBAL_CONFIG_P2E`), `clasificador` (Etapa 2, `CONFIG_CLASIFICADOR`), `deepwalk` y `bgp2vec` |

* **Decimales.** Escríbalos con punto (`0.0005`): PyYAML lee `5e-4` como texto.
* **`split`.** `split_edges_classification` exige `train_size + val_size < 1.0` (el resto es test).
* Los notebooks ya no leen variables de entorno.
* Los notebooks de `notebooks/` todavía leen claves que el `config.yaml` actual no tiene (`grafo_antiguo_dir`, `as_graph_dir`, `experimento_dir`) y fallan al cargarlo.

## Estructura

```
modules/gnn.py             Clase GNN: CSV → torch_geometric.data.Data, splits sin fuga, dataset ToR, poda
modules/gnn_models.py      Encoders GCN / GraphSAGE / GAT (2 y 3 capas, variantes con muestreo) y decoders MLP / Bilinear / DotProduct
modules/experiment.py      set_seed (reproducibilidad) y save_run (guarda métricas, pesos e historial)
modules/bgp2vec.py         BGP2Vec (Word2Vec sobre AS_PATHs) con lectura en streaming
modules/metis_windows.py   METIS para ClusterData (Caso 3) en Windows; todavía no se usa en los notebooks
modules/graph.py           Pipeline legado multi-snapshot (no usado por los notebooks)
config.yaml                Configuración central (rutas en /media/vale/KINGSTON/, semilla, hiperparámetros)
config_nerwito.yaml        Configuración anterior del equipo Windows (rutas en F:/; formato incompatible con los notebooks actuales)
src/                       LACNIC, API PeeringDB, filtro de topología, utilidades (sin uso desde que se eliminaron los scripts)
AS_relationship_inference_end_to_end.ipynb   Entrenamiento end-to-end (encoder + decoder de aristas)
AS_relationship_inference_por_partes.ipynb   Embeddings (link prediction, regresión, DeepWalk, BGP2Vec) + clasificador de aristas
notebooks/                 Análisis del grafo, de las RIBs, de atributos y visualización de embeddings.
                           notebooks/resultados_experimentos.ipynb muestra en tablas los resultados de los dos enfoques
```

`scripts/` está vacío. El pipeline de datos anterior (`scripts/1..4_*.py`, `scripts/create_tor_dataset.py`) se eliminó en el commit `4fcb754`, y `scripts/correr_experimento.py` en `d1495d9`. Siguen en el historial de git (el pipeline, también en `main` / `v1.0-dgl`).

## Instalación

```bash
python3.12 -m venv env312 && source env312/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name env312 --display-name "Python 3.12 (env312)"
```

`requirements.txt` fija todas las versiones (torch 2.8.0 CPU, PyG 2.8 con `pyg_lib`, `torch_scatter`, `torch_sparse` y `torch_cluster`). Para GPU con CUDA 12.8, cambie `cpu` por `cu128` en las dos URLs del archivo y en la versión de torch. Se usa torch 2.8 porque es la última versión con ruedas de las cuatro extensiones.

En Ubuntu, si `python3.12 -m venv` falla por falta de `ensurepip`, cree el venv con `python3.12 -m venv --without-pip env312` e instale pip con `python3.12 -m pip --python env312/bin/python install pip`.

El entorno anterior era `env310` (Python 3.10, torch 2.3.0, CUDA 12.1). Con él se obtuvieron los resultados previos a `env312`.

### En Windows con GPU (equipo con datos en `F:\`)

Probado con Python 3.13, torch 2.11 + CUDA 12.8 (driver compatible con CUDA 13) y PyG 2.8 sin las extensiones opcionales:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install torch_geometric
pip install gensim pandas numpy networkx scikit-learn scipy matplotlib seaborn tqdm requests pyyaml umap-learn jupyter ipykernel
$env:PYTHONHASHSEED = 7   # antes de abrir Jupyter: reproducibilidad de BGP2Vec
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Rutas: las de este equipo (`F:/`) quedaron en `config_nerwito.yaml`. Ese archivo tiene el formato anterior y no sirve tal cual con los notebooks actuales (le faltan, entre otras, `comunes.split`, `scheduler` y `clasificador`): hay que pasar sus rutas a `config.yaml`. Limitaciones conocidas en este entorno:
* `pyg_lib`, `torch_scatter`, `torch_sparse` y `torch_cluster` no están instaladas; pueden fallar los casos que usan `LinkNeighborLoader` (end-to-end Caso 2), `ClusterData` (Caso 3) y `Node2Vec` (DeepWalk).
* Para `ClusterData`, `modules/metis_windows.py` (`activar_metis_windows(dll)`) reemplaza el METIS de PyG por una `metis.dll`. Todavía no está conectado al notebook.
* El Caso 2 del por partes lee `caida_pyg.pt` (el grafo 2022 convertido desde `caida.bin`, formato DGL); la conversión requiere DGL y ese archivo todavía no existe en `F:`.

## Preparar un grafo

1. **Construir el grafo** en `as-level-internet-graph` (ver su README). Conviene hacerlo el día 1 del mes, que coincide con CAIDA.
2. **Descargar CAIDA del mes del grafo** en `rutas.caida_dir`:
   ```bash
   wget -P /media/vale/KINGSTON/LAB/CAIDA/CAIDA_AS_Relationships/Serial_1/ https://publicdata.caida.org/datasets/as-relationships/serial-1/20260901.as-rel.txt.bz2
   ```
3. **Etiquetar.** `GNN.label_edges_from_caida` se eliminó de `modules/gnn.py` (cambio sin commit) y los notebooks ya no etiquetan solos. Los grafos de experimento7 ya tienen `gnn_edges_caida.csv`. Para un grafo nuevo hay que recuperar el método desde git (`git show HEAD:modules/gnn.py`). Con él:
   ```python
   GNN(debug=True).label_edges_from_caida(GRAFO_DIR / "gnn_edges.csv",
                                          GNN.caida_file_for_graph(GRAFO_DIR, CAIDA_DIR),  # CAIDA_DIR/AAAAMM01.as-rel.txt.bz2
                                          output_csv=GRAFO_DIR / "gnn_edges_caida.csv")
   ```
   Solo se etiquetan los enlaces del grafo, en el sentido de cada fila. Lo que no está en CAIDA queda en `-1`; no se agregan aristas ni nodos.

   Con `experimento7 / 20260916T0800` + CAIDA `20260901` queda etiquetado el 73.9 % de los enlaces: 493 982 de 668 426 (P2P 329 589 · C2P 28 715 · P2C 135 678).

## Entrenamiento

Abra Jupyter desde la raíz del repositorio (los notebooks importan `modules/`):

```bash
jupyter lab
```

* `AS_relationship_inference_end_to_end.ipynb`: fije rutas e hiperparámetros en `config.yaml` (`rutas.grafo_dir`, `global_config_endtoend`, …) y ejecute las celdas de configuración y funciones (2–15). Después ejecute el caso deseado: Caso 0 (atributos PeeringDB), 1 (centralidades), 2 (`LinkNeighborLoader`), 3 (ClusterGCN) o 4 (atributos aleatorios). Cada caso entrena los modelos de `modelos` (GCN2L, GraphSAGE2L, GAT2L; los 3L están comentados) o de `modelos_sampler` (Casos 2 y 3), con los decoders MLP y Bilinear.
* `AS_relationship_inference_por_partes.ipynb`: fije rutas e hiperparámetros en `config.yaml` (`rutas.grafo_dir`, `rutas.rib_file_bgp2vec`, `global_config_por_partes`, …) y ejecute las celdas de configuración y funciones (2–23). Cada caso genera los embeddings (link prediction, regresión de grado/PageRank, DeepWalk o BGP2Vec) y enseguida entrena con ellos el clasificador de aristas `EdgeClassifier`, en dos variantes (sin batches y con batches), sobre el mismo split de clasificación que el end-to-end.

### Reproducibilidad y resultados

* **Semilla.** `SEED` sale de `comunes.seed` (42). Con ella se hacen todos los splits, y cada función de entrenamiento llama a `set_seed(SEED)` antes de crear el modelo. Por eso, en CPU, una misma celda da siempre el mismo resultado, aunque se vuelva a ejecutar sola o se comenten otros modelos. En GPU puede variar levemente.
* **Resultados end-to-end.** Cada modelo guarda `metrics.json`, `model.pth` (pesos de la mejor época en validación) e `history.csv` en `<salida>/<nombre>/end_to_end/<caso>/<modelo>_<decoder>_seed<SEED>_<config_id>/`, y agrega una fila a `<salida>/<nombre>/end_to_end/resumen.csv` para comparar corridas (incluye `tiempo_s`, `epocas` y `config_id`). El `config_id` es un hash de hiperparámetros y clase del modelo: cambiar hiperparámetros o capas crea una carpeta nueva en vez de sobrescribir.
* **Resultados por partes.** El clasificador de cada embedding se guarda igual, con `save_run`, en `<salida>/<nombre>/por_etapas/<caso>/<modelo>_<decoder>-<modo>_seed<SEED>_<config_id>/` (más `por_etapas/resumen.csv`). `extra` incluye `tiempo_embedding_s` y `tiempo_total_s` (embedding + clasificador). También se guardan las curvas ROC (`roc_*.png`) y `bgp2vec.word2vec`. Los embeddings no se guardan (`save_embeddings` está comentado). Los tiempos de cada caso se muestran en una tabla (`tabla_tiempos`).

Convención de etiquetas: `0 = P2P`, `1 = C2P` (origen es cliente del destino), `2 = P2C`, `-1 = sin etiqueta`. Cada enlace se representa con dos aristas dirigidas con etiquetas inversas (`1 ↔ 2`).
