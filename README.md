# gnn-as-relationships

Inferencia del tipo de relación entre Sistemas Autónomos de Internet (**P2P / C2P / P2C**) con Graph Neural Networks (PyTorch Geometric) sobre el grafo AS-level construido desde RIBs BGP y enriquecido con PeeringDB (repositorio [`as-level-internet-graph`](https://github.com/valesteban/as-level-internet-graph)), etiquetado aquí con CAIDA AS-Relationships.

Stack: Python 3.10 · PyTorch 2.3 · **PyTorch Geometric 2.8** · pandas · scikit-learn · gensim (BGP2Vec).

> **Versión nueva con PyTorch Geometric** (rama `migracion-pyg`). La versión original con DGL 1.1.3 está en la rama principal `main`, congelada en el tag `v1.0-dgl`.
>
> Documentación técnica completa (arquitectura, pipeline, decisiones de diseño, deuda técnica): **[DOCUMENTATION.md](DOCUMENTATION.md)**.

## Ubicación de los datos

Los grafos los construye el repositorio hermano [`as-level-internet-graph`](https://github.com/valesteban/as-level-internet-graph) (BGPStream + PeeringDB). Quedan en el disco externo, una carpeta por grafo:

```
/media/vale/KINGSTON/as-level-internet-graph/experimentos/experimento<N>/grafos/<año>/<mes>/<AAAAMMDDTHHMM>/
```

Los notebooks eligen el grafo con **una sola variable, `GRAFO_DIR`**, y de ella derivan el resto:

| Archivo en `GRAFO_DIR` | Qué es |
|---|---|
| `as_features.csv` | Nodos: `node_id` (= fila), `asn` y 85 features en [0,1] (PeeringDB, grados, PageRank, k-core). Lo genera `as-level-internet-graph`. |
| `gnn_edges.csv` | Aristas: `src_id, dst_id, src_asn, dst_asn, weight, weight_raw`, una fila por enlace no dirigido y sin etiqueta. Lo genera `as-level-internet-graph`. |
| `gnn_edges_caida.csv` | `gnn_edges.csv` + `relationship` (0/1/2/-1), en el mismo orden. Lo genera `GNN.label_edges_from_caida` (este repo) y es la **entrada de los notebooks**. |
| `etiquetas_caida.json` | Resumen del etiquetado: conteos por clase, cobertura CAIDA. |
| `results/EnfoquePorPartes/Caso*/` | Embeddings y modelos del enfoque por partes. |

Otros datos externos, en `/media/vale/KINGSTON/TESIS/data_2026/`:
* `AAAAMM01.as-rel.txt.bz2`: CAIDA AS-Relationships (`CAIDA_DIR`).
* `rib_*.txt`: RIBs en texto para BGP2Vec (`RIBS_DIR`).
* El grafo antiguo del 1 de marzo de 2026 (`*_enriched_tesis.csv`, 72 features, ya etiquetado).
* El dataset ToR (`*.npy`).

El grafo externo 2022 sigue en `/media/vale/KINGSTON/TESIS/`.

## Configuración (`config.yaml`)

Las rutas y los parámetros con que se corren los notebooks viven en **un solo archivo, `config.yaml`** (raíz del repo, con comentarios). Cada notebook lo lee directamente en su celda de configuración (celda 6 del end-to-end, celda 3 del por partes, primera celda de rutas en `notebooks/`), así que para cambiar de equipo, de disco o de grafo basta editar el YAML (hoy los datos están en `F:/`) y reiniciar el kernel:

```python
import yaml
CONFIG_FILE = Path(os.environ.get("CONFIG_FILE") or next(
    p / "config.yaml" for p in [Path.cwd(), *Path.cwd().parents] if (p / "config.yaml").exists()))
with open(CONFIG_FILE, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
GLOBAL_CONFIG = dict(cfg["global_config_endtoend"])   # o cfg["global_config_por_partes"]
```

| Sección del YAML | Contenido |
|---|---|
| `experimento` | `nombre` de la corrida y carpeta `salida`: el end-to-end escribe en `<salida>/<nombre>/end_to_end/` y el por partes en `<salida>/<nombre>/results/EnfoquePorPartes/` (si `correr_experimento.py` fija `RESULTS_ROOT`, se usa `<RESULTS_ROOT>/results/...`) |
| `rutas` | `grafo_dir`, `caida_dir`, `ribs_dir` (+ `rib_file`, `rib_file_bgp2vec`), `tesis_path`, `internetgraph_file` (grafo 2022), `tor_data_dir`, `mes`/`ano`; para `notebooks/`: `as_graph_dir`, `experimento_dir`, `grafo_antiguo_dir` (+ nombres de los CSV) |
| `comunes` | `seed`, `class_names` |
| `global_config_endtoend` | Hiperparámetros del end-to-end (`GLOBAL_CONFIG` del notebook), `split`, `n_neighbors` y `batch_size_caso2` (Caso 2), `num_parts` y `batch_size_caso3` (Caso 3), `random_features` (Caso 4) |
| `global_config_por_partes` | Hiperparámetros del por partes (`GLOBAL_CONFIG`), `split` (se usa `train_size` en link prediction y split de nodos), `deepwalk`, `bgp2vec_epochs`, `bgp2vec_test_limit`, `MLP` (clasificador sobre los embeddings) |

* **Variables de entorno.** `GRAFO_DIR`, `CAIDA_DIR`, `SEED`, `RESULTS_ROOT` y `GNN_CONFIG` (JSON sobre `GLOBAL_CONFIG`), si existen, tienen prioridad sobre el YAML: `scripts/correr_experimento.py` las fija para cada grafo. `CONFIG_FILE` apunta a otro YAML (solo en los dos notebooks principales).
* **Decimales.** Escríbalos con punto (`0.0005`): PyYAML lee `5e-4` como texto.
* **`split` del end-to-end.** `split_edges_classification` exige `train_size + val_size < 1.0` (el resto es test).
* **Etiquetas de celda.** Las celdas siguen teniendo sus etiquetas (`config`, `caso0`, …): no quitarlas, `correr_experimento.py` ejecuta solo las celdas etiquetadas.

## Estructura

```
modules/gnn.py             Clase GNN: etiquetado CAIDA, CSV → torch_geometric.data.Data, splits sin fuga, poda
modules/gnn_models.py      Encoders GCN / GraphSAGE / GAT (2 y 3 capas, variantes con muestreo) y decoders MLP / Bilinear / DotProduct
modules/experiment.py      set_seed (reproducibilidad) y save_run (guarda métricas, pesos e historial)
config.yaml                Configuración central (rutas en F:/, semilla, hiperparámetros); los notebooks la leen con yaml.safe_load
modules/bgp2vec.py         BGP2Vec (Word2Vec sobre AS_PATHs) con lectura en streaming
modules/graph.py           Pipeline legado multi-snapshot (no usado por los notebooks)
scripts/correr_experimento.py   Corre ambos notebooks sobre todos los grafos de un experimento
src/                       LACNIC, API PeeringDB, filtro de topología, utilidades (sin uso desde que se eliminaron los scripts)
AS_relationship_inference_end_to_end.ipynb   Entrenamiento end-to-end (encoder + decoder de aristas)
AS_relationship_inference_por_partes.ipynb   Embeddings (link prediction, regresión, DeepWalk, BGP2Vec) + clasificador ANN
notebooks/                 Análisis del grafo, de las RIBs, de atributos y visualización de embeddings (aún con rutas antiguas).
                           notebooks/resultados_experimentos.ipynb muestra en tablas los resultados de los dos enfoques
```

El pipeline de datos anterior (`scripts/1..4_*.py`, `scripts/create_tor_dataset.py`) se eliminó en el commit `4fcb754`. Sigue en el historial de git y en `main` / `v1.0-dgl`.

## Instalación

```bash
python3.10 -m venv env310 && source env310/bin/activate
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install pandas numpy networkx scikit-learn scipy matplotlib seaborn tqdm gensim==4.4.0 requests pyyaml umap-learn jupyter ipykernel
python -m ipykernel install --user --name env310
```

(Para CPU use los índices `whl/cpu` y `torch-2.3.0+cpu.html`.)

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

Rutas: ver la sección *Configuración* (`config.yaml`). Limitaciones conocidas en este entorno:
* `pyg_lib`, `torch_scatter`, `torch_sparse` y `torch_cluster` no están instaladas; pueden fallar los casos que usan `LinkNeighborLoader` (end-to-end Caso 2), `ClusterData` (Caso 3) y `Node2Vec` (DeepWalk).
* El Caso 2 del por partes lee `caida_pyg.pt` (el grafo 2022 convertido desde `caida.bin`, formato DGL); la conversión requiere DGL y ese archivo todavía no existe en `F:`.
* Prueba de humo: `python scripts\correr_experimento.py F:\as-level-internet-graph\experimentos\experimento7 --grafos 20260916 --rapido --notebooks end_to_end --casos-e2e caso0`.

## Preparar un grafo

1. **Construir el grafo** en `as-level-internet-graph` (ver su README). Conviene hacerlo el día 1 del mes, que coincide con CAIDA.
2. **Descargar CAIDA del mes del grafo** en `CAIDA_DIR`:
   ```bash
   wget -P /media/vale/KINGSTON/TESIS/data_2026/ https://publicdata.caida.org/datasets/as-relationships/serial-1/20260901.as-rel.txt.bz2
   ```
3. **Etiquetar.** La celda de configuración de los notebooks lo hace sola si falta `gnn_edges_caida.csv`. A mano:
   ```python
   from pathlib import Path
   from modules.gnn import GNN

   GRAFO_DIR = Path("/media/vale/KINGSTON/as-level-internet-graph/experimentos/experimento7/grafos/2026/09/20260916T0800")
   CAIDA_DIR = Path("/media/vale/KINGSTON/TESIS/data_2026")
   GNN(debug=True).label_edges_from_caida(GRAFO_DIR / "gnn_edges.csv",
                                          GNN.caida_file_for_graph(GRAFO_DIR, CAIDA_DIR),  # CAIDA_DIR/20260901.as-rel.txt.bz2
                                          output_csv=GRAFO_DIR / "gnn_edges_caida.csv")
   ```
   Solo se etiquetan los enlaces del grafo, en el sentido de cada fila. Lo que no está en CAIDA queda en `-1`; no se agregan aristas ni nodos.

   Con `experimento7 / 20260916T0800` + CAIDA `20260901` queda etiquetado el 73.9 % de los enlaces: 493 982 de 668 426 (P2P 329 589 · C2P 28 715 · P2C 135 678).

## Entrenamiento

Abra Jupyter desde la raíz del repositorio (los notebooks importan `modules/`):

```bash
jupyter lab
```

* `AS_relationship_inference_end_to_end.ipynb`: fije rutas e hiperparámetros en `config.yaml` (`GRAFO_DIR`, `GLOBAL_CONFIG_ENDTOEND`, …), y ejecute el caso deseado (Caso 0 atributos PeeringDB, 1 centralidades, 2 NeighborSampling, 3 ClusterGCN, 4 atributos aleatorios, comparación completa).
* `AS_relationship_inference_por_partes.ipynb`: fije rutas e hiperparámetros en `config.yaml` (`GRAFO_DIR`, `RIBS_DIR`, `GLOBAL_CONFIG_POR_PARTES`, …); luego ejecute un caso de la etapa de embeddings (link prediction, regresión de grado/PageRank, DeepWalk o BGP2Vec) y después la sección "Clasificación Aristas", eligiendo el embedding a cargar.

### Correr un experimento completo

`scripts/correr_experimento.py` ejecuta los dos notebooks, caso por caso, sobre todos los grafos de un experimento. En cada grafo:
1. Prepara los datos: descarga CAIDA si falta, etiqueta y crea el dataset ToR del grafo.
2. Corre los Casos 0–4 y la comparación del end-to-end.
3. Corre los embeddings del notebook por partes y luego el clasificador ANN con cada embedding.

```bash
EXP=/media/vale/KINGSTON/as-level-internet-graph/experimentos/experimento7
env310/bin/python scripts/correr_experimento.py $EXP --rapido     # prueba de humo, resultados aparte
env310/bin/python scripts/correr_experimento.py $EXP              # corrida real (muchas horas por grafo en CPU)
env310/bin/python scripts/correr_experimento.py $EXP --grafos 2026 --notebooks end_to_end --casos-e2e caso0
env310/bin/python scripts/correr_experimento.py $EXP --reanudar   # retoma donde quedó
```

Cada caso corre en un kernel nuevo, con solo sus celdas (etiquetas `config` / `casoN` en los metadatos). Si un caso falla, se registra y la corrida sigue.

El estado de cada caso queda en `$EXP/resultados_gnn/ejecucion_<fecha>.csv`, y todas las métricas de todos los grafos en `$EXP/resultados_gnn/metricas.csv`. Más detalle en `DOCUMENTATION.md` §4.7.

### Reproducibilidad y resultados

* **Semilla.** Ambos notebooks definen `SEED = 42` en la celda de configuración. Con ella se hacen todos los splits, y `set_seed(SEED)` se llama antes de crear cada modelo. Por eso, en CPU, una misma celda da siempre el mismo resultado, aunque se vuelva a ejecutar sola o se comenten otros modelos. En GPU puede variar levemente.
* **Resultados end-to-end.** Cada modelo guarda `metrics.json`, `model.pth` (pesos de la mejor época en validación) e `history.csv` en `GRAFO_DIR/results/end_to_end/<caso>/<modelo>_<decoder>_seed<SEED>/`, y agrega una fila a `GRAFO_DIR/results/end_to_end/resumen.csv` para comparar corridas (incluye `tiempo_s`, `epocas` y `config_id`). La carpeta de cada modelo termina en un `config_id` (hash de hiperparámetros y clase del modelo): cambiar hiperparámetros o capas crea una carpeta nueva en vez de sobrescribir. En por partes, cada embedding guarda su tiempo en `<embedding>.tiempo.json` y el clasificador ANN reporta el tiempo total embedding + clasificador.
* **Resultados por partes.** Embeddings, modelos y figuras quedan en `GRAFO_DIR/results/EnfoquePorPartes/Caso*/`.

Convención de etiquetas: `0 = P2P`, `1 = C2P` (origen es cliente del destino), `2 = P2C`, `-1 = sin etiqueta`. Cada enlace se representa con dos aristas dirigidas con etiquetas inversas (`1 ↔ 2`).
