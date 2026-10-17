# Contexto del proyecto para Claude

Inferencia del tipo de relación entre Sistemas Autónomos de Internet (P2P / C2P / P2C) con Graph Neural Networks sobre el grafo AS-level construido desde RIBs BGP y enriquecido con PeeringDB por el repo hermano `as-level-internet-graph` (`/home/vale/Escritorio/GIT/LABORATORIO/as-level-internet-graph`), y etiquetado aquí con CAIDA AS-Relationships. Código de tesis (NIC Chile Research Labs / DCC, Universidad de Chile). Documentación técnica completa en `DOCUMENTATION.md`; guía rápida en `README.md`.

## Versiones: PyG (esta rama) vs DGL (original)

- **Esta es una versión nueva del código que usa PyTorch Geometric (PyG)** en lugar de DGL. Vive en la rama `migracion-pyg`.
- **La versión antigua, con DGL 1.1.3, está en la rama principal `main`** y quedó congelada con el **tag `v1.0-dgl`** (commit `4aff18d`). No modificar esa versión; cualquier trabajo nuevo va en `migracion-pyg` (o ramas derivadas).
- La migración cambió únicamente la librería de grafos; la organización del repositorio, los nombres de clases/métodos, la lógica de carga, los splits, las pérdidas e hiperparámetros son los mismos. Los splits con la misma semilla producen exactamente los mismos tamaños que en DGL.
- Para comparar ambas versiones: `git diff v1.0-dgl migracion-pyg -- modules/`.

## Cómo está organizado

- `modules/gnn.py`: clase `GNN`. `GNN.graph` es un `torch_geometric.data.Data` con `x` (features de nodo), `edge_index`, `edge_attr` (peso de arista, `(E,1)`), `edge_label` (0/1/2/-1), `train_mask`/`val_mask`/`test_mask` (aristas) y `node_train_mask`/`node_test_mask` (nodos). Métodos: `label_edges_from_caida` (agrega `relationship` a `gnn_edges.csv` → `gnn_edges_caida.csv` + `etiquetas_caida.json`), `caida_file_for_graph` (CAIDA `AAAAMM01` del mes del grafo), `load_dataset`, `load_dataset_only_cntrality_attr`, `split_edges_classification` (sin fuga entre direcciones), `split_edges_link_prediction`, `split_graph_nodes`, `add_random_features`, `remove_low_degree_nodes`, `_fill_labels_from_caida_stream_fast`. Tiempos: `GNN.timer(nombre)` (context manager, `.segundos`), `formato_tiempo`, `save_tiempo_embedding` / `load_tiempo_embedding` (`<embedding>.tiempo.json`).
- `modules/gnn_models.py`: encoders `GCN2L/3L`, `GraphSAGE2L/3L`, `GAT2L/3L` y variantes `*Sample2L` (mini-batch); decoders `MLPPredictor`, `BilinearPredictor`, `DotProductPredictor`. Interfaz: `encode(x, edge_index)`, `decodeMLP(edge_index, h, edge_attr=None)`, `decodeBilinear(edge_index, h)`, `decodeDotProduct(edge_index, h)`, `forward(x, edge_index)`.
- `modules/experiment.py`: `set_seed(seed)` fija random/numpy/torch/PyG y activa algoritmos deterministas; `save_run(RESULTS_DIR, caso, modelo, decoder, seed=…, y_true, y_pred, class_names, state_dict, history, config, extra)` escribe `metrics.json`, `model.pth` e `history.csv` en `RESULTS_DIR/<caso>/<modelo>_<decoder>_seed<seed>_<config_id>/` (`config_id` = `config_id(config, extra)`: hash de `config` y `extra["clase"]`, así variar hiperparámetros o capas no sobrescribe) y una fila en `RESULTS_DIR/resumen.csv`. Al guardar, pasar `extra["clase"] = Model.__name__`. `notebooks/resultados_experimentos.ipynb` muestra los resultados en tablas.
- `config.yaml` (raíz): configuración central (rutas, hoy en `F:/`; `comunes.seed`; `global_config_endtoend` y `global_config_por_partes` con hiperparámetros, `split`, vecinos/batch/clusters, DeepWalk, BGP2Vec, `MLP`). No hay módulo cargador: cada notebook lo lee con `yaml.safe_load` en su celda `config` (busca el archivo hacia arriba desde el cwd) y lo usa como `cfg` / `GLOBAL_CONFIG`; `GRAFO_DIR`, `CAIDA_DIR`, `SEED`, `RESULTS_ROOT` y `GNN_CONFIG` siguen leyéndose primero de variables de entorno porque `scripts/correr_experimento.py` las fija por grafo, y el script toma `rutas.caida_dir` como `--caida-dir` por defecto. Decimales con punto (`0.0005`): PyYAML lee `5e-4` como texto. `split_edges_classification` exige `train_size + val_size < 1.0`.
- `modules/graph.py`: pipeline legado (no lo usan los notebooks). `modules/bgp2vec.py`: BGP2Vec (gensim).
- `scripts/correr_experimento.py <experimento>`: ejecuta ambos notebooks sobre todos los grafos de un experimento, con `nbclient`. En cada grafo:
  - prepara los datos: descarga CAIDA, `label_edges_from_caida`, `build_tor_dataset`;
  - ejecuta cada caso en un kernel nuevo, solo con las celdas etiquetadas `config` + `<caso>` (`metadata.tags`);
  - pasa los parámetros por variables de entorno que lee la celda de configuración: `GRAFO_DIR`, `CAIDA_DIR`, `SEED`, `RESULTS_ROOT`, `GNN_CONFIG`, `TOR_PATH`, `EMBEDDINGS_PATH`.

  Si se agregan o mueven celdas en los notebooks, **mantener sus etiquetas**, o el script dejará de ejecutarlas. `--rapido` es la prueba de humo: escribe en `GRAFO_DIR/prueba_rapida/`.
- Ya no hay pipeline de datos en este repo: `scripts/1..4_*.py` y `create_tor_dataset.py` se eliminaron en `4fcb754` (siguen en git y en `main`). El grafo lo construye `as-level-internet-graph`; aquí solo se etiqueta con CAIDA.
- `AS_relationship_inference_end_to_end.ipynb` y `AS_relationship_inference_por_partes.ipynb`: aquí viven los training loops (full-batch, `LinkNeighborLoader`, `ClusterLoader`, link prediction, regresión nodal, `Node2Vec`, BGP2Vec, clasificador ANN). `notebooks/`: análisis y visualización.

## Convenciones importantes

- Etiquetas: `0 = P2P`, `1 = C2P` (origen es cliente del destino), `2 = P2C`, `-1 = sin etiqueta`. Cada enlace son dos aristas dirigidas con etiquetas inversas (`1 ↔ 2`, `0 ↔ 0`). Esa regla está en `GNN._reverse_relationship_value` (la usan `_make_edges_bidirectional` y `label_edges_from_caida`); no romperla. `label_edges_from_caida` solo etiqueta: no agrega aristas ni nodos de CAIDA ausentes del grafo.
- El orden de las aristas (filas del CSV tras la bidireccionalización) determina los splits: no reordenar `edge_index` al cargar.
- Reproducibilidad: una sola `SEED` en la celda de configuración de cada notebook. Cada celda de entrenamiento nueva debe:
  - llamar a `set_seed(SEED)` antes de crear cada modelo;
  - usar `seed=SEED` en los splits;
  - en end-to-end, llamar a `save_run(RESULTS_DIR, …)` después de evaluar en test.

  Para guardar el mejor estado, copiar los pesos (`{k: v.detach().clone() …}`), nunca `best_w = model.state_dict()`, que apunta a los pesos vivos.
- `RESULTS_DIR = GRAFO_DIR/results/end_to_end` (end-to-end) y `data_path = GRAFO_DIR/` + `results/EnfoquePorPartes/Caso*/` (por partes): los resultados viven junto a cada grafo, en el disco externo.
- Los grafos están fuera del repo, en `/media/vale/KINGSTON/as-level-internet-graph/experimentos/experimento<N>/grafos/<año>/<mes>/<AAAAMMDDTHHMM>/`:
  - `as_features.csv` y `gnn_edges.csv` los genera el otro repo, que no se toca;
  - `gnn_edges_caida.csv` y `etiquetas_caida.json` los escribe este repo.

  El grafo cambia: los notebooks lo eligen con **una sola variable `GRAFO_DIR`** en la celda de configuración (celda 6 end-to-end, celda 3 por partes) y derivan de ella `nodes_file`, `edges_file`, `RUN_TAG`/`data_path` y el archivo CAIDA. No hardcodear nombres de grafo en otras celdas.
- La carpeta antigua `as-level-internet-graph/grafos/` ya no existe y `data/` se eliminó del repo. En `/media/vale/KINGSTON/TESIS/data_2026/` están:
  - CAIDA (`AAAAMM01.as-rel.txt.bz2`, `CAIDA_DIR`);
  - las RIBs en texto para BGP2Vec (`RIBS_DIR`);
  - el grafo antiguo de marzo (`*_enriched_tesis.csv`, 72 features);
  - el dataset ToR.

  El grafo externo 2022 sigue en `/media/vale/KINGSTON/TESIS/`. Los notebooks de `notebooks/` todavía apuntan a la ruta antigua.
- `.gitignore` ignora `*.md`, `*.csv`, `*.json`, `*.txt`, etc.; los archivos Markdown del repo están como excepciones explícitas (`!README.md`, `!DOCUMENTATION.md`, `!CLAUDE.md`).

## Entorno

- Python 3.10, torch 2.3.0, `torch_geometric` 2.8 + `pyg_lib`, `torch_scatter`, `torch_sparse`, `torch_cluster` (necesarios para `LinkNeighborLoader`, `ClusterData` y `Node2Vec`). Venv local en `env310/` (ignorado por git); kernel Jupyter `env310`.
- La máquina de desarrollo no tiene GPU y tiene 15 GB de RAM: los encoders GAT full-batch con `hidden=256` sobre el grafo completo se quedan sin memoria. Para pruebas rápidas usar dims pequeñas (p. ej. `hidden=32`, `out=16`) y pocas épocas.
- Prueba de humo mínima: `GNN.load_dataset(GRAFO_DIR/as_features.csv, GRAFO_DIR/gnn_edges_caida.csv)` con `experimento7/grafos/2026/09/20260916T0800` → 87 796 nodos, 1 336 852 aristas, 85 features; `split_edges_classification(0.7, 0.15, seed=42)` → 691 574 / 148 194 / 148 196. Etiquetado con CAIDA `20260901`: 493 982 de 668 426 enlaces (73.9 %).
- Regresión con el grafo antiguo (`TESIS/data_2026/*_enriched_tesis.csv`): 79 504 nodos, 807 496 aristas, 72 features → 462 306 / 99 064 / 99 068.

## Preferencias de trabajo

- Responder y documentar en español.
- No añadir líneas de atribución ni referencias a Claude en mensajes de commit ni en el código.
- No corregir bugs preexistentes ni refactorizar salvo que se pida explícitamente; los defectos conocidos están listados en `DOCUMENTATION.md` §5.1.
