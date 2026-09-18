# gnn-as-relationships

Inferencia del tipo de relación entre Sistemas Autónomos de Internet (**P2P / C2P / P2C**) con Graph Neural Networks sobre el grafo AS-level construido desde RIBs BGP, enriquecido con PeeringDB y etiquetado con CAIDA AS-Relationships.

Stack: Python 3.10 · PyTorch 2.3 · **DGL 1.1.3** · NetworkX · pandas · scikit-learn · gensim (BGP2Vec) · pybgpstream.

> Documentación técnica completa (arquitectura, pipeline, decisiones de diseño, deuda técnica): **[DOCUMENTATION.md](DOCUMENTATION.md)**.

## Ubicación de los datos

Los archivos de los grafos viven en el disco externo, fuera del repositorio:

```
/media/vale/KINGSTON/as-level-internet-graph/grafos/
```

Ahí leen y escriben todos los scripts (`data_path` / `output_dir`) y notebooks (`data_path` / `DATA_PATH` / `BASE`):

| Archivo | Producido por |
|---|---|
| `rib_<inicio>_to_<fin>.txt` | `scripts/1_ribs_extraction.py` — un AS_PATH por línea (`AS1\|AS2\|…`) |
| `nodes_rib_*.csv`, `edges_rib_*.csv` (+ `_pruned`) | `scripts/2_create_edges_from_ribs.py` |
| `nodes_cl_rib_*.csv`, `edges_cl_rib_*.csv` | `scripts/3_filter_by_country.py` (opcional) |
| `nodes_rib_*_enriched_tesis.csv`, `edges_rib_*_enriched_tesis.csv` | `scripts/4_add_atributes.py` — **entrada de los notebooks** |
| `peeringdb_2_dump_<año>_<mes>_01.json`, `<fecha>.as-rel.txt.bz2` | Fuentes externas (PeeringDB vía CAIDA, CAIDA AS-Relationships) |
| `results/EnfoquePorPartes/Caso*/` | Embeddings y modelos del enfoque por partes |

El mismo directorio es la salida del proyecto hermano [`as-level-internet-graph`](https://github.com/valesteban/as-level-internet-graph), que genera `grafos/<año>/<mes>/<fecha>/as_edges.csv` (`as_a,as_b,weight`). Ese formato no es el que consume `GNN.load_dataset`; para entrenar con uno de esos grafos hay que convertirlo antes al formato `nodes_*/edges_*` y pasar por `4_add_atributes.py`.

El dataset ToR (`ToR_data/`) y el grafo externo 2022 usados solo por el enfoque por partes siguen en `/media/vale/KINGSTON/TESIS/`.

Una copia del grafo del 1 de marzo de 2026 (24 h de RIBs) ya enriquecida está incluida en [`data/`](data/) para poder ejecutar los notebooks sin rehacer el pipeline.

## Estructura

```
modules/gnn.py             Clase GNN: CSV → DGLGraph, etiquetado CAIDA, splits sin fuga, poda
modules/gnn_models.py      Encoders GCN / GraphSAGE / GAT (2 y 3 capas, variantes con muestreo) y decoders MLP / Bilinear / DotProduct
modules/bgp2vec.py         BGP2Vec (Word2Vec sobre AS_PATHs) con lectura en streaming
modules/graph.py           Pipeline legado multi-snapshot (no usado por los notebooks)
scripts/1..4_*.py          Pipeline de datos, en orden
scripts/create_tor_dataset.py   Dataset ToR desde CAIDA (enfoque por partes)
src/                       LACNIC, API PeeringDB, filtro de topología, utilidades
AS_relationship_inference_end_to_end.ipynb   Entrenamiento end-to-end (encoder + decoder de aristas)
AS_relationship_inference_por_partes.ipynb   Embeddings (link prediction, regresión, DeepWalk, BGP2Vec) + clasificador ANN
notebooks/                 Análisis del grafo, de las RIBs, de atributos y visualización de embeddings
```

## Instalación

```bash
python3.10 -m venv env310 && source env310/bin/activate
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
pip install pandas numpy networkx scikit-learn scipy matplotlib seaborn tqdm gensim==4.4.0 requests pyyaml umap-learn jupyter ipykernel
pip install pybgpstream==2.0.4      # solo para el paso 1; requiere libbgpstream instalada
python -m ipykernel install --user --name env310
```

(Para CPU use los índices `whl/cpu` y `wheels/torch-2.3/repo.html`.)

## Pipeline de datos

Los scripts no reciben argumentos: la configuración (fechas, archivos, rutas) está en el bloque `if __name__ == '__main__':` de cada uno.

```bash
python3 scripts/1_ribs_extraction.py        # RIBs (BGPStream) → rib_*.txt
python3 scripts/2_create_edges_from_ribs.py # rib_*.txt → nodes_*.csv / edges_*.csv
python3 scripts/3_filter_by_country.py      # (opcional) subgrafo por país (LACNIC + IXPs + facilities + vecinos)
python3 scripts/4_add_atributes.py          # centralidades + PeeringDB + etiquetas CAIDA → *_enriched_tesis.csv
python3 scripts/create_tor_dataset.py       # (solo enfoque por partes) dataset ToR .npy
```

Antes del paso 4 descargue el archivo de relaciones CAIDA del mismo mes en el directorio de grafos, p. ej. `20260301.as-rel.txt.bz2` desde `https://publicdata.caida.org/datasets/as-relationships/serial-1/`. El dump de PeeringDB se descarga solo si falta.

## Entrenamiento

Abra Jupyter desde la raíz del repositorio (los notebooks importan `modules/`):

```bash
jupyter lab
```

* `AS_relationship_inference_end_to_end.ipynb`: ajuste `nodes_file`/`edges_file` en la celda de rutas, `GLOBAL_CONFIG` para hiperparámetros, y ejecute el caso deseado (Caso 0 atributos PeeringDB, 1 centralidades, 2 NeighborSampling, 3 ClusterGCN, 4 atributos aleatorios, comparación completa).
* `AS_relationship_inference_por_partes.ipynb`: primero un caso de la etapa de embeddings (link prediction, regresión de grado/PageRank, DeepWalk o BGP2Vec), luego la sección "Clasificación Aristas" eligiendo el embedding a cargar.

Convención de etiquetas: `0 = P2P`, `1 = C2P` (origen es cliente del destino), `2 = P2C`, `-1 = sin etiqueta`. Cada enlace se representa con dos aristas dirigidas con etiquetas inversas (`1 ↔ 2`).
