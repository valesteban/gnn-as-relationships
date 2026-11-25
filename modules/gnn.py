import torch
import numpy as np
from collections import defaultdict
import random
import bz2, gzip
import json, re
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict
import random
import torch
import numpy as np
from collections import defaultdict, Counter

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import degree as pyg_degree, subgraph


TOR_LABELS_DICT = {'P2P':0, 'C2P': 1,'P2C': 2}

import pandas as pd
import torch


def _edge_keys(data: Data):
    """Nombres de los atributos de arista de un Data (equivale a g.edata.keys())."""
    return [k for k in data.keys() if k != 'edge_index' and data.is_edge_attr(k)]


def _node_keys(data: Data):
    """Nombres de los atributos de nodo de un Data (equivale a g.ndata.keys())."""
    return [k for k in data.keys() if data.is_node_attr(k)]


class GNN:
    def __init__(self, debug=False):
        self.debug = debug
        self.graph = None

    def _reverse_relationship_value(self, value):
        """Invierte la relación para la arista en sentido contrario.

        Convención usada:
        - 0 (P2P) -> 0
        - 1 (C2P) -> 2 (P2C)
        - 2 (P2C) -> 1 (C2P)
        También soporta valores string equivalentes.
        """
        mapping = {
            0: 0, 1: 2, 2: 1,
            "0": 0, "1": 2, "2": 1,
            "P2P": 0,
            "C2P": 2,
            "P2C": 1,
        }
        return mapping.get(value, value)
    # TODO: rvisar
    def _make_edges_bidirectional(self, df_e: pd.DataFrame) -> pd.DataFrame:
        """Devuelve un DataFrame de aristas en ambos sentidos sin duplicados exactos."""
        required_cols = {"src_id", "dst_id"}
        if not required_cols.issubset(df_e.columns):
            missing = required_cols - set(df_e.columns)
            raise ValueError(f"Faltan columnas requeridas en edges_csv: {missing}")

        df_rev = df_e.copy()
        df_rev[["src_id", "dst_id"]] = df_rev[["dst_id", "src_id"]]

        # Invertir etiqueta solo para la copia en sentido contrario
        if "relationship" in df_rev.columns:
            df_rev["relationship"] = df_rev["relationship"].map(self._reverse_relationship_value)

        df_bi = pd.concat([df_e, df_rev], ignore_index=True)
        df_bi = df_bi.drop_duplicates(subset=["src_id", "dst_id"], keep="first")
        df_bi = df_bi.reset_index(drop=True)
        return df_bi

    # TODO: rvisar qu hac
    def _normalize_relationship_column(self, df_e: pd.DataFrame) -> pd.DataFrame:
        """Normaliza columna relationship a int (0/1/2 y opcionalmente -1)."""
        if "relationship" not in df_e.columns:
            return df_e

        # Estandariza etiquetas string -> numéricas
        raw = df_e["relationship"]
        mapped = raw.map(
            lambda x: {
                "P2P": 0,
                "C2P": 1,
                "P2C": 2,
                "-1": -1,
                "0": 0,
                "1": 1,
                "2": 2,
            }.get(x, x)
        )

        # Fuerza numérico y falla explícitamente si hay valores no válidos
        numeric = pd.to_numeric(mapped, errors="coerce")
        if numeric.isna().any():
            bad = sorted(set(raw[numeric.isna()].astype(str).tolist()))
            raise ValueError(
                f"Valores de 'relationship' no reconocidos en edges_csv: {bad}"
            )

        df_e = df_e.copy()
        df_e["relationship"] = numeric.astype("int64")
        return df_e

    def load_dataset(self, nodes_csv, edges_csv, add_reverse_edges=True):
        # 1. Cargar CSVs
        # -----------------------------
        df_n = pd.read_csv(nodes_csv)
        df_e = pd.read_csv(edges_csv)

        if add_reverse_edges:
            df_e = self._make_edges_bidirectional(df_e)
        df_e = self._normalize_relationship_column(df_e)

        # 2. Crear estructura del grafo
        # -----------------------------
        # src_id y dst_id deben coincidir con el índice de las filas de df_n
        src = torch.as_tensor(df_e['src_id'].to_numpy(dtype='int64'))
        dst = torch.as_tensor(df_e['dst_id'].to_numpy(dtype='int64'))
        self.graph = Data(edge_index=torch.stack([src, dst]), num_nodes=len(df_n))

        # Guardar mapeo asn → node_id (lo usa AS_relationship_evaluacion_otro_grafo.ipynb)
        if 'asn' in df_n.columns:
            self.asn_to_node_id = dict(zip(df_n['asn'].astype(int), df_n['node_id'].astype(int)))
        else:
            self.asn_to_node_id = None

        # 3. Nodos: Extraer features
        # -----------------------------
        # Excluimos IDs y metadatos; el resto son las columnas que procesamos de PeeringDB
        feat_cols = [c for c in df_n.columns if c not in ['node_id', 'asn', 'country']]
        self.graph.x = torch.tensor(df_n[feat_cols].values, dtype=torch.float32)

        # 4. Aristas: Extraer etiquetas y features
        # -----------------------------
        if 'relationship' in df_e.columns:
            # Importante: para clasificación, labels deben ser long (enteros)
            self.graph.edge_label = torch.tensor(df_e['relationship'].values, dtype=torch.long)

        if 'weight' in df_e.columns:
            ew = torch.tensor(df_e['weight'].values, dtype=torch.float32)
            self.graph.edge_attr = ew.unsqueeze(1)  # (E, 1)

        if self.debug:
            print(f"Grafo cargado: {self.graph.num_nodes} nodos, {self.graph.num_edges} aristas")
            print(f"Aristas en ambos sentidos: {add_reverse_edges}")
            print(f"Dimensión de entrada (in_feats): {self.graph.x.shape[1]}")
            if hasattr(self.graph, 'edge_attr') and self.graph.edge_attr is not None:
                print(f"Edge features: {self.graph.edge_attr.shape[1]} dim")

    # TODO: rvisar s usa n caso 1 ndtond
    def load_dataset_only_cntrality_attr(self, nodes_csv, edges_csv, add_reverse_edges=True):

        # 1. Cargar CSVs
        # -----------------------------
        df_n = pd.read_csv(nodes_csv)
        df_e = pd.read_csv(edges_csv)

        if add_reverse_edges:
            df_e = self._make_edges_bidirectional(df_e)
        df_e = self._normalize_relationship_column(df_e)

        # 2. Crear estructura del grafo
        # -----------------------------
        src = torch.as_tensor(df_e['src_id'].to_numpy(dtype='int64'))
        dst = torch.as_tensor(df_e['dst_id'].to_numpy(dtype='int64'))
        self.graph = Data(edge_index=torch.stack([src, dst]), num_nodes=len(df_n))

        # 3. Nodos: solo weight + atributos de centralidad
        # -----------------------------
        centrality_cols = [
            'weight',
            'PageRank',
            'degree_centrality',
            'betweenness_centrality',
            'eigenvector_centrality',
            # Formato as-level-internet-graph (as_features.csv)
            'degree',
            'weighted_degree',
            'transit_degree',
            'pagerank',
            'k_core',
        ]
        feat_cols = [c for c in centrality_cols if c in df_n.columns]
        if not feat_cols:
            raise ValueError(
                f"Ninguna columna de centralidad encontrada en {nodes_csv}. "
                f"Columnas disponibles: {list(df_n.columns)}"
            )
        feat_arr = df_n[feat_cols].values.astype('float32')
        # Centralidad features tienen escalas muy distintas (PageRank~1e-5, degree~1).
        # StandardScaler (media=0, std=1) evita que el GNN colapse al entrenar.
        mean = feat_arr.mean(axis=0)
        std  = feat_arr.std(axis=0)
        std[std == 0] = 1.0  # evitar división por cero en columnas constantes
        feat_arr = (feat_arr - mean) / std
        self.graph.x = torch.tensor(feat_arr, dtype=torch.float32)

        # 4. Aristas: etiquetas de relación y features
        # -----------------------------
        if 'relationship' in df_e.columns:
            self.graph.edge_label = torch.tensor(
                df_e['relationship'].values, dtype=torch.long
            )

        if 'weight' in df_e.columns:
            ew = torch.tensor(df_e['weight'].values, dtype=torch.float32)
            self.graph.edge_attr = ew.unsqueeze(1)  # (E, 1)

        if self.debug:
            print(f"Grafo cargado: {self.graph.num_nodes} nodos, {self.graph.num_edges} aristas")
            print(f"Aristas en ambos sentidos: {add_reverse_edges}")
            print(f"Features usadas ({len(feat_cols)}): {feat_cols}")
            print(f"Dimensión de entrada (in_feats): {self.graph.x.shape[1]}")
            if hasattr(self.graph, 'edge_attr') and self.graph.edge_attr is not None:
                print(f"Edge features: {self.graph.edge_attr.shape[1]} dim")

    def _read_caida_relationships(self, caida_file):
        """Lee un archivo CAIDA AS-Relationships (serial-1).

        Formato CAIDA:
          <provider-as>|<customer-as>|-1
          <peer-as>|<peer-as>|0
        Cualquier otro tipo de relación se ignora.

        Devuelve {(asn_menor, asn_mayor): etiqueta}, con la etiqueta vista desde
        el ASN menor: 0 = P2P, 1 = C2P, 2 = P2C.
        """
        caida_file = str(caida_file)
        opener = (bz2.open  if caida_file.endswith(".bz2") else
                  gzip.open if caida_file.endswith(".gz")  else
                  open)

        relaciones = {}
        with opener(caida_file, "rt") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                campos = line.strip().split("|")
                a, b, rel = int(campos[0]), int(campos[1]), campos[2]
                if rel == "0":          # P2P simétrico
                    lbl = 0
                elif rel == "-1":       # a es proveedor de b
                    lbl = 2 if a < b else 1
                else:
                    continue
                relaciones[(min(a, b), max(a, b))] = lbl
        return relaciones

    # ===========================
    # Dscarga archivos
    # ============================
    @staticmethod
    def caida_file_for_graph(graph_dir, caida_dir):
        """Devuelve el archivo CAIDA del mes del grafo.

        La carpeta del grafo se llama AAAAMMDDTHHMM (as-level-internet-graph) y
        CAIDA publica un snapshot el día 1 de cada mes: para 20260916T0800 se usa
        caida_dir/20260901.as-rel.txt.bz2.
        """
        graph_dir = Path(graph_dir)
        m = re.match(r"(\d{4})(\d{2})\d{2}T\d{4}$", graph_dir.name)
        if m is None:
            raise ValueError(
                f"La carpeta del grafo debe llamarse AAAAMMDDTHHMM: {graph_dir.name}"
            )
        nombre = f"{m.group(1)}{m.group(2)}01.as-rel.txt.bz2"
        caida_file = Path(caida_dir) / nombre
        if not caida_file.exists():
            raise FileNotFoundError(
                f"No existe {caida_file}. Descargar desde "
                f"https://publicdata.caida.org/datasets/as-relationships/serial-1/{nombre}"
            )
        return caida_file

    # FIXME: guarad mas caras para bnchmark, pro s supon qu como socupa sd todas las vcs qcro las mascaras dbriaan sr iguals?
    def build_tor_dataset(self, output_dir, **info):
        """Exporta el split de clasificación actual como dataset ToR para el benchmark.

        Usa train_mask / val_mask / test_mask de split_edges_classification (o de
        split_edges_kfold), así los métodos de BenchmarckASRelationships se evalúan
        sobre las mismas aristas de test que la GNN. Cada arista dirigida del split es
        una fila: las dos direcciones de un enlace quedan en el mismo conjunto y con
        etiquetas inversas, sin fuga entre train y test.

        Escribe en output_dir: x_training.npy, x_val.npy y x_test.npy (pares de ASN como
        strings, forma (N, 2)); y_training.npy, y_val.npy e y_test.npy (0 = P2P, 1 = C2P,
        2 = P2C); y tor_resumen.json con los tamaños, la distribución de clases y `info`
        (p. ej. grafo, seed y split). Requiere un grafo cargado con load_dataset (mapeo
        asn → node_id) y un split previo. Devuelve el resumen.
        """
        g = self.graph
        if getattr(g, "test_mask", None) is None:
            raise RuntimeError("Falta el split: llamar antes a split_edges_classification o split_edges_kfold")
        if not getattr(self, "asn_to_node_id", None):
            raise RuntimeError("Falta asn_to_node_id: cargar el grafo con load_dataset")

        asn = np.zeros(g.num_nodes, dtype=np.int64)            # node_id → ASN
        asn[np.fromiter(self.asn_to_node_id.values(), dtype=np.int64)] = \
            np.fromiter(self.asn_to_node_id.keys(), dtype=np.int64)
        ei = g.edge_index.cpu().numpy()
        y_all = g.edge_label.long().cpu().numpy()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        resumen = {**info, "num_nodes": int(g.num_nodes), "num_edges": int(g.num_edges)}
        for nombre, mask in [("training", g.train_mask), ("val", g.val_mask), ("test", g.test_mask)]:
            m = mask.bool().cpu().numpy()
            x = asn[ei[:, m]].T.astype(f"<U{len(str(asn.max()))}")   # ASN como texto, mismo formato que el ToR antiguo
            y = y_all[m].astype(np.int64)
            np.save(output_dir / f"x_{nombre}.npy", x)
            np.save(output_dir / f"y_{nombre}.npy", y)
            resumen[nombre] = {"aristas": int(m.sum()),
                               "dist": {int(k): int(v) for k, v in sorted(Counter(y.tolist()).items())}}
        with open(output_dir / "tor_resumen.json", "w", encoding="utf-8") as f:
            json.dump(resumen, f, indent=2, ensure_ascii=False, default=str)
        if self.debug:
            print(f"[ToR] {output_dir}: train {resumen['training']['aristas']} | "
                  f"val {resumen['val']['aristas']} | test {resumen['test']['aristas']}")
        return resumen


    def split_edges_classification(self, train_size=0.7, val_size=0.15, seed=0,
                                    return_eids=False, store_eids=True,
                                    balance_mode="proportional"):
        """
        Divide aristas en train/val/test SIN fuga entre direcciones opuestas,
        y estratifica por tipo de par para reducir desbalance entre clases.

        Estratificación usada:
        - Se agrupan eids por par no dirigido (min(u,v), max(u,v)).
        - Cada par recibe una firma de etiquetas (p.ej. (0,0), (1,2), (2,), ...).
        - Se hace split por cada firma preservando proporciones train/val/test.

                Parámetros de balance:
                - balance_mode="proportional" (default): mantiene proporciones naturales.
                - balance_mode="strict_equal": submuestrea cada split para igualar
                    cantidad de aristas por clase (0/1/2) usando el mínimo disponible.
        """
        rng = random.Random(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        if not (0.0 < train_size < 1.0):
            raise ValueError(f"train_size debe estar entre 0 y 1. Recibido: {train_size}")
        if not (0.0 <= val_size < 1.0):
            raise ValueError(f"val_size debe estar entre 0 y 1. Recibido: {val_size}")
        if train_size + val_size >= 1.0:
            raise ValueError("train_size + val_size debe ser < 1.0 para dejar espacio a test")
        if balance_mode not in {"proportional", "strict_equal"}:
            raise ValueError(
                f"balance_mode inválido: {balance_mode}. Usa 'proportional' o 'strict_equal'"
            )

        u, v = self.graph.edge_index[0], self.graph.edge_index[1]

        label_key = next((k for k in ["edge_label", "label", "relationship", "Relationship"]
                          if getattr(self.graph, k, None) is not None), None)
        if label_key is None:
            raise KeyError(f"No se encontró etiqueta de arista en el grafo. Claves disponibles: {_edge_keys(self.graph)}")

        rel = getattr(self.graph, label_key).long()

        # Solo aristas con etiqueta válida
        is_lbl = rel >= 0
        labeled_eids = torch.where(is_lbl)[0].tolist()

        # 1) Agrupar por par no dirigido (no leakage)
        pair2eids = defaultdict(list)
        for eid in labeled_eids:
            ui, vi = int(u[eid].item()), int(v[eid].item())
            pair = (ui, vi) if ui <= vi else (vi, ui)
            pair2eids[pair].append(eid)

        # 2) Estrato por firma de etiquetas del par
        #    Ejemplos comunes: (0,0), (1,2), (2,), (1,)
        signature2pairs = defaultdict(list)
        for pair, eids in pair2eids.items():
            sig = tuple(sorted(int(rel[eid].item()) for eid in eids))
            signature2pairs[sig].append(pair)

        train_pairs, val_pairs, test_pairs = [], [], []

        for sig, pairs in signature2pairs.items():
            rng.shuffle(pairs)
            n = len(pairs)

            n_train = int(round(n * train_size))
            n_val = int(round(n * val_size))

            # Ajuste de bordes para no exceder n
            if n_train > n:
                n_train = n
            if n_train + n_val > n:
                n_val = max(0, n - n_train)

            # Garantizar test cuando haya suficiente cardinalidad
            if n >= 3 and (n_train + n_val) >= n:
                if n_val > 0:
                    n_val -= 1
                elif n_train > 1:
                    n_train -= 1

            train_pairs.extend(pairs[:n_train])
            val_pairs.extend(pairs[n_train:n_train + n_val])
            test_pairs.extend(pairs[n_train + n_val:])

        # Mezcla global para no dejar bloques por estrato
        rng.shuffle(train_pairs)
        rng.shuffle(val_pairs)
        rng.shuffle(test_pairs)

        def gather_eids(pairs_subset):
            return [eid for p in pairs_subset for eid in pair2eids[p]]

        train_list = gather_eids(train_pairs)
        val_list = gather_eids(val_pairs)
        test_list = gather_eids(test_pairs)

        train_eids = torch.tensor(train_list, dtype=torch.long)
        val_eids = torch.tensor(val_list, dtype=torch.long)
        test_eids = torch.tensor(test_list, dtype=torch.long)

        # 3) Balance opcional por clase dentro de cada split
        # Nota: se hace a nivel de aristas para no mover elementos entre splits.
        if balance_mode == "strict_equal":
            classes_present = sorted(set(int(x) for x in rel[labeled_eids].tolist() if int(x) >= 0))

            def downsample_equal(eids_tensor):
                if eids_tensor.numel() == 0:
                    return eids_tensor

                by_class = {}
                for c in classes_present:
                    mask_c = rel[eids_tensor] == c
                    cls_eids = eids_tensor[mask_c]
                    if cls_eids.numel() > 0:
                        by_class[c] = cls_eids

                # Si falta alguna clase en este split, no forzamos balance para evitar vaciarlo
                if len(by_class) < max(2, len(classes_present)):
                    return eids_tensor

                target = min(x.numel() for x in by_class.values())
                if target <= 0:
                    return eids_tensor

                selected = []
                for c in classes_present:
                    cls_eids = by_class[c]
                    perm = torch.randperm(cls_eids.numel())
                    selected.append(cls_eids[perm[:target]])

                out = torch.cat(selected)
                perm_out = torch.randperm(out.numel())
                return out[perm_out]

            train_eids = downsample_equal(train_eids)
            val_eids = downsample_equal(val_eids)
            test_eids = downsample_equal(test_eids)

        # 4) Crear máscaras booleanas
        num_e = self.graph.num_edges
        for name, eids in [("train_mask", train_eids), ("val_mask", val_eids), ("test_mask", test_eids)]:
            mask = torch.zeros(num_e, dtype=torch.bool)
            if eids.numel() > 0:
                mask[eids] = True
            setattr(self.graph, name, mask)

        # 5) Guardado opcional
        if store_eids:
            self.train_eids = train_eids
            self.val_eids = val_eids
            self.test_eids = test_eids

        if self.debug:
            def dist(eids_tensor):
                if eids_tensor.numel() == 0:
                    return {}
                return dict(Counter(rel[eids_tensor].tolist()))

            print("\n[SPLIT COMPLETO - ESTRATIFICADO]")
            print(f"Modo balanceo: {balance_mode}")
            print(f"Etiqueta usada: {label_key}")
            print(f"Total aristas etiquetadas: {len(labeled_eids)}")
            print(f"Train: {train_eids.shape[0]} | Val: {val_eids.shape[0]} | Test: {test_eids.shape[0]}")
            print(f"Distribución Train: {dist(train_eids)}")
            print(f"Distribución Val:   {dist(val_eids)}")
            print(f"Distribución Test:  {dist(test_eids)}")

        if return_eids:
            return train_eids, val_eids, test_eids


    def split_edges_kfold(self, k=5, fold=0, val_size=0.15, seed=0,
                          balance_mode="proportional", return_eids=False, store_eids=True):
        """
        Split K-fold de aristas para clasificación, SIN fuga entre direcciones opuestas.

        Misma unidad y estratificación que `split_edges_classification`:
        - Se agrupan eids por par no dirigido (min(u,v), max(u,v)).
        - Cada par recibe una firma de etiquetas (p.ej. (0,0), (1,2), (2,), ...).
        - Dentro de cada firma, los pares se barajan con `seed` (la misma permutación
          para todos los folds) y se reparten en k folds (fold_id = i % k).

        Para el fold `fold`:
        - test  = pares con fold_id == fold (≈ 1/k de los pares de cada firma).
        - val   = fracción `val_size` del resto, por firma (barajado con seed + fold),
                  para early stopping.
        - train = lo demás.

        Los k conjuntos de test son disjuntos y su unión cubre todas las aristas etiquetadas.
        Escribe train_mask/val_mask/test_mask (y train_eids/val_eids/test_eids) igual que
        `split_edges_classification`. `balance_mode` funciona igual que en ese método.
        """
        if not (isinstance(k, int) and k >= 2):
            raise ValueError(f"k debe ser un entero >= 2. Recibido: {k}")
        if not (isinstance(fold, int) and 0 <= fold < k):
            raise ValueError(f"fold debe estar entre 0 y k-1 ({k - 1}). Recibido: {fold}")
        if not (0.0 <= val_size < 1.0):
            raise ValueError(f"val_size debe estar entre 0 y 1. Recibido: {val_size}")
        if balance_mode not in {"proportional", "strict_equal"}:
            raise ValueError(
                f"balance_mode inválido: {balance_mode}. Usa 'proportional' o 'strict_equal'"
            )

        rng_folds = random.Random(seed)          # asignación a folds: igual para todos los folds
        rng_val = random.Random(seed + fold)     # separación de val dentro del fold
        torch.manual_seed(seed + fold)
        np.random.seed(seed + fold)

        u, v = self.graph.edge_index[0], self.graph.edge_index[1]

        label_key = next((k_ for k_ in ["edge_label", "label", "relationship", "Relationship"]
                          if getattr(self.graph, k_, None) is not None), None)
        if label_key is None:
            raise KeyError(f"No se encontró etiqueta de arista en el grafo. Claves disponibles: {_edge_keys(self.graph)}")

        rel = getattr(self.graph, label_key).long()

        # Solo aristas con etiqueta válida
        labeled_eids = torch.where(rel >= 0)[0].tolist()

        # 1) Agrupar por par no dirigido (no leakage)
        pair2eids = defaultdict(list)
        for eid in labeled_eids:
            ui, vi = int(u[eid].item()), int(v[eid].item())
            pair = (ui, vi) if ui <= vi else (vi, ui)
            pair2eids[pair].append(eid)

        # 2) Estrato por firma de etiquetas del par
        signature2pairs = defaultdict(list)
        for pair, eids in pair2eids.items():
            sig = tuple(sorted(int(rel[eid].item()) for eid in eids))
            signature2pairs[sig].append(pair)

        # 3) Folds por firma; val sale del resto del mismo estrato
        train_pairs, val_pairs, test_pairs = [], [], []
        for sig in sorted(signature2pairs):
            pairs = signature2pairs[sig]
            rng_folds.shuffle(pairs)

            test_sig = [p for i, p in enumerate(pairs) if i % k == fold]
            rest_sig = [p for i, p in enumerate(pairs) if i % k != fold]

            rng_val.shuffle(rest_sig)
            n_val = int(round(len(rest_sig) * val_size))
            # Dejar al menos un par en train cuando haya resto
            if rest_sig and n_val >= len(rest_sig):
                n_val = len(rest_sig) - 1

            test_pairs.extend(test_sig)
            val_pairs.extend(rest_sig[:n_val])
            train_pairs.extend(rest_sig[n_val:])

        # Mezcla global para no dejar bloques por estrato
        rng_val.shuffle(train_pairs)
        rng_val.shuffle(val_pairs)
        rng_val.shuffle(test_pairs)

        def gather_eids(pairs_subset):
            return torch.tensor([eid for p in pairs_subset for eid in pair2eids[p]], dtype=torch.long)

        train_eids = gather_eids(train_pairs)
        val_eids = gather_eids(val_pairs)
        test_eids = gather_eids(test_pairs)

        # 4) Balance opcional por clase dentro de cada split (igual que split_edges_classification)
        if balance_mode == "strict_equal":
            classes_present = sorted(set(int(x) for x in rel[labeled_eids].tolist() if int(x) >= 0))

            def downsample_equal(eids_tensor):
                if eids_tensor.numel() == 0:
                    return eids_tensor
                by_class = {}
                for c in classes_present:
                    cls_eids = eids_tensor[rel[eids_tensor] == c]
                    if cls_eids.numel() > 0:
                        by_class[c] = cls_eids
                if len(by_class) < max(2, len(classes_present)):
                    return eids_tensor
                target = min(x.numel() for x in by_class.values())
                if target <= 0:
                    return eids_tensor
                selected = []
                for c in classes_present:
                    cls_eids = by_class[c]
                    selected.append(cls_eids[torch.randperm(cls_eids.numel())[:target]])
                out = torch.cat(selected)
                return out[torch.randperm(out.numel())]

            train_eids = downsample_equal(train_eids)
            val_eids = downsample_equal(val_eids)
            test_eids = downsample_equal(test_eids)

        # 5) Crear máscaras booleanas
        num_e = self.graph.num_edges
        for name, eids in [("train_mask", train_eids), ("val_mask", val_eids), ("test_mask", test_eids)]:
            mask = torch.zeros(num_e, dtype=torch.bool)
            if eids.numel() > 0:
                mask[eids] = True
            setattr(self.graph, name, mask)

        # 6) Guardado opcional
        if store_eids:
            self.train_eids = train_eids
            self.val_eids = val_eids
            self.test_eids = test_eids

        if self.debug:
            def dist(eids_tensor):
                if eids_tensor.numel() == 0:
                    return {}
                return dict(Counter(rel[eids_tensor].tolist()))

            print(f"\n[SPLIT K-FOLD - ESTRATIFICADO] fold {fold + 1}/{k}")
            print(f"Modo balanceo: {balance_mode}")
            print(f"Etiqueta usada: {label_key}")
            print(f"Total aristas etiquetadas: {len(labeled_eids)}")
            print(f"Train: {train_eids.shape[0]} | Val: {val_eids.shape[0]} | Test: {test_eids.shape[0]}")
            print(f"Distribución Train: {dist(train_eids)}")
            print(f"Distribución Val:   {dist(val_eids)}")
            print(f"Distribución Test:  {dist(test_eids)}")

        if return_eids:
            return train_eids, val_eids, test_eids


    # TODO: rvisar funcion complta
    def split_edges_link_prediction(self, train_size: float = 0.8, val_size: float = 0.0,
                                    seed: int = 42, balance_mode=None, train_ratio=None):
        """
        Divide las aristas en train/val/test para link prediction SIN fuga entre
        direcciones opuestas: el split se hace por enlace no dirigido {u, v}, de modo
        que u->v y v->u siempre caen en el mismo conjunto.

        Crea:
            • self.train_g                      : grafo del encoder (solo enlaces de train)
            • self.train_pos_g / train_neg_g    : positivos / negativos de entrenamiento
            • self.val_pos_g   / val_neg_g      : positivos / negativos de validación
            • self.test_pos_g  / test_neg_g     : positivos / negativos de test

        Negativos: pares {u, v} con u != v sin arista en ninguna dirección en el grafo
        completo, distintos entre sí y entre splits, con dirección aleatoria; hay
        tantos negativos como positivos en cada split.

        Args
        ----
        train_size   : float  fracción de enlaces para train
        val_size     : float  fracción de enlaces para validación (0 → sin validación)
        seed         : int    semilla de reproducibilidad
        balance_mode : ignorado (link prediction no tiene clases); se acepta para poder
                       pasar `comunes.split` de config.yaml con **
        train_ratio  : nombre antiguo de train_size
        """
        if train_ratio is not None:
            train_size = train_ratio
        if not (0.0 < train_size < 1.0):
            raise ValueError(f"train_size debe estar entre 0 y 1. Recibido: {train_size}")
        if not (0.0 <= val_size < 1.0):
            raise ValueError(f"val_size debe estar entre 0 y 1. Recibido: {val_size}")
        if train_size + val_size >= 1.0:
            raise ValueError("train_size + val_size debe ser < 1.0 para dejar espacio a test")

        g = self.graph
        num_nodes = g.num_nodes
        if g.num_edges == 0:
            raise ValueError("El grafo no contiene aristas.")
        gen = torch.Generator().manual_seed(seed)

        # 1.- Agrupar aristas dirigidas por enlace no dirigido {min(u,v), max(u,v)}
        # --------------------------
        src, dst = g.edge_index[0].long(), g.edge_index[1].long()
        link_keys = torch.minimum(src, dst) * num_nodes + torch.maximum(src, dst)
        uniq_links, link_of_edge = torch.unique(link_keys, return_inverse=True)
        num_links = uniq_links.numel()

        # 2.- Barajar enlaces y asignar cada uno a un split (0 = train, 1 = val, 2 = test)
        # --------------------------
        perm = torch.randperm(num_links, generator=gen)
        n_train = int(num_links * train_size)
        n_val   = int(num_links * val_size)
        split_of_link = torch.full((num_links,), 2, dtype=torch.long)
        split_of_link[perm[:n_train]] = 0
        split_of_link[perm[n_train:n_train + n_val]] = 1
        split_of_edge = split_of_link[link_of_edge]

        train_eids = torch.where(split_of_edge == 0)[0]
        val_eids   = torch.where(split_of_edge == 1)[0]
        test_eids  = torch.where(split_of_edge == 2)[0]

        # 3.- Grafo del encoder: solo enlaces de train (ninguna dirección de val/test)
        # --------------------------
        keep = split_of_edge == 0
        self.train_g = Data(edge_index=g.edge_index[:, keep], num_nodes=num_nodes)
        for k in _edge_keys(g):
            setattr(self.train_g, k, getattr(g, k)[keep])

        # 4.- Subgrafos POSITIVOS
        # --------------------------
        self.train_pos_g = Data(edge_index=g.edge_index[:, train_eids], num_nodes=num_nodes)
        self.val_pos_g   = Data(edge_index=g.edge_index[:, val_eids],   num_nodes=num_nodes)
        self.test_pos_g  = Data(edge_index=g.edge_index[:, test_eids],  num_nodes=num_nodes)

        label_key = next((k for k in ["edge_label", "label", "relationship", "Relationship"]
                          if getattr(g, k, None) is not None), None)
        if label_key is not None:
            rel = getattr(g, label_key)
            for gg, eids in ((self.train_pos_g, train_eids), (self.val_pos_g, val_eids),
                             (self.test_pos_g, test_eids)):
                setattr(gg, label_key, rel[eids])

        # 5.- Subgrafos NEGATIVOS (muestreo uniforme de pares no conectados)
        # --------------------------
        n_neg = (train_eids.numel(), val_eids.numel(), test_eids.numel())
        need_total = sum(n_neg)
        neg_keys = torch.empty(0, dtype=torch.long)
        for _ in range(20):
            if neg_keys.numel() >= need_total:
                break
            m = 2 * (need_total - neg_keys.numel())
            cu = torch.randint(0, num_nodes, (m,), generator=gen)
            cv = torch.randint(0, num_nodes, (m,), generator=gen)
            ok = cu != cv
            cand = torch.minimum(cu, cv)[ok] * num_nodes + torch.maximum(cu, cv)[ok]
            cand = cand[~torch.isin(cand, uniq_links)]
            neg_keys = torch.unique(torch.cat([neg_keys, cand]))
        if neg_keys.numel() < need_total:
            raise RuntimeError(f"Solo se encontraron {neg_keys.numel()} negativos de {need_total}.")
        neg_keys = neg_keys[torch.randperm(neg_keys.numel(), generator=gen)[:need_total]]

        lo, hi = neg_keys // num_nodes, neg_keys % num_nodes
        flip = torch.rand(need_total, generator=gen) < 0.5   # u<v no debe delatar a los negativos
        neg_u = torch.where(flip, hi, lo)
        neg_v = torch.where(flip, lo, hi)

        a, b = n_neg[0], n_neg[0] + n_neg[1]
        self.train_neg_g = Data(edge_index=torch.stack([neg_u[:a],  neg_v[:a]]),  num_nodes=num_nodes)
        self.val_neg_g   = Data(edge_index=torch.stack([neg_u[a:b], neg_v[a:b]]), num_nodes=num_nodes)
        self.test_neg_g  = Data(edge_index=torch.stack([neg_u[b:],  neg_v[b:]]),  num_nodes=num_nodes)

        if getattr(g, "x", None) is not None:
            for gg in (self.train_g, self.train_pos_g, self.val_pos_g, self.test_pos_g,
                       self.train_neg_g, self.val_neg_g, self.test_neg_g):
                gg.x = g.x

        if self.debug:
            print(f"[split_link_pred] enlaces: train={n_train:,}  val={n_val:,}  "
                  f"test={num_links - n_train - n_val:,}")
            print(f"[split_link_pred] +pos train={train_eids.numel():,}  +pos val={val_eids.numel():,}  "
                  f"+pos test={test_eids.numel():,}")
            print(f"[split_link_pred] -neg train={a:,}  -neg val={b - a:,}  -neg test={need_total - b:,}")

    # TODO: rvisar
    def add_random_features(self, dim: int = 64,
                                std: float = 0.05,
                                seed: int | None = None,
                                mode: str = "minmax"  # opciones: "minmax", "zscore", "uniform"
                                ):
        """
        Crea/repone graph.x con ruido controlado y lo normaliza.

        • dim  : nº de columnas
        • std  : σ inicial del N(0,σ²)  (cuanto menor, menos dispersión)
        • mode :
            'zscore' → normaliza cada columna a media 0 y varianza 1
            'minmax' → normaliza cada columna al rango [0,1]
            'uniform' → genera valores directamente en [0,1] sin normalizar
        """
        if seed is not None:
            torch.manual_seed(seed)

        n = self.graph.num_nodes

        if mode == "zscore":
            x = torch.randn(n, dim) * std
            mu  = x.mean(dim=0, keepdim=True)
            sig = x.std(dim=0, keepdim=True).clamp_min(1e-6)
            x   = (x - mu) / sig

        elif mode == "minmax":
            x = torch.randn(n, dim) * std
            x_min = x.min(dim=0, keepdim=True).values
            x_max = x.max(dim=0, keepdim=True).values
            rng   = (x_max - x_min).clamp_min(1e-6)
            x     = (x - x_min) / rng

        elif mode == "uniform":
            x = torch.rand(n, dim)  # ya está entre [0, 1]

        else:
            raise ValueError("mode debe ser 'zscore', 'minmax' o 'uniform'")

        self.graph.x = x

        if self.debug:
            print(f"[add_random_features] feat ← ({n}, {dim})  |  mode={mode}")

    # TODO: usar lugo y  vr si cambia
    def remove_low_degree_nodes(self, degree: int = 1, iterations: int = 3):
        """Elimina iterativamente nodos con grado undirected <= degree.

        Como el grafo es bidireccional, el grado undirected se aproxima con
        in_degree (cada arista no dirigida aparece como una arista entrante).
        Se repite 'iterations' veces porque al eliminar nodos de grado bajo
        pueden quedar nuevos nodos que también cumplen la condición.

        Args:
            degree: umbral de grado (se eliminan nodos con grado <= degree)
            iterations: número de pasadas
        """
        g = self.graph
        # IDs originales de los nodos que sobreviven (equivale a g.ndata[dgl.NID])
        original_ids = torch.arange(g.num_nodes)

        for i in range(iterations):
            # in_degree ≈ grado undirected en grafo bidireccional
            deg = pyg_degree(g.edge_index[1], g.num_nodes, dtype=torch.long)
            nodes_to_remove = torch.where(deg <= degree)[0]

            if nodes_to_remove.numel() == 0:
                if self.debug:
                    print(f"[remove_low_degree iter {i+1}] No hay nodos con grado <= {degree}. Terminando.")
                break

            keep_nodes = torch.where(deg > degree)[0]
            new_ei, _, edge_mask = subgraph(keep_nodes, g.edge_index, relabel_nodes=True,
                                            num_nodes=g.num_nodes, return_edge_mask=True)
            new_g = Data(edge_index=new_ei, num_nodes=keep_nodes.numel())
            for k in _edge_keys(g):
                setattr(new_g, k, getattr(g, k)[edge_mask])
            for k in _node_keys(g):
                setattr(new_g, k, getattr(g, k)[keep_nodes])
            original_ids = original_ids[keep_nodes]
            g = new_g

            if self.debug:
                print(f"[remove_low_degree iter {i+1}] Eliminados {nodes_to_remove.numel()} nodos "
                      f"(grado <= {degree}) → quedan {g.num_nodes} nodos, {g.num_edges} aristas")

        self.graph = g

        # Actualizar mapeo asn → node_id usando los IDs originales
        if hasattr(self, 'asn_to_node_id') and self.asn_to_node_id is not None:
            orig_to_new  = {int(orig): new for new, orig in enumerate(original_ids.tolist())}
            self.asn_to_node_id = {
                asn: orig_to_new[orig_nid]
                for asn, orig_nid in self.asn_to_node_id.items()
                if orig_nid in orig_to_new
            }

        if self.debug:
            if getattr(self.graph, 'edge_label', None) is not None:
                c = Counter(self.graph.edge_label.tolist())
                print(f"[remove_low_degree] Distribución final de etiquetas: {dict(c)}")
