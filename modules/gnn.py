import torch
import numpy as np
from collections import defaultdict
import random
import bz2, gzip
from collections import Counter, defaultdict
from tqdm import tqdm
import random
import torch
import numpy as np
from collections import defaultdict, Counter

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import degree as pyg_degree, subgraph


TOR_LABELS_DICT = {'P2P':0, 'C2P': 1,'P2C': 2}

class GNN:
    def __init__(self, debug=False):
        self.debug = debug
        self.graph = None

        # Atributos opcionales para almacenar edge IDs después de split
        # (se asignan en split_edges_classification si store_eids=True)
        self.train_eids = None
        self.val_eids = None
        self.test_eids = None

import pandas as pd
import torch


def _edge_ids(edge_index: torch.Tensor, num_nodes: int, u: torch.Tensor, v: torch.Tensor):
    """Devuelve, para cada par (u[i], v[i]), el id de la arista u->v o -1 si no existe.

    Reemplaza a `g.has_edges_between` / `g.edge_ids` de DGL. Si hay aristas
    duplicadas se devuelve la primera aparición.
    """
    key_all = edge_index[0].long() * num_nodes + edge_index[1].long()
    order = torch.argsort(key_all, stable=True)
    key_sorted = key_all[order]
    key_q = u.long() * num_nodes + v.long()
    pos = torch.searchsorted(key_sorted, key_q)
    pos_c = pos.clamp(max=key_sorted.numel() - 1)
    found = (pos < key_sorted.numel()) & (key_sorted[pos_c] == key_q)
    eids = torch.full_like(key_q, -1)
    eids[found] = order[pos_c[found]]
    return eids


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

        # Guardar mapeo asn → node_id para poder usar _fill_labels_from_caida_stream_fast
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

    def _fill_labels_from_caida_stream_fast(self, caida_file: str):
        """Etiqueta aristas existentes con relaciones CAIDA y agrega las que faltan.

        Para aristas CAIDA que no están en el grafo:
          - Si ambos nodos ya existen: agrega solo la arista.
          - Si algún ASN no existe en el grafo: crea el nodo con features=0
            y luego agrega la arista.

        Requiere que load_dataset() haya guardado self.asn_to_node_id.
        """
        if not hasattr(self, 'asn_to_node_id') or self.asn_to_node_id is None:
            raise RuntimeError(
                "asn_to_node_id no disponible. "
                "Llama a load_dataset() con nodes_csv que contenga columna 'asn'."
            )

        g = self.graph

        # Copia mutable del mapeo asn → node_id
        asn_to_nid = dict(self.asn_to_node_id)
        next_nid   = g.num_nodes
        feat_dim      = (g.x.shape[1]
                         if getattr(g, 'x', None) is not None else 0)
        edge_feat_dim = (g.edge_attr.shape[1]
                         if getattr(g, 'edge_attr', None) is not None else 0)

        # 1.- Inicializar labels si no existen (−1 = sin etiquetar)
        if getattr(g, 'edge_label', None) is None:
            g.edge_label = torch.full(
                (g.num_edges,), -1, dtype=torch.long
            )

        # 2.- Diccionario rápido (node_id_u, node_id_v) → eid
        u_all, v_all = g.edge_index[0].tolist(), g.edge_index[1].tolist()
        eid_map = {(u_all[i], v_all[i]): i
                   for i in range(g.num_edges)}

        # Buffers para elementos nuevos
        buffer_src, buffer_dst, buffer_lbl = [], [], []
        new_node_asns: list[int] = []   # ASNs que necesitan nodo nuevo

        # 3.- Elegir opener según extensión
        opener = (bz2.open  if caida_file.endswith(".bz2") else
                  gzip.open if caida_file.endswith(".gz")  else
                  open)

        # 4.- Recorrer archivo CAIDA
        with opener(caida_file, "rt") as f:
            for line in tqdm(f, desc="Etiquetando CAIDA"):
                if line.startswith("#") or not line.strip():
                    continue

                src_asn, dst_asn, rel = line.strip().split("|")
                src_asn, dst_asn = int(src_asn), int(dst_asn)

                if rel == "0":          # P2P simétrico
                    pares     = [(src_asn, dst_asn), (dst_asn, src_asn)]
                    etiquetas = [0, 0]
                else:                   # -1: provider→customer / customer→provider
                    pares     = [(src_asn, dst_asn), (dst_asn, src_asn)]
                    etiquetas = [2, 1]  # P2C=2, C2P=1

                for (u_asn, v_asn), lbl in zip(pares, etiquetas):
                    # Convertir ASN → node_id, creando nodo nuevo si es necesario
                    if u_asn not in asn_to_nid:
                        asn_to_nid[u_asn] = next_nid
                        new_node_asns.append(u_asn)
                        next_nid += 1
                    if v_asn not in asn_to_nid:
                        asn_to_nid[v_asn] = next_nid
                        new_node_asns.append(v_asn)
                        next_nid += 1

                    u_nid = asn_to_nid[u_asn]
                    v_nid = asn_to_nid[v_asn]

                    eid = eid_map.get((u_nid, v_nid))
                    if eid is not None:
                        # Arista ya existe → actualizar etiqueta
                        g.edge_label[eid] = lbl
                    else:
                        # Arista no existe → agregar al buffer
                        buffer_src.append(u_nid)
                        buffer_dst.append(v_nid)
                        buffer_lbl.append(lbl)

        # 5.- Agregar nodos nuevos (ASNs que no estaban en el grafo)
        if new_node_asns:
            n_new = len(new_node_asns)
            if feat_dim > 0:
                new_feat = torch.zeros(n_new, feat_dim, dtype=torch.float32)
                g.x = torch.cat([g.x, new_feat], dim=0)
            g.num_nodes = g.num_nodes + n_new
            if self.debug:
                print(f"[CAIDA] Añadidos {n_new} nodos nuevos (ASNs sin info de PeeringDB → feat=0)")

        # 6.- Agregar aristas nuevas
        if buffer_src:
            n_new_edges = len(buffer_src)
            new_ei = torch.stack([torch.tensor(buffer_src, dtype=torch.long),
                                  torch.tensor(buffer_dst, dtype=torch.long)])
            g.edge_index = torch.cat([g.edge_index, new_ei], dim=1)
            g.edge_label = torch.cat([g.edge_label, torch.tensor(buffer_lbl, dtype=torch.long)])
            if edge_feat_dim > 0:
                g.edge_attr = torch.cat(
                    [g.edge_attr, torch.zeros(n_new_edges, edge_feat_dim, dtype=torch.float32)], dim=0)
            if self.debug:
                print(f"[CAIDA] Añadidas {len(buffer_src)} aristas nuevas al grafo")

        # 7.- Actualizar mapeo y resumen
        self.asn_to_node_id = asn_to_nid

        if self.debug:
            c = Counter(g.edge_label.tolist())
            print(f"[CAIDA] Conteo final de etiquetas 0/1/2/−1 → {c}")


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


    def split_graph_nodes(self, train_size=0.8):
        num_nodes = self.graph.num_nodes

        # Índices aleatorios de nodos
        node_ids = torch.randperm(num_nodes)

        num_train = int(train_size * num_nodes)
        num_test = num_nodes - num_train

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[node_ids[:num_train]] = True
        test_mask[node_ids[num_train:]] = True

        # Guardar en el grafo (prefijo node_ para distinguirlas de las máscaras de arista)
        self.graph.node_train_mask = train_mask
        self.graph.node_test_mask = test_mask

        if self.debug:
            print(f"Train nodes: {train_mask.sum().item()}, Test nodes: {test_mask.sum().item()}")


    def split_edges_link_prediction(self, train_ratio: float = 0.8, seed: int = 42):

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch_geometric.seed_everything(seed)  # asegura consistencia en operaciones internas de PyG

        g = self.graph
        num_edges  = g.num_edges
        num_nodes  = g.num_nodes
        if num_edges == 0:
            raise ValueError("El grafo no contiene aristas.")

        rng = np.random.default_rng(seed)

        # 1.- Barajar eids y separar train / test
        # --------------------------
        eids = np.arange(num_edges)
        rng.shuffle(eids)
        split_idx   = int(num_edges * train_ratio)
        train_eids  = torch.as_tensor(eids[:split_idx],  dtype=torch.int64)
        test_eids   = torch.as_tensor(eids[split_idx:], dtype=torch.int64)

        # 2.- Grafo para el encoder (sin aristas de test)
        # --------------------------
        # IMPORTANTE:  mantenemos todas las aristas menos las POSITIVAS de test
        keep = torch.ones(num_edges, dtype=torch.bool)
        keep[test_eids] = False
        self.train_g = Data(edge_index=g.edge_index[:, keep], num_nodes=num_nodes)
        for k in _edge_keys(g):
            setattr(self.train_g, k, getattr(g, k)[keep])

        # 3.- Subgrafos POSITIVOS
        # --------------------------
        self.train_pos_g = Data(edge_index=g.edge_index[:, train_eids], num_nodes=num_nodes)
        self.test_pos_g  = Data(edge_index=g.edge_index[:, test_eids],  num_nodes=num_nodes)

        # Copiamos feats (si existen) a los subgrafos positivos
        if getattr(g, "x", None) is not None:
            for gg in (self.train_g, self.train_pos_g, self.test_pos_g):
                gg.x = g.x

        # Copiamos etiquetas (si existen) a POS
        if getattr(g, "Relationship", None) is not None:
            rel = g.Relationship
            self.train_pos_g.Relationship = rel[train_eids]
            self.test_pos_g.Relationship  = rel[test_eids]

        # 4.- Subgrafos NEGATIVOS (muestreo uniforme)
        # --------------------------
        def sample_negative(k: int):
            """Devuelve k pares (u,v) que NO existen en g."""
            # sobre-muestramos y filtramos para vectorizar
            max_trials = 3
            collected_u, collected_v = [], []
            while len(collected_u) < k and max_trials:
                need   = k - len(collected_u)
                cand_u = torch.randint(0, num_nodes, (need * 2,))
                cand_v = torch.randint(0, num_nodes, (need * 2,))
                mask   = _edge_ids(g.edge_index, num_nodes, cand_u, cand_v) < 0
                cand_u, cand_v = cand_u[mask], cand_v[mask]
                collected_u.extend(cand_u[:need].tolist())
                collected_v.extend(cand_v[:need].tolist())
                max_trials -= 1
            return torch.as_tensor(collected_u), torch.as_tensor(collected_v)

        train_neg_u, train_neg_v = sample_negative(len(train_eids))
        test_neg_u,  test_neg_v  = sample_negative(len(test_eids))

        self.train_neg_g = Data(edge_index=torch.stack([train_neg_u, train_neg_v]), num_nodes=num_nodes)
        self.test_neg_g  = Data(edge_index=torch.stack([test_neg_u,  test_neg_v]),  num_nodes=num_nodes)

        if getattr(g, "x", None) is not None:
            for gg in (self.train_neg_g, self.test_neg_g):
                gg.x = g.x

        if self.debug:
            print(f"[split_basic] +pos train={len(train_eids)}  +pos test={len(test_eids)}")
            print(f"[split_basic] -neg train={self.train_neg_g.num_edges} "
                f"-neg test={self.test_neg_g.num_edges}")

    def split_edges_classification_leaky(self, train_size=0.8, seed=0):
        """
        Crea dos máscaras booleanas en el grafo:
            • 'train_mask' : aristas usadas para entrenar
            • 'test_mask'  : aristas usadas para evaluar
        NO se agrupan las direcciones opuestas ⇒ posible fuga de información.

        Args
        ----
        train_ratio : float   proporción de aristas etiquetadas 0/1/2 que van a train
        seed        : int     semilla de reproducibilidad
        """


        rng = random.Random(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        rel = self.graph.Relationship      # 0/1/2/-1
        is_lbl = rel >= 0                               # sólo etiquetadas

        all_eids = torch.nonzero(is_lbl, as_tuple=False).squeeze()   # tensor 1-D
        n_total  = len(all_eids)
        n_train  = int(n_total * train_size)

        # barajamos y partimos
        idx = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))
        train_eids = all_eids[idx[:n_train]]
        test_eids  = all_eids[idx[n_train:]]

        # máscaras
        num_e = self.graph.num_edges
        train_mask = torch.zeros(num_e, dtype=torch.bool)
        test_mask  = torch.zeros(num_e, dtype=torch.bool)

        train_mask[train_eids] = True
        test_mask[test_eids]   = True
        # las aristas −1 quedan con ambas máscaras a False

        self.graph.train_mask = train_mask
        self.graph.test_mask  = test_mask

        # ── resumen rápido ────────────────────────────────────────────
        cnt_tr = Counter(rel[train_mask].tolist())
        cnt_te = Counter(rel[test_mask].tolist())
        print(f"[split_leaky] train={train_mask.sum().item():,}  "
            f"test={test_mask.sum().item():,}")
        print(f"  clases train 0/1/2 → {dict(cnt_tr)}")
        print(f"  clases test  0/1/2 → {dict(cnt_te)}")


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

    def split_edges_classification_v0(self, train_size=0.7, val_size=0.15, seed=0,
                                return_eids=False, store_eids=True):
        """
        Split no-leak para edge-classification con train/val/test.

        • train_size: fracción para entrenamiento (default 0.7 = 70%)
        • val_size: fracción para validación (default 0.15 = 15%)
        • test_size = 1 - train_size - val_size (default 0.15 = 15%)
        • Crea graph.train_mask, val_mask y test_mask.
        • Devuelve (train_eids, val_eids, test_eids) si `return_eids=True`.
        • Opcionalmente los guarda como atributos (para reutilizarlos).
        """


        rng = random.Random(seed)
        torch.manual_seed(seed); np.random.seed(seed)

        u, v  = self.graph.edge_index[0], self.graph.edge_index[1]
        rel   = self.graph.Relationship
        is_lbl = rel >= 0

        # 1.- Agrupar dos direcciones
        # --------------------------
        pair2eids = defaultdict(list)
        for eid, (ui, vi) in enumerate(zip(u.tolist(), v.tolist())):
            if is_lbl[eid]:
                pair2eids[(min(ui, vi), max(ui, vi))].append(eid)

        pairs = list(pair2eids.keys());   rng.shuffle(pairs)

        # Split train/val/test
        n_train = int(len(pairs) * train_size)
        n_val = int(len(pairs) * val_size)

        train_pairs = pairs[:n_train]
        val_pairs   = pairs[n_train:n_train + n_val]
        test_pairs  = pairs[n_train + n_val:]

        gather = lambda subset: [eid for p in subset for eid in pair2eids[p]]
        train_eids = torch.tensor(gather(train_pairs), dtype=torch.int64)
        val_eids   = torch.tensor(gather(val_pairs),   dtype=torch.int64)
        test_eids  = torch.tensor(gather(test_pairs),  dtype=torch.int64)

        # 2.- Máscaras booleanas
        # --------------------------
        num_e = self.graph.num_edges
        train_mask = torch.zeros(num_e, dtype=torch.bool)
        val_mask   = torch.zeros(num_e, dtype=torch.bool)
        test_mask  = torch.zeros(num_e, dtype=torch.bool)

        train_mask[train_eids] = True
        val_mask[val_eids]     = True
        test_mask[test_eids]   = True

        self.graph.train_mask = train_mask
        self.graph.val_mask   = val_mask
        self.graph.test_mask  = test_mask

        # 3.- Opcional: guardo para sampling
        # --------------------------
        if store_eids:
            self.train_eids = train_eids
            self.val_eids   = val_eids
            self.test_eids  = test_eids

        if self.debug:
            print(f"[split] train={train_mask.sum()}  val={val_mask.sum()}  test={test_mask.sum()}")
            print("  clases train:", dict(Counter(rel[train_mask].tolist())))
            print("  clases val:",   dict(Counter(rel[val_mask].tolist())))

        if return_eids:
            return (train_eids, val_eids, test_eids)
        return None
