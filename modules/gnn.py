import dgl
import torch
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
import random
import gzip, torch, dgl, numpy as np
import bz2, gzip, dgl, torch, re
from collections import Counter     # ← agrégalo al principio de modules/gnn.py
import torch, gzip, bz2, re
from collections import Counter
import dgl
import os, dgl, torch
from collections import Counter
import bz2, gzip, re, torch
from tqdm import tqdm




TOR_LABELS_DICT = {'P2P':0, 'C2P': 1,'P2C': 2}

class GNN:
    def __init__(self, debug=False):
        self.debug = debug
        self.graph = None

        # Grafo DGL que contiene los training edges
        self.train_graph = None

        # Grafo DGL que contiene los test edges
        self.test_graph = None

        self.nx_graph = None
        self.dgl_graph = None

    def load_dataset(self,
                     data_path: str,
                     index_graph: int = 0,
                     caida_path: str | None = None,
                     force_reload: bool = False,
                     simplify: bool = False):
            
            # 1. Cargar CSV ----------------------------------------------------------
            if self.debug:
                    print("[load_dataset] Leyendo CSV …")
            self.dgl_graph = dgl.data.CSVDataset(data_path, force_reload=force_reload)[index_graph]
            
            if self.debug:
                    print(f"[load_dataset] Grafo {self.dgl_graph}")

            # 2. Simplificar grafo si se pide ----------------------------------------
            if simplify:
                if self.debug:
                    print("[load_dataset] Simplificando  …")
                self.dgl_graph = dgl.to_simple(self.dgl_graph, return_counts=False)

            # 3. Asignar atributo de aristas 'Relationship' como -1 por defecto ------
            num_e = self.dgl_graph.num_edges()
            self.dgl_graph.edata["Relationship"] = torch.full( (num_e,), -1, dtype=torch.int8)

            # 4. Etiquetar desde archivo CAIDA si se entrega --------------------------
            if caida_path is not None:
                if self.debug:
                    print(f"[load_dataset] Etiquetando CAIDA → {caida_path}")
                self._fill_labels_from_caida_stream_fast(caida_path)
            
            
            if self.debug:
                cnt = Counter(self.dgl_graph.edata["Relationship"].tolist())
                print(f"[load_dataset] etiquetas 0/1/2/-1 → {cnt}")
        
    def _fill_labels_from_caida_stream_fast(self, caida_file: str):


        # ---------- 1. Diccionario rápido: (u, v) -> eid -------------------
        u, v = self.dgl_graph.edges()
        eid_map = {(int(u[i]), int(v[i])): i           # clave   = (u,v)
                for i in range(self.dgl_graph.num_edges())}  # valor  = id de arista

        # Buffers para aristas nuevas que no estén en el grafo
        buffer_src, buffer_dst, buffer_lbl = [], [], []

        # ---------- 2. Elegir la función open según extensión --------------
        opener = (bz2.open  if caida_file.endswith(".bz2") else
                gzip.open if caida_file.endswith(".gz")  else
                open)

        # ---------- 3. Recorrer el archivo CAIDA ---------------------------
        with opener(caida_file, "rt") as f:
            for line in tqdm(f, desc="Etiquetando CAIDA"):
                if line.startswith("#") or not line.strip():
                    continue  # saltar comentarios y líneas vacías

                src, dst, rel = line.strip().split("|")
                src, dst = int(src), int(dst)

                # Definir pares y etiquetas según CAIDA
                if rel == "0":              # P2P  (simétrico)
                    pares     = [(src, dst), (dst, src)]
                    etiquetas = [0, 0]      # 0 = P2P
                else:                       # -1  (P2C / C2P)
                    pares     = [(src, dst), (dst, src)]
                    etiquetas = [2, 1]      # 2 = P2C, 1 = C2P

                # ---------- 4. Procesar cada dirección --------------------
                for (u_nodo, v_nodo), lbl in zip(pares, etiquetas):
                    eid = eid_map.get((u_nodo, v_nodo))
                    if eid is not None:
                        # Arista ya existe → solo etiquetar
                        self.dgl_graph.edata["Relationship"][eid] = lbl
                    else:
                        # Arista no existe → guardar para añadirla luego
                        buffer_src.append(u_nodo)
                        buffer_dst.append(v_nodo)
                        buffer_lbl.append(lbl)

        # ---------- 5. Añadir aristas nuevas (si hay) ----------------------
        if buffer_src:   # lista no vacía
            self.dgl_graph = dgl.add_edges(
                self.dgl_graph,
                torch.tensor(buffer_src),
                torch.tensor(buffer_dst),
                data={"Relationship": torch.tensor(buffer_lbl, dtype=torch.int8)}
            )
            if self.debug:
                print(f"[CAIDA] Añadidas {len(buffer_src)} aristas que faltaban")

        # ---------- 6. Resumen final (opcional) ---------------------------
        if self.debug:
            from collections import Counter
            c = Counter(self.dgl_graph.edata["Relationship"].tolist())
            print(f"[CAIDA] Conteo final de etiquetas 0/1/2/-1 → {c}")

    def split_edges_classification(self, train_size=0.8, seed=0,
                                return_eids=False, store_eids=True):
        """
        Split no-leak para edge-classification.

        • Crea edata['train_mask'] y ['test_mask'] (igual que antes).
        • Devuelve (train_eids, test_eids) si `return_eids=True`.
        • Opcionalmente los guarda como atributos (para reutilizarlos).
        """
        import random, torch, numpy as np
        from collections import defaultdict, Counter

        rng = random.Random(seed)
        torch.manual_seed(seed); np.random.seed(seed)

        u, v  = self.dgl_graph.edges()
        rel   = self.dgl_graph.edata["Relationship"]
        is_lbl = rel >= 0

        # 1️⃣  agrupar dos direcciones ────────────────────────────────
        pair2eids = defaultdict(list)
        for eid, (ui, vi) in enumerate(zip(u.tolist(), v.tolist())):
            if is_lbl[eid]:
                pair2eids[(min(ui, vi), max(ui, vi))].append(eid)

        pairs = list(pair2eids.keys());   rng.shuffle(pairs)
        n_train = int(len(pairs) * train_size)

        train_pairs = pairs[:n_train]
        test_pairs  = pairs[n_train:]

        gather = lambda subset: [eid for p in subset for eid in pair2eids[p]]
        train_eids = torch.tensor(gather(train_pairs), dtype=torch.int64)
        test_eids  = torch.tensor(gather(test_pairs),  dtype=torch.int64)

        # 2️⃣  máscaras booleanas ─────────────────────────────────────
        num_e = self.dgl_graph.num_edges()
        train_mask = torch.zeros(num_e, dtype=torch.bool)
        test_mask  = torch.zeros_like(train_mask)
        train_mask[train_eids] = True
        test_mask[test_eids]   = True

        self.dgl_graph.edata["train_mask"] = train_mask
        self.dgl_graph.edata["test_mask"]  = test_mask

        # 3️⃣  opcional: guardo para sampling -------------------------
        if store_eids:
            self.train_eids = train_eids
            self.test_eids  = test_eids

        if self.debug:
            from collections import Counter
            print("[split] train={}  test={}".format(train_mask.sum(), test_mask.sum()))
            print("  clases train:", dict(Counter(rel[train_mask].tolist())))

        return (train_eids, test_eids) if return_eids else None


    def split_graph_nodes(self, train_size=0.8):
        num_nodes = self.dgl_graph.num_nodes()

        # Índices aleatorios de nodos
        node_ids = torch.randperm(num_nodes)

        num_train = int(train_size * num_nodes)
        num_test = num_nodes - num_train

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[node_ids[:num_train]] = True
        test_mask[node_ids[num_train:]] = True

        # Guardar en el grafo
        self.dgl_graph.ndata['train_mask'] = train_mask
        self.dgl_graph.ndata['test_mask'] = test_mask

        if self.debug:
            print(f"Train nodes: {train_mask.sum().item()}, Test nodes: {test_mask.sum().item()}")

    import torch, numpy as np, dgl

    def split_edges_link_prediction(self, train_ratio: float = 0.8, seed: int = 42):
        
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        dgl.random.seed(seed)  # asegura consistencia en operaciones internas de DGL
        
        g = self.dgl_graph
        num_edges  = g.num_edges()
        num_nodes  = g.num_nodes()
        if num_edges == 0:
            raise ValueError("El grafo no contiene aristas.")

        rng = np.random.default_rng(seed)

        # ---------- 1. barajar eids y separar train / test ------------------
        eids = np.arange(num_edges)
        rng.shuffle(eids)
        split_idx   = int(num_edges * train_ratio)
        train_eids  = torch.as_tensor(eids[:split_idx],  dtype=torch.int64)
        test_eids   = torch.as_tensor(eids[split_idx:], dtype=torch.int64)

        # ---------- 2. grafo para el encoder (sin aristas de test) ----------
        # IMPORTANTE:  mantenemos todas las aristas menos las POSITIVAS de test
        self.train_g = dgl.remove_edges(g, test_eids)

        # ---------- 3. subgrafos POSITIVOS ----------------------------------
        self.train_pos_g = dgl.edge_subgraph(g, train_eids, relabel_nodes=False)
        self.test_pos_g  = dgl.edge_subgraph(g, test_eids,  relabel_nodes=False)

        # Copiamos feats (si existen) a los subgrafos positivos
        if "feat" in g.ndata:
            for gg in (self.train_pos_g, self.test_pos_g):
                gg.ndata["feat"] = g.ndata["feat"]

        # Copiamos etiquetas (si existen) a POS
        if "Relationship" in g.edata:
            rel = g.edata["Relationship"]
            self.train_pos_g.edata["Relationship"] = rel[train_eids]
            self.test_pos_g.edata["Relationship"]  = rel[test_eids]

        # ---------- 4. subgrafos NEGATIVOS (muestreo uniforme) ---------------
        def sample_negative(k: int):
            """Devuelve k pares (u,v) que NO existen en g."""
            # sobre-muestramos y filtramos para vectorizar
            max_trials = 3
            collected_u, collected_v = [], []
            while len(collected_u) < k and max_trials:
                need   = k - len(collected_u)
                cand_u = torch.randint(0, num_nodes, (need * 2,))
                cand_v = torch.randint(0, num_nodes, (need * 2,))
                mask   = ~g.has_edges_between(cand_u, cand_v)
                cand_u, cand_v = cand_u[mask], cand_v[mask]
                collected_u.extend(cand_u[:need].tolist())
                collected_v.extend(cand_v[:need].tolist())
                max_trials -= 1
            return torch.as_tensor(collected_u), torch.as_tensor(collected_v)

        train_neg_u, train_neg_v = sample_negative(len(train_eids))
        test_neg_u,  test_neg_v  = sample_negative(len(test_eids))

        self.train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=num_nodes)
        self.test_neg_g  = dgl.graph((test_neg_u,  test_neg_v),  num_nodes=num_nodes)

        if "feat" in g.ndata:
            for gg in (self.train_neg_g, self.test_neg_g):
                gg.ndata["feat"] = g.ndata["feat"]

        if self.debug:
            print(f"[split_basic] +pos train={len(train_eids)}  +pos test={len(test_eids)}")
            print(f"[split_basic] -neg train={self.train_neg_g.num_edges()} "
                f"-neg test={self.test_neg_g.num_edges()}")
            
    def split_edges_classification_leaky(self, train_size=0.8, seed=0):
        """
        Crea dos máscaras booleanas en edata:
            • 'train_mask' : aristas usadas para entrenar
            • 'test_mask'  : aristas usadas para evaluar
        NO se agrupan las direcciones opuestas ⇒ posible fuga de información.

        Args
        ----
        train_ratio : float   proporción de aristas etiquetadas 0/1/2 que van a train
        seed        : int     semilla de reproducibilidad
        """
        import torch, numpy as np, random
        from collections import Counter

        rng = random.Random(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        rel = self.dgl_graph.edata["Relationship"]      # 0/1/2/-1
        is_lbl = rel >= 0                               # sólo etiquetadas

        all_eids = torch.nonzero(is_lbl, as_tuple=False).squeeze()   # tensor 1-D
        n_total  = len(all_eids)
        n_train  = int(n_total * train_size)

        # barajamos y partimos
        idx = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))
        train_eids = all_eids[idx[:n_train]]
        test_eids  = all_eids[idx[n_train:]]

        # máscaras
        num_e = self.dgl_graph.num_edges()
        train_mask = torch.zeros(num_e, dtype=torch.bool)
        test_mask  = torch.zeros(num_e, dtype=torch.bool)

        train_mask[train_eids] = True
        test_mask[test_eids]   = True
        # las aristas −1 quedan con ambas máscaras a False

        self.dgl_graph.edata["train_mask"] = train_mask
        self.dgl_graph.edata["test_mask"]  = test_mask

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
        Crea/repone ndata['feat'] con ruido controlado y lo normaliza.

        • dim  : nº de columnas
        • std  : σ inicial del N(0,σ²)  (cuanto menor, menos dispersión)
        • mode : 
            'zscore' → normaliza cada columna a media 0 y varianza 1
            'minmax' → normaliza cada columna al rango [0,1]
            'uniform' → genera valores directamente en [0,1] sin normalizar
        """
        if seed is not None:
            torch.manual_seed(seed)

        n = self.dgl_graph.num_nodes()

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

        self.dgl_graph.ndata['feat'] = x

        if self.debug:
            print(f"[add_random_features] feat ← ({n}, {dim})  |  mode={mode}")



    # def split_graph_edges(self, train_ratio=0.8):
    #     u, v = self.dgl_graph.edges()
    #     num_edges = self.dgl_graph.num_edges()
    #     relationship = self.dgl_graph.edata["Relationship"]

    #     # Agrupar índices por clase
    #     class_to_eids = defaultdict(list)
    #     for i, rel in enumerate(relationship.tolist()):
    #         class_to_eids[rel].append(i)

    #     train_eids, test_eids = [], []

    #     # Split estratificado por clase
    #     for rel_class, eids in class_to_eids.items():
    #         eids = np.array(eids)
    #         np.random.shuffle(eids)
    #         split_point = int(len(eids) * train_ratio)
    #         train_eids.extend(eids[:split_point])
    #         test_eids.extend(eids[split_point:])

    #     train_eids = np.array(train_eids)
    #     test_eids = np.array(test_eids)

    #     train_pos_u, train_pos_v = u[train_eids], v[train_eids]
    #     test_pos_u, test_pos_v = u[test_eids], v[test_eids]

    #     # Negativos
    #     neg_u, neg_v = self.get_negative_edges(num_edges)
    #     train_neg_u, train_neg_v = neg_u[:len(train_eids)], neg_v[:len(train_eids)]
    #     test_neg_u, test_neg_v = neg_u[len(train_eids):], neg_v[len(train_eids):]

    #     # Crear grafos
    #     self.train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=self.dgl_graph.num_nodes())
    #     self.test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=self.dgl_graph.num_nodes())
    #     self.train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=self.dgl_graph.num_nodes())
    #     self.test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=self.dgl_graph.num_nodes())

    #     if "feat" in self.dgl_graph.ndata:
    #         for g in [self.train_pos_g, self.test_pos_g, self.train_neg_g, self.test_neg_g]:
    #             g.ndata["feat"] = self.dgl_graph.ndata["feat"]

    #     # Copiar labels a los subgrafos positivos
    #     self.train_pos_g.edata["Relationship"] = relationship[torch.tensor(train_eids)]
    #     self.test_pos_g.edata["Relationship"] = relationship[torch.tensor(test_eids)]
    
    # def split_graph_edges_basic_full_graph(self, train_size=0.8):

    #     if self.debug:
    #         print("Dividiendo edges del grafo...")

    # def split_graph_edges_basic(self, train_size=0.8):
                
    #     u,v = self.dgl_graph.edges()

    #     # IDs de lo edges
    #     eids = np.arange(self.dgl_graph.num_edges()) 
    #     # Shuffle the edges
    #     eids = np.random.permutation(eids)

    #     # Tamaño de train y test
    #     test_size = int(len(eids) * train_size)
    #     train_size = self.dgl_graph.num_edges() - test_size 

    #     # Selecciona los edges de test y train
    #     test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    #     train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    #     # Matriz de adyacencia
    #     adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))

    #     neg_u, neg_v = self.get_negative_edges( self.dgl_graph.num_edges())
    #     #  167.564 -> 30 seg  -> 
    #     #  334.996 -> 50 seg -> 100.000 paths
    #     #1.676.722 -> 4 min 17 seg -> 500.000 paths

    #     test_neg_u, test_neg_v = (
    #         neg_u[:test_size],
    #         neg_v[:test_size],
    #     )

    #     train_neg_u, train_neg_v = (
    #         neg_u[test_size:],
    #         neg_v[test_size:],
    #     )

    #     # Eliminar edges de test
    #     self.train_g = dgl.remove_edges(self.dgl_graph, eids[:test_size])

    #     self.train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=self.dgl_graph.num_nodes())
    #     self.train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=self.dgl_graph.num_nodes())    

    #     self.test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=self.dgl_graph.num_nodes())
    #     self.test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=self.dgl_graph.num_nodes())

    #     if "feat" in self.dgl_graph.ndata:
    #         self.train_pos_g.ndata["feat"] = self.dgl_graph.ndata["feat"]
    #         self.train_neg_g.ndata["feat"] = self.dgl_graph.ndata["feat"]
    #         self.test_pos_g.ndata["feat"] = self.dgl_graph.ndata["feat"]
    #         self.test_neg_g.ndata["feat"] = self.dgl_graph.ndata["feat"]
        
    #     # Copiar edata["Relationship"] si existe
    #     if "Relationship" in self.dgl_graph.edata:
    #         test_eids = torch.tensor(eids[:test_size])
    #         train_eids = torch.tensor(eids[test_size:])
    #         relationship = self.dgl_graph.edata["Relationship"]
    #         self.train_pos_g.edata["Relationship"] = relationship[train_eids]
    #         self.test_pos_g.edata["Relationship"] = relationship[test_eids]
    
    
    
    # def split_graph_nodes(self, train_size=0.8):

    #     num_nodes = self.dgl_graph.num_nodes()
    #     nodes_ids = np.arange(num_nodes)
    #     nodes = self.dgl_graph.nodes()
    #     # Shuffle the nodes
    #     nodes_ids = np.random.permutation(nodes_ids)

    #     # Tamaño de train y test
    #     train_size = int(num_nodes * train_size)
    #     test_size = num_nodes - train_size 

    #     # Split
    #     train_nodes = nodes[nodes_ids[:train_size]]
    #     test_nodes = nodes[nodes_ids[train_size:]]

    #     # Crear máscaras
    #     train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    #     test_mask = torch.zeros(num_nodes, dtype=torch.bool)


    #     train_mask[train_nodes] = True
    #     test_mask[test_nodes] = True

    #     # Asignar al grafo
    #     self.dgl_graph.train_mask = train_mask
    #     self.dgl_graph.test_mask = test_mask

    
    # def get_negative_edges(self, num_neg_samples): #FIXME: optimizar
    #     """
    #     Genera aristas negativas para el grafo dado.
    #     """
    #     print(f"Generando {num_neg_samples} aristas negativas...")
    #     neg_src_u = []
    #     neg_dst_v = []
    #     num_nodes = self.dgl_graph.num_nodes()

    #     for i in range(num_neg_samples):
    #         src = np.random.randint(0, num_nodes)
    #         dst = np.random.randint(0, num_nodes)
    #         while self.dgl_graph.has_edges_between(src, dst):
    #             src = np.random.randint(0, num_nodes)
    #             dst = np.random.randint(0, num_nodes)
    #         neg_src_u.append(src)
    #         neg_dst_v.append(dst)

    #     print(f"Aristas negativas generadas: {i}")

    #     return torch.tensor(neg_src_u), torch.tensor(neg_dst_v)
    # # ----------------------------------------------------------------------
    # def split_edges_keep_full_graph(self, train_ratio=0.8, val_ratio=0.1, seed=0):
    #     """
    #     • self.full_g        → grafo ORIGINAL, sin quitar aristas (para el encoder)
    #     • self.train_pos_g   → aristas etiquetadas (0/1/2) de entrenamiento
    #     • self.val_pos_g     → aristas etiquetadas de validación
    #     • self.test_pos_g    → aristas etiquetadas de test

    #     Las aristas –1 quedan solo en full_g; no aparecen en *_pos_g, por lo que
    #     no influyen ni en la loss ni en las métricas.
    #     """
    #     import random, torch, numpy as np
    #     from collections import Counter
    #     rng = random.Random(seed)
    #     torch.manual_seed(seed)
    #     np.random.seed(seed)

    #     # 0)  guardamos el grafo completo para el encoder
    #     self.full_g = self.dgl_graph                     # alias corto

    #     # 1)  agrupamos direcciones opuestas ----------------------------
    #     u, v = self.dgl_graph.edges()
    #     is_lbl = self.dgl_graph.edata['Relationship'] >= 0        # solo 0/1/2

    #     pairs = {}                                                # (min,max) -> [eid]
    #     for eid, (ui, vi) in enumerate(zip(u.tolist(), v.tolist())):
    #         key = (min(ui, vi), max(ui, vi))
    #         pairs.setdefault(key, []).append(eid)

    #     keys = list(pairs.keys())
    #     rng.shuffle(keys)

    #     n_pairs = len(keys)
    #     n_train = int(n_pairs * train_ratio)
    #     n_val   = int(n_pairs * val_ratio)

    #     split = {
    #         "train": keys[:n_train],
    #         "val":   keys[n_train:n_train + n_val],
    #         "test":  keys[n_train + n_val:],
    #     }

    #     def eids_for(keylist):
    #         return torch.tensor(
    #             [eid for k in keylist for eid in pairs[k] if is_lbl[eid]],
    #             dtype=torch.int64)

    #     train_eids = eids_for(split["train"])
    #     val_eids   = eids_for(split["val"])
    #     test_eids  = eids_for(split["test"])

    #     # 2)  helper para subgrafo positivo ----------------------------
    #     def build_pos(eids):
    #         g_pos = self.dgl_graph.edge_subgraph(eids, relabel_nodes=False)
    #         for n, d in self.dgl_graph.ndata.items():
    #             g_pos.ndata[n] = d
    #         return g_pos

    #     self.train_pos_g = build_pos(train_eids)
    #     self.val_pos_g   = build_pos(val_eids)
    #     self.test_pos_g  = build_pos(test_eids)

    #     # 3)  resumen de depuración ------------------------------------
    #     if self.debug:
    #         c = Counter(self.train_pos_g.edata["Relationship"].tolist())
    #         print(f"[split] train={len(train_eids)}  val={len(val_eids)}  "
    #             f"test={len(test_eids)}  clases={dict(c)}")

    # def split_dataset_v0(self):
    #     """
    #     Divide los edges en tres conjuntos: training, validación y test.
    #     #FIXME: el edge (1,2) es equivalente a (2,1)), es importante asegurarse de que ambos casos se asignen al mismo conjunto (entrenamiento, validación o prueba). La implementación actual, sin embargo, no garantiza esto.
    #     """
    #     number_of_edges = self.dgl_graph.num_edges()
    #     u, v = self.dgl_graph.edges()

    #     # Ordenar los nodos en cada edge
    #     edges = torch.stack([torch.min(u, v), torch.max(u, v)], dim=1)
        
    #     # Crear un diccionario para almacenar las asignaciones de los edges
    #     edge_dict = {}
        
    #     # Generar las máscaras
    #     train_mask = torch.zeros(number_of_edges, dtype=torch.bool)
    #     val_mask = torch.zeros(number_of_edges, dtype=torch.bool)
    #     test_mask = torch.zeros(number_of_edges, dtype=torch.bool)
        
    #     for i in range(number_of_edges):
    #         edge = tuple(edges[i].tolist())
    #         if edge not in edge_dict:
    #             # Asignar el edge a un conjunto basado en una muestra aleatoria
    #             rand_value = torch.rand(1).item()
    #             if rand_value < 0.6:
    #                 edge_dict[edge] = 'train'
    #                 train_mask[i] = True
    #             elif rand_value < 0.8:
    #                 edge_dict[edge] = 'val'
    #                 val_mask[i] = True
    #             else:
    #                 edge_dict[edge] = 'test'
    #                 test_mask[i] = True
    #         else:
    #             # Asignar el edge al mismo conjunto que su par
    #             if edge_dict[edge] == 'train':
    #                 train_mask[i] = True
    #             elif edge_dict[edge] == 'val':
    #                 val_mask[i] = True
    #             else:
    #                 test_mask[i] = True

    #     self.train_mask = train_mask
    #     self.val_mask = val_mask
    #     self.test_mask = test_mask

    #     if self.debug:
    #         print(f"Training edges: {self.train_mask.sum().item()}")
    #         print(f"Validation edges: {self.val_mask.sum().item()}")
    #         print(f"Test edges: {self.test_mask.sum().item()}")
    
    # def split_dataset_v00(self):

    #     number_of_edges = self.dgl_graph.num_edges()
    #     u, v = self.dgl_graph.edges()

    #     # Creamos un identificador canónico para cada par (ignora dirección)
    #     canonical_edges = [tuple(sorted((u[i].item(), v[i].item()))) for i in range(number_of_edges)]

    #     # Asignamos cada par a un conjunto solo una vez
    #     edge_to_set = dict()
    #     for edge in set(canonical_edges):
    #         rand_value = torch.rand(1).item()
    #         if rand_value < 0.6:
    #             edge_to_set[edge] = 'train'
    #         elif rand_value < 0.8:
    #             edge_to_set[edge] = 'val'
    #         else:
    #             edge_to_set[edge] = 'test'

    #     # Creamos las máscaras
    #     train_mask = torch.zeros(number_of_edges, dtype=torch.bool)
    #     val_mask = torch.zeros(number_of_edges, dtype=torch.bool)
    #     test_mask = torch.zeros(number_of_edges, dtype=torch.bool)

    #     # Asignamos según el conjunto de cada par
    #     for i, edge in enumerate(canonical_edges):
    #         set_name = edge_to_set[edge]
    #         if set_name == 'train':
    #             train_mask[i] = True
    #         elif set_name == 'val':
    #             val_mask[i] = True
    #         else:
    #             test_mask[i] = True

    #     self.train_mask = train_mask
    #     self.val_mask = val_mask
    #     self.test_mask = test_mask
    

    #     if self.debug:
    #         print(f"Training edges: {self.train_mask.sum().item()}")
    #         print(f"Validation edges: {self.val_mask.sum().item()}")
    #         print(f"Test edges: {self.test_mask.sum().item()}")    
              
    # def split_dataset_v01(self, train_ratio=0.6, val_ratio=0.2, seed=42):
    #     """
    #     Divide los edges en train/val/test, asegurando que ambos sentidos de cada par vayan al mismo conjunto.
    #     Args:
    #         train_ratio: Proporción de edges en train.
    #         val_ratio: Proporción de edges en val.
    #         seed: Semilla para reproducibilidad.
    #     """
    #     import random
    #     random.seed(seed)
    #     torch.manual_seed(seed)

    #     number_of_edges = self.dgl_graph.num_edges()
    #     u, v = self.dgl_graph.edges()
        
    #     # Calcula el par canónico para todos los edges (vectorizado)
    #     u_np = u.numpy()
    #     v_np = v.numpy()
    #     canonical_edges = np.stack([np.minimum(u_np, v_np), np.maximum(u_np, v_np)], axis=1)
    #     canonical_edges_tuples = [tuple(edge) for edge in canonical_edges]
        
    #     # Asigna cada par canónico a un split una sola vez
    #     edge_to_set = dict()
    #     for edge in set(canonical_edges_tuples):
    #         rand_value = random.random()
    #         if rand_value < train_ratio:
    #             edge_to_set[edge] = 'train'
    #         elif rand_value < train_ratio + val_ratio:
    #             edge_to_set[edge] = 'val'
    #         else:
    #             edge_to_set[edge] = 'test'

    #     # Crea máscaras
    #     train_mask = torch.zeros(number_of_edges, dtype=torch.bool)
    #     val_mask = torch.zeros(number_of_edges, dtype=torch.bool)
    #     test_mask = torch.zeros(number_of_edges, dtype=torch.bool)

    #     for i, edge in enumerate(canonical_edges_tuples):
    #         set_name = edge_to_set[edge]
    #         if set_name == 'train':
    #             train_mask[i] = True
    #         elif set_name == 'val':
    #             val_mask[i] = True
    #         else:
    #             test_mask[i] = True

    #     self.train_mask = train_mask
    #     self.val_mask = val_mask
    #     self.test_mask = test_mask

    #     if self.debug:
    #         total = number_of_edges
    #         print(f"Training edges: {train_mask.sum().item()} ({100*train_mask.sum().item()/total:.2f}%)")
    #         print(f"Validation edges: {val_mask.sum().item()} ({100*val_mask.sum().item()/total:.2f}%)")
    #         print(f"Test edges: {test_mask.sum().item()} ({100*test_mask.sum().item()/total:.2f}%)")

    # def split_dataset_v1(self, train_ratio=0.6, val_ratio=0.2, seed=0):
    #     """
    #     Versión mejorada de split_dataset_v0:
    #     - Agrupa aristas simétricas para mantenerlas juntas.
    #     - Sólo las aristas con etiqueta >=0 se reparten entre train/val/test.
    #     - Las aristas con etiqueta -1 se mantienen en train_mask.
    #     Parámetros:
    #         train_ratio (float): proporción de pares en entrenamiento.
    #         val_ratio   (float): proporción de pares en validación.
    #         seed        (int):   semilla para reproducibilidad.
    #     """
    #     import random
    #     from collections import defaultdict
    #     import torch

    #     # 1) Obtener edges y máscara de etiquetado
    #     u, v   = self.dgl_graph.edges()
    #     rel    = self.dgl_graph.edata["Relationship"]
    #     is_lbl = rel >= 0

    #     # 2) Agrupar eids por par no direccional (min, max)
    #     pairs = defaultdict(list)
    #     for eid, (ui, vi) in enumerate(zip(u.tolist(), v.tolist())):
    #         key = (min(ui, vi), max(ui, vi))
    #         pairs[key].append(eid)

    #     # 3) Barajar los pares y calcular índices de split
    #     keys = list(pairs.keys())
    #     rng  = random.Random(seed)
    #     rng.shuffle(keys)

    #     n_keys  = len(keys)
    #     n_train = int(n_keys * train_ratio)
    #     n_val   = int(n_keys * val_ratio)

    #     train_keys = keys[:n_train]
    #     val_keys   = keys[n_train:n_train + n_val]
    #     test_keys  = keys[n_train + n_val:]

    #     # 4) Reunir eids etiquetados para cada split
    #     train_eids = [eid for key in train_keys for eid in pairs[key] if is_lbl[eid]]
    #     val_eids   = [eid for key in val_keys   for eid in pairs[key] if is_lbl[eid]]
    #     test_eids  = [eid for key in test_keys  for eid in pairs[key] if is_lbl[eid]]

    #     # 5) Construir máscaras booleanas
    #     num_e = self.dgl_graph.num_edges()
    #     train_mask = torch.zeros(num_e, dtype=torch.bool)
    #     val_mask   = torch.zeros(num_e, dtype=torch.bool)
    #     test_mask  = torch.zeros(num_e, dtype=torch.bool)

    #     train_mask[train_eids] = True
    #     val_mask[val_eids]     = True
    #     test_mask[test_eids]   = True

    #     # 6) Incluir sin-etiqueta sólo en entrenamiento
    #     train_mask[~is_lbl] = True

    #     # 7) Guardar en el objeto
    #     self.train_mask = train_mask
    #     self.val_mask   = val_mask
    #     self.test_mask  = test_mask

    #     if self.debug:
    #         print(f"[split_v1] Aristas total: {num_e}")
    #         print(f"[split_v1] Train: {train_mask.sum().item()} (etiquetadas={len(train_eids)})")
    #         print(f"[split_v1] Val:   {val_mask.sum().item()} (etiquetadas={len(val_eids)})")
    #         print(f"[split_v1] Test:  {test_mask.sum().item()} (etiquetadas={len(test_eids)})")
