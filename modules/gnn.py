import dgl
import torch
import numpy as np
from collections import defaultdict
import random
import bz2, gzip
from collections import Counter, defaultdict
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
            
            # 1.- Cargar CSV
            # --------------------------
            if self.debug:
                    print("[LOAD DATASET] Leyendo CSV …")
            self.dgl_graph = dgl.data.CSVDataset(data_path, force_reload=force_reload)[index_graph]
            
            if self.debug:
                    print(f"[LOAD DATASET] Grafo {self.dgl_graph}")
            # 2.- Simplificar grafo si se pide
            # --------------------------
            if simplify:
                if self.debug:
                    print("[LOAD DATASET] Simplificando  …")
                self.dgl_graph = dgl.to_simple(self.dgl_graph, return_counts=False)

            # 3.- Asignar atributo de aristas 'Relationship' como -1 por defecto
            # --------------------------
            num_e = self.dgl_graph.num_edges()
            self.dgl_graph.edata["Relationship"] = torch.full( (num_e,), -1, dtype=torch.int8)

            # 4.- Etiquetar desde archivo CAIDA si se entrega
            # --------------------------
            if caida_path is not None:
                if self.debug:
                    print(f"[LOAD DATASET] Etiquetando CAIDA → {caida_path}")
                self._fill_labels_from_caida_stream_fast(caida_path)
            
            
            if self.debug:
                cnt = Counter(self.dgl_graph.edata["Relationship"].tolist())
                print(f"[LOAD DATASET] etiquetas 0/1/2/-1 → {cnt}")
        
    def _fill_labels_from_caida_stream_fast(self, caida_file: str):


        # 1.- Diccionario rápido: (u, v) -> eid
        # --------------------------
        u, v = self.dgl_graph.edges()
        eid_map = {(int(u[i]), int(v[i])): i           # clave   = (u,v)
                for i in range(self.dgl_graph.num_edges())}  # valor  = id de arista

        # Buffers para aristas nuevas que no estén en el grafo
        buffer_src, buffer_dst, buffer_lbl = [], [], []

        # 2.- Elegir la función open según extensión
        # --------------------------
        opener = (bz2.open  if caida_file.endswith(".bz2") else
                gzip.open if caida_file.endswith(".gz")  else
                open)

        # 3.- Recorrer el archivo CAIDA
        # --------------------------
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

                # 4.- Procesar cada dirección
                # --------------------------
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

        # 5.- Añadir aristas nuevas (si hay)
        # --------------------------
        if buffer_src:   # lista no vacía
            self.dgl_graph = dgl.add_edges(
                self.dgl_graph,
                torch.tensor(buffer_src),
                torch.tensor(buffer_dst),
                data={"Relationship": torch.tensor(buffer_lbl, dtype=torch.int8)}
            )
            if self.debug:
                print(f"[CAIDA] Añadidas {len(buffer_src)} aristas que faltaban")

        # 6.- Resumen final (opcional)
        # --------------------------
        if self.debug:
            
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


        rng = random.Random(seed)
        torch.manual_seed(seed); np.random.seed(seed)

        u, v  = self.dgl_graph.edges()
        rel   = self.dgl_graph.edata["Relationship"]
        is_lbl = rel >= 0

        # 1.- Agrupar dos direcciones
        # --------------------------
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

        # 2.- Máscaras booleanas
        # --------------------------
        num_e = self.dgl_graph.num_edges()
        train_mask = torch.zeros(num_e, dtype=torch.bool)
        test_mask  = torch.zeros_like(train_mask)
        train_mask[train_eids] = True
        test_mask[test_eids]   = True

        self.dgl_graph.edata["train_mask"] = train_mask
        self.dgl_graph.edata["test_mask"]  = test_mask

        # 3.- Opcional: guardo para sampling
        # --------------------------
        if store_eids:
            self.train_eids = train_eids
            self.test_eids  = test_eids

        if self.debug:
            
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
        self.train_g = dgl.remove_edges(g, test_eids)

        # 3.- Subgrafos POSITIVOS
        # --------------------------
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


