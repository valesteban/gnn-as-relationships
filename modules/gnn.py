import dgl
import torch
import numpy as np
import scipy.sparse as sp
from collections import defaultdict



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

    def load_dataset(self, data_path, force_reload=False):
        """
        Carga un dataset y lo convierte en un grafo DGL.

        Parameters:
            data_path (str): Ruta al directorio que contiene los archivos CSV del dataset.
            force_reload (bool, optional): Si es True, fuerza la recarga del dataset desde el archivo CSV. Por defecto es False.

        Returns:
            None
        """

        self.dgl_graph = dgl.data.CSVDataset(data_path, force_reload=force_reload)[0]
        self.dgl_graph.ndata["feat"] = self.dgl_graph.ndata["feat"].float()

        if self.debug:
            print(self.dgl_graph)
    

    def split_graph_edges(self, train_ratio=0.8):
        u, v = self.dgl_graph.edges()
        num_edges = self.dgl_graph.num_edges()
        relationship = self.dgl_graph.edata["Relationship"]

        # Agrupar índices por clase
        class_to_eids = defaultdict(list)
        for i, rel in enumerate(relationship.tolist()):
            class_to_eids[rel].append(i)

        train_eids, test_eids = [], []

        # Split estratificado por clase
        for rel_class, eids in class_to_eids.items():
            eids = np.array(eids)
            np.random.shuffle(eids)
            split_point = int(len(eids) * train_ratio)
            train_eids.extend(eids[:split_point])
            test_eids.extend(eids[split_point:])

        train_eids = np.array(train_eids)
        test_eids = np.array(test_eids)

        train_pos_u, train_pos_v = u[train_eids], v[train_eids]
        test_pos_u, test_pos_v = u[test_eids], v[test_eids]

        # Negativos
        neg_u, neg_v = self.get_negative_edges(num_edges)
        train_neg_u, train_neg_v = neg_u[:len(train_eids)], neg_v[:len(train_eids)]
        test_neg_u, test_neg_v = neg_u[len(train_eids):], neg_v[len(train_eids):]

        # Crear grafos
        self.train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=self.dgl_graph.num_nodes())
        self.test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=self.dgl_graph.num_nodes())
        self.train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=self.dgl_graph.num_nodes())
        self.test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=self.dgl_graph.num_nodes())

        if "feat" in self.dgl_graph.ndata:
            for g in [self.train_pos_g, self.test_pos_g, self.train_neg_g, self.test_neg_g]:
                g.ndata["feat"] = self.dgl_graph.ndata["feat"]

        # Copiar labels a los subgrafos positivos
        self.train_pos_g.edata["Relationship"] = relationship[torch.tensor(train_eids)]
        self.test_pos_g.edata["Relationship"] = relationship[torch.tensor(test_eids)]
    
    def split_graph_edges_basic_full_graph(self, train_size=0.8):

        if self.debug:
            print("Dividiendo edges del grafo...")

    def split_graph_edges_basic(self, train_size=0.8):
                
        u,v = self.dgl_graph.edges()

        # IDs de lo edges
        eids = np.arange(self.dgl_graph.num_edges()) 
        # Shuffle the edges
        eids = np.random.permutation(eids)

        # Tamaño de train y test
        test_size = int(len(eids) * train_size)
        train_size = self.dgl_graph.num_edges() - test_size 

        # Selecciona los edges de test y train
        test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
        train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

        # Matriz de adyacencia
        adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))

        neg_u, neg_v = self.get_negative_edges( self.dgl_graph.num_edges())
        #  167.564 -> 30 seg  -> 
        #  334.996 -> 50 seg -> 100.000 paths
        #1.676.722 -> 4 min 17 seg -> 500.000 paths

        test_neg_u, test_neg_v = (
            neg_u[:test_size],
            neg_v[:test_size],
        )

        train_neg_u, train_neg_v = (
            neg_u[test_size:],
            neg_v[test_size:],
        )

        # Eliminar edges de test
        self.train_g = dgl.remove_edges(self.dgl_graph, eids[:test_size])

        self.train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=self.dgl_graph.num_nodes())
        self.train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=self.dgl_graph.num_nodes())    

        self.test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=self.dgl_graph.num_nodes())
        self.test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=self.dgl_graph.num_nodes())

        if "feat" in self.dgl_graph.ndata:
            self.train_pos_g.ndata["feat"] = self.dgl_graph.ndata["feat"]
            self.train_neg_g.ndata["feat"] = self.dgl_graph.ndata["feat"]
            self.test_pos_g.ndata["feat"] = self.dgl_graph.ndata["feat"]
            self.test_neg_g.ndata["feat"] = self.dgl_graph.ndata["feat"]
        
        # Copiar edata["Relationship"] si existe
        if "Relationship" in self.dgl_graph.edata:
            test_eids = torch.tensor(eids[:test_size])
            train_eids = torch.tensor(eids[test_size:])
            relationship = self.dgl_graph.edata["Relationship"]
            self.train_pos_g.edata["Relationship"] = relationship[train_eids]
            self.test_pos_g.edata["Relationship"] = relationship[test_eids]
    
    def split_graph_nodes(self, train_size=0.8):

        num_nodes = self.dgl_graph.num_nodes()
        nodes_ids = np.arange(num_nodes)
        nodes = self.dgl_graph.nodes()
        # Shuffle the nodes
        nodes_ids = np.random.permutation(nodes_ids)

        # Tamaño de train y test
        train_size = int(num_nodes * train_size)
        test_size = num_nodes - train_size 

        # Split
        train_nodes = nodes[nodes_ids[:train_size]]
        test_nodes = nodes[nodes_ids[train_size:]]

        # Crear máscaras
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)


        train_mask[train_nodes] = True
        test_mask[test_nodes] = True

        # Asignar al grafo
        self.dgl_graph.train_mask = train_mask
        self.dgl_graph.test_mask = test_mask

    
    def get_negative_edges(self, num_neg_samples): #FIXME: optimizar
        """
        Genera aristas negativas para el grafo dado.
        """
        print(f"Generando {num_neg_samples} aristas negativas...")
        neg_src_u = []
        neg_dst_v = []
        num_nodes = self.dgl_graph.num_nodes()

        for i in range(num_neg_samples):
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)
            while self.dgl_graph.has_edges_between(src, dst):
                src = np.random.randint(0, num_nodes)
                dst = np.random.randint(0, num_nodes)
            neg_src_u.append(src)
            neg_dst_v.append(dst)

        print(f"Aristas negativas generadas: {i}")

        return torch.tensor(neg_src_u), torch.tensor(neg_dst_v)
