import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv, GraphConv, GATConv
import dgl.function as fn
import dgl
import torch

# 1. Decodificadores
class MLPPredictor(nn.Module):
    def __init__(self, h_dim, n_classes, edge_dim=1, drop=0.3):
        super().__init__()
        self.edge_dim = edge_dim
        self.fc1  = nn.Linear(h_dim * 2 + edge_dim, h_dim)
        self.drop = nn.Dropout(drop)
        self.fc2  = nn.Linear(h_dim, n_classes)

    def apply_edges(self, edges):
        hu, hv = edges.src["h"], edges.dst["h"]
        parts = [hu, hv]
        if self.edge_dim > 0 and "edge_feat" in edges.data:
            w = edges.data["edge_feat"]
            if w.dim() == 1: w = w.unsqueeze(1)
            parts.append(w)
        z = torch.cat(parts, dim=1)
        h = self.drop(F.relu(self.fc1(z)))
        score = self.fc2(h)
        return {"score": score.squeeze(-1) if score.shape[-1] == 1 else score}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]


class DotProductPredictor(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            return g.edata["score"].squeeze(-1)


class BilinearPredictor(nn.Module):
    def __init__(self, h_dim, n_cls):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_cls, h_dim, h_dim))  # (C,F,F)
        self.b = nn.Parameter(torch.zeros(n_cls))                # (C,)

    def forward(self, g, h):                 # → logits (E, C)
        with g.local_scope():
            g.ndata["h"] = h                 # (N, F)

            def edge_fn(e):
                hu, hv = e.src["h"], e.dst["h"]         # (E,F) cada uno
                # score_ec = hu_f · W_c_fk · hv_k
                scores = torch.einsum("ef,cfk,ek->ec", hu, self.W, hv) + self.b
                return {"score": scores}                # (E,C)

            g.apply_edges(edge_fn)
            return g.edata["score"]

# 2. Modelos de 3 Capas
class GCN3L(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, out_feats_mlp=1, drop=0.3, edge_dim=0):
        super().__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats)
        self.conv2 = GraphConv(hidden_feats, hidden_feats)
        self.conv3 = GraphConv(hidden_feats, out_feats)

        self.MLP = MLPPredictor(out_feats, out_feats_mlp, edge_dim=edge_dim)
        self.Bilinear = BilinearPredictor(out_feats, out_feats_mlp)
        self.DotProduct = DotProductPredictor() # Reemplaza Bilinear
        self.regressor = nn.Linear(out_feats, 1)
        self.drop = nn.Dropout(drop)

    def encode(self, g, x):
        g = dgl.add_self_loop(g)
        h = self.drop(F.relu(self.conv1(g, x)))
        h = self.drop(F.relu(self.conv2(g, h)))
        return self.conv3(g, h)

    def decodeMLP(self, g, h): 
        return self.MLP(g, h)
    
    def decodeDotProduct(self, g, h): 
        return self.DotProduct(g, h)

    def decodeBilinear(self, g, h):
        return self.Bilinear(g, h)
    
    def forward(self, g, x):
        """Forward para regresión de atributos de nodos.
        Retorna predicciones para cada nodo.
        Para link prediction, usar: encode() 
        """
        h = self.encode(g, x)  # Embeddings
        return self.regressor(h)  # Predicción de atributo por nodo 

class GraphSAGE3L(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, out_feats_mlp=1, drop=0.3, edge_dim=0):
        super().__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats, 'mean')
        self.conv2 = SAGEConv(hidden_feats, hidden_feats, 'mean')
        self.conv3 = SAGEConv(hidden_feats, out_feats, 'mean')

        self.MLP = MLPPredictor(out_feats, out_feats_mlp, edge_dim=edge_dim)
        self.Bilinear = BilinearPredictor(out_feats, out_feats_mlp)

        self.DotProduct = DotProductPredictor() # Reemplaza Bilinear
        self.regressor = nn.Linear(out_feats, 1)
        self.drop = nn.Dropout(drop)

    def encode(self, g, x):
        h = self.drop(F.relu(self.conv1(g, x)))
        h = self.drop(F.relu(self.conv2(g, h)))
        return self.conv3(g, h)

    def decodeMLP(self, g, h): 
        return self.MLP(g, h)
    
    def decodeBilinear(self, g, h):
        return self.Bilinear(g, h)
    
    def decodeDotProduct(self, g, h): 
        return self.DotProduct(g, h)

    def forward(self, g, x):
        """Forward para regresión de atributos de nodos.
        Retorna predicciones para cada nodo.
        Para link prediction, usar: encode() 
        """
        h = self.encode(g, x)  # Embeddings
        return self.regressor(h)  # Predicción de atributo por nodo 

class GAT3L(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, out_feats_mlp=1, num_heads=4, drop=0.3, edge_dim=0):
        super().__init__()
        self.conv1 = GATConv(in_feats, hidden_feats, num_heads)
        self.conv2 = GATConv(hidden_feats * num_heads, hidden_feats, num_heads)
        self.conv3 = GATConv(hidden_feats * num_heads, out_feats, num_heads)
        
        self.MLP = MLPPredictor(out_feats * num_heads, out_feats_mlp, edge_dim=edge_dim)
        self.Bilinear = BilinearPredictor(out_feats * num_heads, out_feats_mlp)
        self.DotProduct = DotProductPredictor() # Reemplaza Bilinear
        self.regressor = nn.Linear(out_feats * num_heads, 1)
        self.drop = nn.Dropout(drop)

    def encode(self, g, x):
        g = dgl.add_self_loop(g)
        h = self.drop(F.relu(self.conv1(g, x).flatten(1)))
        h = self.drop(F.relu(self.conv2(g, h).flatten(1)))
        return self.conv3(g, h).flatten(1)

    def decodeMLP(self, g, h): 
        return self.MLP(g, h)
    
    def decodeBilinear(self, g, h):
        return self.Bilinear(g, h)
    
    def decodeDotProduct(self, g, h): 
        return self.DotProduct(g, h)

    def forward(self, g, x):
        """Forward para regresión de atributos de nodos.
        Retorna predicciones para cada nodo.
        Para link prediction, usar: encode() 
        """
        h = self.encode(g, x)  # Embeddings
        return self.regressor(h)  # Predicción de atributo por nodo 

# 3. Modelos de 2 Capas
class GCN2L(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, out_feats_mlp=1, drop=0.3, edge_dim=0):
        super().__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats)
        self.conv2 = GraphConv(hidden_feats, out_feats)

        self.MLP = MLPPredictor(out_feats, out_feats_mlp, edge_dim=edge_dim)
        self.Bilinear = BilinearPredictor(out_feats, out_feats_mlp)
        self.DotProduct = DotProductPredictor() # Reemplaza Bilinear
        self.drop = nn.Dropout(drop)

    def encode(self, g, x):
        g = dgl.add_self_loop(g)
        h = self.drop(F.relu(self.conv1(g, x)))
        return self.conv2(g, h)

    def decodeMLP(self, g, h): 
        return self.MLP(g, h)
    
    def decodeBilinear(self, g, h):
        return self.Bilinear(g, h)
    
    def decodeDotProduct(self, g, h): 
        return self.DotProduct(g, h)
    
    def forward(self, g, x):
        """Forward para regresión de atributos de nodos.
        Retorna predicciones para cada nodo.
        Para link prediction, usar: encode() 
        """
        h = self.encode(g, x)  # Embeddings
        return self.regressor(h)  # Predicción de atributo por nodo 

class GraphSAGE2L(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, out_feats_mlp=1, drop=0.3, edge_dim=0):
        super().__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats, 'mean')
        self.conv2 = SAGEConv(hidden_feats, out_feats, 'mean')

        self.MLP = MLPPredictor(out_feats, out_feats_mlp, edge_dim=edge_dim)
        self.Bilinear = BilinearPredictor(out_feats, out_feats_mlp)

        self.DotProduct = DotProductPredictor() # Reemplaza Bilinear
        self.drop = nn.Dropout(drop)

    def encode(self, g, x):
        h = self.drop(F.relu(self.conv1(g, x)))
        return self.conv2(g, h)

    def decodeMLP(self, g, h): 
        return self.MLP(g, h)
    
    def decodeBilinear(self, g, h):
        return self.Bilinear(g, h)

    def decodeDotProduct(self, g, h): 
        return self.DotProduct(g, h)

    def forward(self, g, x):
        """Forward para regresión de atributos de nodos.
        Retorna predicciones para cada nodo.
        Para link prediction, usar: encode() 
        """
        h = self.encode(g, x)  # Embeddings
        return self.regressor(h)  # Predicción de atributo por nodo 

class GAT2L(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, out_feats_mlp=1, num_heads=4, drop=0.3, edge_dim=0):
        super().__init__()
        self.conv1 = GATConv(in_feats, hidden_feats, num_heads)
        self.conv2 = GATConv(hidden_feats * num_heads, out_feats, num_heads)

        self.MLP = MLPPredictor(out_feats * num_heads, out_feats_mlp, edge_dim=edge_dim)
        self.Bilinear = BilinearPredictor(out_feats, out_feats_mlp)

        self.DotProduct = DotProductPredictor() # Reemplaza Bilinear
        self.drop = nn.Dropout(drop)

    def encode(self, g, x):
        g = dgl.add_self_loop(g)
        h = self.drop(F.relu(self.conv1(g, x).flatten(1)))
        return self.conv2(g, h).flatten(1)

    def decodeMLP(self, g, h): 
        return self.MLP(g, h)
    
    def decodeBilinear(self, g, h):
        return self.Bilinear(g, h)

    def decodeDotProduct(self, g, h): 
        return self.DotProduct(g, h)

    def forward(self, g, x):
        """Forward para regresión de atributos de nodos.
        Retorna predicciones para cada nodo.
        Para link prediction, usar: encode() 
        """
        h = self.encode(g, x)  # Embeddings
        return self.regressor(h)  # Predicción de atributo por nodo 


################################################################################################################
## CON BATCHES
################################################################################################################

#  Helper para saber si recibimos lista de blocks o grafo suelto
def _as_graph(g_or_blocks):
    """Devuelve el grafo (o el primer Block) con el que se hará message-passing."""
    return g_or_blocks[0] if isinstance(g_or_blocks, list) else g_or_blocks


class GCNSampler2L(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, out_feats_mlp=1):
        super().__init__()
        self.conv1 = GraphConv(in_feats,  hidden_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_feats, out_feats, allow_zero_in_degree=True)
        self.MLP   = MLPPredictor(out_feats, out_feats_mlp)

    # g_or_blocks puede ser un DGLGraph o [Block,...]
    def encode(self, g_or_blocks, x):
        # Caso 1: bloques → Neighbor Sampling
        if isinstance(g_or_blocks, list):
            h = F.relu(self.conv1(g_or_blocks[0], x))
            h = self.conv2(g_or_blocks[1], h)
        else:
            # Caso 2: subgrafo → ClusterGCN o entrenamiento completo
            h = F.relu(self.conv1(g_or_blocks, x))
            h = self.conv2(g_or_blocks, h)
        return h

    def decodeMLP(self, g, h):
        return self.MLP(g, h)

    def forward(self, g_or_blocks, x):
        return self.encode(g_or_blocks, x)


class GraphSAGESample2L(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, out_feats_mlp=1, aggregator='mean'):
        super().__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats, aggregator)
        self.conv2 = SAGEConv(hidden_feats, out_feats, aggregator)
        self.MLP   = MLPPredictor(out_feats, out_feats_mlp)


    def encode(self, g_or_blocks, x):
        # Caso 1: bloques → Neighbor Sampling
        if isinstance(g_or_blocks, list):
            h = F.relu(self.conv1(g_or_blocks[0], x))
            h = self.conv2(g_or_blocks[1], h)
        else:
            # Caso 2: subgrafo → ClusterGCN o entrenamiento completo
            h = F.relu(self.conv1(g_or_blocks, x))
            h = self.conv2(g_or_blocks, h)
        return h

    def decodeMLP(self, g, h):
        return self.MLP(g, h)

    def forward(self, g_or_blocks, x):
        return self.encode(g_or_blocks, x)


class GATSample2L(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, out_feats_mlp=1, num_heads=4):
        super().__init__()
        self.conv1 = GATConv(in_feats, hidden_feats, num_heads, allow_zero_in_degree=True)
        self.conv2 = GATConv(hidden_feats, out_feats, num_heads, allow_zero_in_degree=True)
        self.MLP   = MLPPredictor(out_feats, out_feats_mlp)
        self.num_heads = num_heads

    def encode(self, g_or_blocks, x):
        # Caso 1: bloques → Neighbor Sampling
        if isinstance(g_or_blocks, list):
            h = F.relu(self.conv1(g_or_blocks[0], x))
            h = self.conv2(g_or_blocks[1], h)
        else:
            # Caso 2: subgrafo → ClusterGCN o entrenamiento completo
            h = F.relu(self.conv1(g_or_blocks, x))
            h = self.conv2(g_or_blocks, h)
        h = h.flatten(1)  
        return h
    

    def decodeMLP(self, g, h):
        return self.MLP(g, h)

    def forward(self, g_or_blocks, x):
        return self.encode(g_or_blocks, x)
