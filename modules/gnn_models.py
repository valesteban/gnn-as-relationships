import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv, GraphConv, GATConv
import dgl.function as fn
import dgl
import torch


# Link Prediction Models
# --------------------------

# GCN
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, out_feats_mlp=1, drop=0.3):
        super().__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats)
        self.conv2 = GraphConv(hidden_feats, out_feats)
        self.bn1 = nn.BatchNorm1d(hidden_feats)
        self.bn2 = nn.BatchNorm1d(out_feats)
        self.res1 = nn.Linear(in_feats, hidden_feats, bias=False)
        self.res2 = nn.Linear(hidden_feats, out_feats, bias=False)

        self.MLP = MLPPredictor(out_feats,out_feats_mlp)
        self.BilinearDecoder = BilinearPredictor(out_feats, 3)
        self.regressor = nn.Linear(out_feats, 1)
        self.drop  = nn.Dropout(drop)           


    def encode(self, g, x):
        g = dgl.add_self_loop(g)
        h = F.relu(self.bn1(self.conv1(g, x)) + self.res1(x))
        h = self.drop(h)
        h = self.bn2(self.conv2(g, h) + self.res2(h))
        return h

    
    def decodeDotProduct(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata["score"][:, 0]
        
    def decodeMLP(self, g, h):
        return self.MLP(g, h)  
    
    def decodeBilinear(self, g, h):
        return self.BilinearDecoder(g, h)

    def forward(self, g, x):
        """Forward para regresión de atributos de nodos.
        Retorna predicciones para cada nodo.
        Para link prediction, usar: encode() + decodeDotProduct/decodeMLP/decodeBilinear()
        """
        h = self.encode(g, x)  # Embeddings
        return self.regressor(h)  # Predicción de atributo por nodo 

    # def decode_all(self, z):
    #     return (z @ z.T) > 0

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats,  out_feats, out_feats_mlp=1, drop=0.3):
        super().__init__()  # ✅ Esto es lo correcto
        self.conv1 = SAGEConv(in_feats, hidden_feats, 'mean')
        self.conv2 = SAGEConv(hidden_feats, out_feats, 'mean')
        self.bn1 = nn.BatchNorm1d(hidden_feats)
        self.bn2 = nn.BatchNorm1d(out_feats)
        self.res1 = nn.Linear(in_feats, hidden_feats, bias=False)
        self.res2 = nn.Linear(hidden_feats, out_feats, bias=False)

        self.MLP = MLPPredictor(out_feats,out_feats_mlp)
        # Decodificador para prediccion de aristas
        self.BilinearDecoder = BilinearPredictor(out_feats, 3)

        # Regresor para prediccion atributo
        self.regressor = nn.Linear(out_feats, 1)

        self.drop  = nn.Dropout(drop)


    # def encode(self, g, in_feat):
    #     h = self.conv1(g, in_feat)
    #     h = F.relu(h)
    #     h = self.conv2(g, h)
    #     return h
    
    def encode(self, g, x):
        h = self.drop(F.relu(self.bn1(self.conv1(g, x)) + self.res1(x)))
        h = self.bn2(self.conv2(g, h) + self.res2(h))
        return h
    
    def decodeDotProduct(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata["score"][:, 0]
        
    def decodeMLP(self, g, h):
        return self.MLP(g, h)  
    

    def decodeBilinear(self, g, h):
        return self.BilinearDecoder(g, h)
    
    def forward(self, g, x):
        """Forward para regresión de atributos de nodos.
        Retorna predicciones para cada nodo.
        Para link prediction, usar: encode() + decodeDotProduct/decodeMLP/decodeBilinear()
        """
        h = self.encode(g, x)  # Embeddings
        return self.regressor(h)  # Predicción de atributo por nodo 
       

class GAT(nn.Module):
    def __init__(self, in_feats, hidden_feats,  out_feats, out_feats_mlp=1,num_heads=1, drop=0.3):
        super().__init__()
        self.conv1 = GATConv(in_feats, hidden_feats, num_heads)
        self.conv2 = GATConv(hidden_feats * num_heads, out_feats, num_heads)
        self.bn1 = nn.BatchNorm1d(hidden_feats * num_heads)
        self.bn2 = nn.BatchNorm1d(out_feats * num_heads)
        self.res1 = nn.Linear(in_feats, hidden_feats * num_heads, bias=False)
        self.res2 = nn.Linear(hidden_feats * num_heads, out_feats * num_heads, bias=False)

        self.MLP = MLPPredictor(out_feats * num_heads, out_feats_mlp)
        # Decodificador para prediccion de aristas
        self.BilinearDecoder = BilinearPredictor(out_feats * num_heads, 3)

        # Regresor para prediccion atributo
        self.regressor = nn.Linear(out_feats * num_heads, 1)

        self.drop  = nn.Dropout(drop)


    # def encode(self, g, in_feat):
    #     g = dgl.add_self_loop(g)
    #     h = self.conv1(g, in_feat)
    #     # h = torch.flatten(h, start_dim=1, end_dim=2) # only for 1 layer
    #     h = F.relu(h)
    #     h = self.conv2(g, h)
    #     h = torch.flatten(h, start_dim=1, end_dim=3)        
    #     return h
    
    def encode(self, g, x):
        g = dgl.add_self_loop(g)
        h = self.drop(F.relu(self.bn1(self.conv1(g, x).flatten(1)) + self.res1(x)))
        h = self.bn2(self.conv2(g, h).flatten(1) + self.res2(h))
        return h
    
    def decodeDotProduct(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata["score"][:, 0]
    
    def decodeMLP(self, g, h):
        return self.MLP(g, h)  
    

    def decodeBilinear(self, g, h):
        return self.BilinearDecoder(g, h)

    def forward(self, g, x):
        """Forward para regresión de atributos de nodos.
        Retorna predicciones para cada nodo.
        Para link prediction, usar: encode() + decodeDotProduct/decodeMLP/decodeBilinear()
        """
        h = self.encode(g, x)  # Embeddings
        return self.regressor(h)  # Predicción de atributo por nodo 
    

# Predictor Models
# --------------------------
# Nota: MLPPredictor está definido más abajo (línea ~307) con versión actualizada

class BilinearPredictor(nn.Module):
    def __init__(self, h_dim, n_cls):
        super().__init__()
        # Scale by 1/h_dim so initial logits are O(1) instead of O(h_dim^2)
        self.W = nn.Parameter(torch.randn(n_cls, h_dim, h_dim) / h_dim)  # (C,F,F)
        self.b = nn.Parameter(torch.zeros(n_cls))                         # (C,)

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

# CON BATCHES
# --------------------------

#  Helper para saber si recibimos lista de blocks o grafo suelto
def _as_graph(g_or_blocks):
    """Devuelve el grafo (o el primer Block) con el que se hará message-passing."""
    return g_or_blocks[0] if isinstance(g_or_blocks, list) else g_or_blocks



class GCNSampler(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, out_feats_mlp=1):
        super().__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats, allow_zero_in_degree=True)
        self.bn1   = nn.BatchNorm1d(hidden_feats)
        self.conv2 = GraphConv(hidden_feats, out_feats, allow_zero_in_degree=True)
        self.bn2   = nn.BatchNorm1d(out_feats)
        self.MLP   = MLPPredictor(out_feats, out_feats_mlp)

    def encode(self, g_or_blocks, x):
        if isinstance(g_or_blocks, list):
            # Neighbor Sampling: g_or_blocks es una lista de bloques
            h = F.relu(self.bn1(self.conv1(g_or_blocks[0], x)))
            h = self.bn2(self.conv2(g_or_blocks[1], h))
        else:
            # ClusterGCN o grafo completo
            h = F.relu(self.bn1(self.conv1(g_or_blocks, x)))
            h = self.bn2(self.conv2(g_or_blocks, h))
        return h

    def decodeMLP(self, g, h):
        return self.MLP(g, h)

    def forward(self, g_or_blocks, x):
        return self.encode(g_or_blocks, x)


class GraphSAGESample(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, out_feats_mlp=1, aggregator='mean'):
        super().__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats, aggregator)
        self.bn1   = nn.BatchNorm1d(hidden_feats)
        self.conv2 = SAGEConv(hidden_feats, out_feats, aggregator)
        self.bn2   = nn.BatchNorm1d(out_feats)
        self.MLP   = MLPPredictor(out_feats, out_feats_mlp)

    def encode(self, g_or_blocks, x):
        if isinstance(g_or_blocks, list):
            h = F.relu(self.bn1(self.conv1(g_or_blocks[0], x)))
            h = self.bn2(self.conv2(g_or_blocks[1], h))
        else:
            h = F.relu(self.bn1(self.conv1(g_or_blocks, x)))
            h = self.bn2(self.conv2(g_or_blocks, h))
        return h

    def decodeMLP(self, g, h):
        return self.MLP(g, h)

    def forward(self, g_or_blocks, x):
        return self.encode(g_or_blocks, x)


class GATSample(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, out_feats_mlp=1, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        # conv1: salida [N, num_heads, hidden_feats] → se aplana a [N, num_heads*hidden_feats]
        self.conv1 = GATConv(in_feats, hidden_feats, num_heads, allow_zero_in_degree=True)
        self.bn1   = nn.BatchNorm1d(hidden_feats * num_heads)
        # conv2: entrada [N, num_heads*hidden_feats] → salida [N, num_heads, out_feats]
        self.conv2 = GATConv(hidden_feats * num_heads, out_feats, num_heads, allow_zero_in_degree=True)
        self.bn2   = nn.BatchNorm1d(out_feats * num_heads)
        self.MLP   = MLPPredictor(out_feats * num_heads, out_feats_mlp)

    def encode(self, g_or_blocks, x):
        if isinstance(g_or_blocks, list):
            h = F.relu(self.bn1(self.conv1(g_or_blocks[0], x).flatten(1)))
            h = self.bn2(self.conv2(g_or_blocks[1], h).flatten(1))
        else:
            h = F.relu(self.bn1(self.conv1(g_or_blocks, x).flatten(1)))
            h = self.bn2(self.conv2(g_or_blocks, h).flatten(1))
        return h

    def decodeMLP(self, g, h):
        return self.MLP(g, h)

    def forward(self, g_or_blocks, x):
        return self.encode(g_or_blocks, x)


class MLPPredictor(nn.Module):
    """Decodificador MLP para clasificación de aristas.

    Usa [h_u | h_v | h_u - h_v | h_u * h_v] como feature de arista,
    capturando tanto información absoluta como la asimetría direccional
    (clave para distinguir C2P de P2C en el grafo de Internet).
    """
    def __init__(self, h_dim, n_classes, dropout=0.3):
        super().__init__()
        # 4 * h_dim: concat + diferencia + producto elemento a elemento
        self.fc1     = nn.Linear(h_dim * 4, h_dim * 2)
        self.bn1     = nn.BatchNorm1d(h_dim * 2)
        self.fc2     = nn.Linear(h_dim * 2, h_dim)
        self.fc3     = nn.Linear(h_dim, n_classes)
        self.dropout = nn.Dropout(dropout)

    def apply_edges(self, edges):
        hu, hv = edges.src["h"], edges.dst["h"]
        # Diferencia y producto capturan la direccionalidad explícitamente
        z = torch.cat([hu, hv, hu - hv, hu * hv], dim=1)
        z = self.dropout(F.relu(self.bn1(self.fc1(z))))
        z = self.dropout(F.relu(self.fc2(z)))
        return {"score": self.fc3(z)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]
