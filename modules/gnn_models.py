import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv, GraphConv, GATConv, GINConv, GatedGCNConv, GatedGraphConv
import dgl.function as fn
import dgl
import torch


# ------- Link Prediction Models -------

# GCN
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, out_feats_mlp=1, drop=0.3):
        super().__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats)
        self.conv2 = GraphConv(hidden_feats, out_feats)
    
        # Enfoque encoder-decoder
        self.MLP = MLPPredictor(out_feats,out_feats_mlp)

        # Decodificador para prediccion de aristas
        self.BilinearDecoder = BilinearPredictor(out_feats, 3)

        # Regresor para prediccion atributo
        self.regressor = nn.Linear(out_feats, 1)

        self.drop  = nn.Dropout(drop)           




    # def encode(self, g, in_feat):
    #     g = dgl.add_self_loop(g)

    #     h = self.conv1(g, in_feat)
    #     h = F.relu(h)
    #     h = self.conv2(g, h)
    #     return h
    
    def encode(self, g, x):
        g = dgl.add_self_loop(g)
        h = self.drop(F.relu(self.conv1(g, x)))  # dropout tras 1.ª capa
        h = self.conv2(g, h)                     # última capa sin ReLU
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
        h = self.encode(g, x)  # Embeddings
        out = self.regressor(h).squeeze(-1)  # Predicción final
        return out 

    # def decode_all(self, z):
    #     return (z @ z.T) > 0

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats,  out_feats, out_feats_mlp=1, drop=0.3):
        super().__init__()  # ✅ Esto es lo correcto
        self.conv1 = SAGEConv(in_feats, hidden_feats, 'mean')
        self.conv2 = SAGEConv(hidden_feats, out_feats, 'mean')

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
        h = self.drop(F.relu(self.conv1(g, x)))
        h = self.conv2(g, h)
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
        h = self.encode(g, x)  # Embeddings
        out = self.regressor(h).squeeze(-1)  # Predicción final
        return out 
       

class GAT(nn.Module):
    def __init__(self, in_feats, hidden_feats,  out_feats, out_feats_mlp=1,num_heads=1, drop=0.3):
        super().__init__()
        self.conv1 = GATConv(in_feats, hidden_feats, num_heads)
        self.conv2 = GATConv(hidden_feats, out_feats, num_heads)

        self.MLP = MLPPredictor(out_feats,out_feats_mlp)
        # Decodificador para prediccion de aristas
        self.BilinearDecoder = BilinearPredictor(out_feats, 3)

        # Regresor para prediccion atributo
        self.regressor = nn.Linear(out_feats, 1)

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
        h = self.drop(F.relu(self.conv1(g, x)))
        h = self.conv2(g, h)
        h = torch.flatten(h, start_dim=1, end_dim=3)

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
        h = self.encode(g, x)  # Embeddings
        out = self.regressor(h).squeeze(-1)  # Predicción final
        return out 
    

# Predictor Models -------------------------------------

class MLPPredictor(nn.Module):
    def __init__(self, h_feats,out_features, drop=0.3):
        super().__init__()
        # Entrada es 2 * h_feats porque porque toma las entradas de los dos nodos que comparteb una arista
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.act  = nn.ReLU()
        self.drop = nn.Dropout(drop)           # ← NUEVO
        self.W2 = nn.Linear(h_feats, out_features)


    def apply_edges(self, edges):
        h = torch.cat([edges.src["h"], edges.dst["h"]], 1) # (E, 2F)
        h1 = self.act(self.W1(h))   
        h1 = self.drop(h1)                                       # dropout
        out = self.W2(h1).squeeze(1)                             # (E, C)  ó  (E,) si C=1
        return {"score": out}                             # (E, F)

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]


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

# CON BATCHES -------------------------------------------------------------------------------------

#  Helper para saber si recibimos lista de blocks o grafo suelto
def _as_graph(g_or_blocks):
    """Devuelve el grafo (o el primer Block) con el que se hará message-passing."""
    return g_or_blocks[0] if isinstance(g_or_blocks, list) else g_or_blocks


class GCNSampler(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, out_feats_mlp=1):
        super().__init__()
        self.conv1 = GraphConv(in_feats,  hidden_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_feats, out_feats, allow_zero_in_degree=True)
        self.MLP   = MLPPredictor(out_feats, out_feats_mlp)

    # g_or_blocks puede ser un DGLGraph o [Block,...]
    def encode(self, g_or_blocks, x):
        g = _as_graph(g_or_blocks)
        h = F.relu(self.conv1(g, x))
        h = self.conv2(g, h)
        return h

    def decodeMLP(self, g, h):
        return self.MLP(g, h)

    def forward(self, g_or_blocks, x):
        return self.encode(g_or_blocks, x)


class GraphSAGESample(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, out_feats_mlp=1):
        super().__init__()
        self.conv1 = GraphConv(in_feats,  hidden_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_feats, out_feats, allow_zero_in_degree=True)
        self.MLP   = MLPPredictor(out_feats, out_feats_mlp)

    def encode(self, g_or_blocks, x):
        g = _as_graph(g_or_blocks)
        h = F.relu(self.conv1(g, x))
        h = self.conv2(g, h)
        return h

    def decodeMLP(self, g, h):
        return self.MLP(g, h)

    def forward(self, g_or_blocks, x):
        return self.encode(g_or_blocks, x)


class GATSample(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, out_feats_mlp=1, num_heads=1):
        super().__init__()
        self.conv1 = GATConv(in_feats, hidden_feats, num_heads, allow_zero_in_degree=True)
        self.conv2 = GATConv(hidden_feats, out_feats, num_heads, allow_zero_in_degree=True)
        self.MLP   = MLPPredictor(out_feats, out_feats_mlp)
        self.num_heads = num_heads

    def encode(self, g_or_blocks, x):
        g = _as_graph(g_or_blocks)
        h = F.relu(self.conv1(g, x))
        h = self.conv2(g, h)                 # (N, out_feats, heads)
        h = h.flatten(1)                     # → (N, out_feats * heads)
        return h

    def decodeMLP(self, g, h):
        return self.MLP(g, h)

    def forward(self, g_or_blocks, x):
        return self.encode(g_or_blocks, x)


class MLPPredictor(nn.Module):
    """Decodificador MLP para clasificación de aristas."""
    def __init__(self, h_dim, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(h_dim * 2, h_dim)
        self.fc2 = nn.Linear(h_dim, n_classes)

    def apply_edges(self, edges):
        z = torch.cat([edges.src["h"], edges.dst["h"]], dim=1)
        return {"score": self.fc2(F.relu(self.fc1(z))).squeeze(-1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]
