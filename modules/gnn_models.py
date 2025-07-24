import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv, GraphConv, GATConv, GINConv, GatedGCNConv, GatedGraphConv
import dgl.function as fn
import dgl
import torch


# ------- Link Prediction Models -------

# GCN
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, out_feats_mlp=1):
        super().__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats)
        self.conv2 = GraphConv(hidden_feats, out_feats)

        # Enfoque encoder-decoder
        self.MLP = MLPPredictor(out_feats,out_feats_mlp)

        # Regresor para prediccion atributo
        self.regressor = nn.Linear(out_feats, 1)


    def encode(self, g, in_feat):
        g = dgl.add_self_loop(g)

        h = self.conv1(g, in_feat)
        h = F.relu(h)
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

    def forward(self, g, x):
        h = self.encode(g, x)  # Embeddings
        out = self.regressor(h)  # Predicción final
        return out 

    # def decode_all(self, z):
    #     return (z @ z.T) > 0

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats,  out_feats, out_feats_mlp=1):
        super().__init__()  # ✅ Esto es lo correcto
        self.conv1 = SAGEConv(in_feats, hidden_feats, 'mean')
        self.conv2 = SAGEConv(hidden_feats, out_feats, 'mean')

        self.MLP = MLPPredictor(out_feats,out_feats_mlp)

        # Regresor para prediccion atributo
        self.regressor = nn.Linear(out_feats, 1)


    def encode(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
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
    
    def forward(self, g, x):
        h = self.encode(g, x)  # Embeddings
        out = self.regressor(h)  # Predicción final
        return out 
       

class GAT(nn.Module):
    def __init__(self, in_feats, hidden_feats,  out_feats, out_feats_mlp=1,num_heads=1):
        super().__init__()
        self.conv1 = GATConv(in_feats, hidden_feats, num_heads)
        self.conv2 = GATConv(hidden_feats, out_feats, num_heads)

        self.MLP = MLPPredictor(out_feats,out_feats_mlp)

        # Regresor para prediccion atributo
        self.regressor = nn.Linear(out_feats, 1)


    def encode(self, g, in_feat):
        g = dgl.add_self_loop(g)
        h = self.conv1(g, in_feat)
        # h = torch.flatten(h, start_dim=1, end_dim=2) # only for 1 layer
        h = F.relu(h)
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

    def forward(self, g, x):
        h = self.encode(g, x)  # Embeddings
        out = self.regressor(h)  # Predicción final
        return out 
    



class MLPPredictor(nn.Module):
    def __init__(self, h_feats,out_features):
        super().__init__()
        # Entrada es 2 * h_feats porque porque toma las entradas de los dos nodos que comparteb una arista
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, out_features)

    def apply_edges(self, edges):
        h = torch.cat([edges.src["h"], edges.dst["h"]], 1)
        return {"score": self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]


