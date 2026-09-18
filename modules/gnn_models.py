import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
import torch

# 1. Decodificadores
#    Reciben `edge_index` (2, E) con las aristas a puntuar y los embeddings `h` (N, F).
class MLPPredictor(nn.Module):
    def __init__(self, h_dim, n_classes, edge_dim=1, drop=0.3):
        super().__init__()
        self.edge_dim = edge_dim
        self.fc1  = nn.Linear(h_dim * 2 + edge_dim, h_dim)
        self.drop = nn.Dropout(drop)
        self.fc2  = nn.Linear(h_dim, n_classes)

    def forward(self, edge_index, h, edge_attr=None):
        hu, hv = h[edge_index[0]], h[edge_index[1]]
        parts = [hu, hv]
        if self.edge_dim > 0 and edge_attr is not None:
            w = edge_attr
            if w.dim() == 1: w = w.unsqueeze(1)
            parts.append(w)
        z = torch.cat(parts, dim=1)
        z = self.drop(F.relu(self.fc1(z)))
        score = self.fc2(z)
        return score.squeeze(-1) if score.shape[-1] == 1 else score


class DotProductPredictor(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, edge_index, h):
        hu, hv = h[edge_index[0]], h[edge_index[1]]
        return (hu * hv).sum(dim=-1)


class BilinearPredictor(nn.Module):
    def __init__(self, h_dim, n_cls):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_cls, h_dim, h_dim))  # (C,F,F)
        self.b = nn.Parameter(torch.zeros(n_cls))                # (C,)

    def forward(self, edge_index, h):        # → logits (E, C)
        hu, hv = h[edge_index[0]], h[edge_index[1]]     # (E,F) cada uno
        # score_ec = hu_f · W_c_fk · hv_k
        return torch.einsum("ef,cfk,ek->ec", hu, self.W, hv) + self.b   # (E,C)

# 2. Modelos de 3 Capas
class GCN3L(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, out_feats_mlp=1, drop=0.3, edge_dim=0):
        super().__init__()
        # GCNConv agrega self-loops y normaliza (equivale a add_self_loop + GraphConv)
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, hidden_feats)
        self.conv3 = GCNConv(hidden_feats, out_feats)

        self.MLP = MLPPredictor(out_feats, out_feats_mlp, edge_dim=edge_dim)
        self.Bilinear = BilinearPredictor(out_feats, out_feats_mlp)
        self.DotProduct = DotProductPredictor() # Reemplaza Bilinear
        self.regressor = nn.Linear(out_feats, 1)
        self.drop = nn.Dropout(drop)

    def encode(self, x, edge_index):
        h = self.drop(F.relu(self.conv1(x, edge_index)))
        h = self.drop(F.relu(self.conv2(h, edge_index)))
        return self.conv3(h, edge_index)

    def decodeMLP(self, edge_index, h, edge_attr=None):
        return self.MLP(edge_index, h, edge_attr)

    def decodeDotProduct(self, edge_index, h):
        return self.DotProduct(edge_index, h)

    def decodeBilinear(self, edge_index, h):
        return self.Bilinear(edge_index, h)

    def forward(self, x, edge_index):
        """Forward para regresión de atributos de nodos.
        Retorna predicciones para cada nodo.
        Para link prediction, usar: encode()
        """
        h = self.encode(x, edge_index)  # Embeddings
        return self.regressor(h)  # Predicción de atributo por nodo

class GraphSAGE3L(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, out_feats_mlp=1, drop=0.3, edge_dim=0):
        super().__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats, aggr='mean')
        self.conv2 = SAGEConv(hidden_feats, hidden_feats, aggr='mean')
        self.conv3 = SAGEConv(hidden_feats, out_feats, aggr='mean')

        self.MLP = MLPPredictor(out_feats, out_feats_mlp, edge_dim=edge_dim)
        self.Bilinear = BilinearPredictor(out_feats, out_feats_mlp)

        self.DotProduct = DotProductPredictor() # Reemplaza Bilinear
        self.regressor = nn.Linear(out_feats, 1)
        self.drop = nn.Dropout(drop)

    def encode(self, x, edge_index):
        h = self.drop(F.relu(self.conv1(x, edge_index)))
        h = self.drop(F.relu(self.conv2(h, edge_index)))
        return self.conv3(h, edge_index)

    def decodeMLP(self, edge_index, h, edge_attr=None):
        return self.MLP(edge_index, h, edge_attr)

    def decodeBilinear(self, edge_index, h):
        return self.Bilinear(edge_index, h)

    def decodeDotProduct(self, edge_index, h):
        return self.DotProduct(edge_index, h)

    def forward(self, x, edge_index):
        """Forward para regresión de atributos de nodos.
        Retorna predicciones para cada nodo.
        Para link prediction, usar: encode()
        """
        h = self.encode(x, edge_index)  # Embeddings
        return self.regressor(h)  # Predicción de atributo por nodo

class GAT3L(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, out_feats_mlp=1, num_heads=4, drop=0.3, edge_dim=0):
        super().__init__()
        # GATConv con concat=True devuelve (N, heads*out): equivale al flatten(1) de DGL.
        # Agrega self-loops por defecto (equivale a add_self_loop + GATConv).
        self.conv1 = GATConv(in_feats, hidden_feats, heads=num_heads)
        self.conv2 = GATConv(hidden_feats * num_heads, hidden_feats, heads=num_heads)
        self.conv3 = GATConv(hidden_feats * num_heads, out_feats, heads=num_heads)

        self.MLP = MLPPredictor(out_feats * num_heads, out_feats_mlp, edge_dim=edge_dim)
        self.Bilinear = BilinearPredictor(out_feats * num_heads, out_feats_mlp)
        self.DotProduct = DotProductPredictor() # Reemplaza Bilinear
        self.regressor = nn.Linear(out_feats * num_heads, 1)
        self.drop = nn.Dropout(drop)

    def encode(self, x, edge_index):
        h = self.drop(F.relu(self.conv1(x, edge_index)))
        h = self.drop(F.relu(self.conv2(h, edge_index)))
        return self.conv3(h, edge_index)

    def decodeMLP(self, edge_index, h, edge_attr=None):
        return self.MLP(edge_index, h, edge_attr)

    def decodeBilinear(self, edge_index, h):
        return self.Bilinear(edge_index, h)

    def decodeDotProduct(self, edge_index, h):
        return self.DotProduct(edge_index, h)

    def forward(self, x, edge_index):
        """Forward para regresión de atributos de nodos.
        Retorna predicciones para cada nodo.
        Para link prediction, usar: encode()
        """
        h = self.encode(x, edge_index)  # Embeddings
        return self.regressor(h)  # Predicción de atributo por nodo

# 3. Modelos de 2 Capas
class GCN2L(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, out_feats_mlp=1, drop=0.3, edge_dim=0):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, out_feats)

        self.MLP = MLPPredictor(out_feats, out_feats_mlp, edge_dim=edge_dim)
        self.Bilinear = BilinearPredictor(out_feats, out_feats_mlp)
        self.DotProduct = DotProductPredictor() # Reemplaza Bilinear
        self.drop = nn.Dropout(drop)

    def encode(self, x, edge_index):
        h = self.drop(F.relu(self.conv1(x, edge_index)))
        return self.conv2(h, edge_index)

    def decodeMLP(self, edge_index, h, edge_attr=None):
        return self.MLP(edge_index, h, edge_attr)

    def decodeBilinear(self, edge_index, h):
        return self.Bilinear(edge_index, h)

    def decodeDotProduct(self, edge_index, h):
        return self.DotProduct(edge_index, h)

    def forward(self, x, edge_index):
        """Forward para regresión de atributos de nodos.
        Retorna predicciones para cada nodo.
        Para link prediction, usar: encode()
        """
        h = self.encode(x, edge_index)  # Embeddings
        return self.regressor(h)  # Predicción de atributo por nodo

class GraphSAGE2L(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, out_feats_mlp=1, drop=0.3, edge_dim=0):
        super().__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats, aggr='mean')
        self.conv2 = SAGEConv(hidden_feats, out_feats, aggr='mean')

        self.MLP = MLPPredictor(out_feats, out_feats_mlp, edge_dim=edge_dim)
        self.Bilinear = BilinearPredictor(out_feats, out_feats_mlp)

        self.DotProduct = DotProductPredictor() # Reemplaza Bilinear
        self.drop = nn.Dropout(drop)

    def encode(self, x, edge_index):
        h = self.drop(F.relu(self.conv1(x, edge_index)))
        return self.conv2(h, edge_index)

    def decodeMLP(self, edge_index, h, edge_attr=None):
        return self.MLP(edge_index, h, edge_attr)

    def decodeBilinear(self, edge_index, h):
        return self.Bilinear(edge_index, h)

    def decodeDotProduct(self, edge_index, h):
        return self.DotProduct(edge_index, h)

    def forward(self, x, edge_index):
        """Forward para regresión de atributos de nodos.
        Retorna predicciones para cada nodo.
        Para link prediction, usar: encode()
        """
        h = self.encode(x, edge_index)  # Embeddings
        return self.regressor(h)  # Predicción de atributo por nodo

class GAT2L(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, out_feats_mlp=1, num_heads=4, drop=0.3, edge_dim=0):
        super().__init__()
        self.conv1 = GATConv(in_feats, hidden_feats, heads=num_heads)
        self.conv2 = GATConv(hidden_feats * num_heads, out_feats, heads=num_heads)

        self.MLP = MLPPredictor(out_feats * num_heads, out_feats_mlp, edge_dim=edge_dim)
        self.Bilinear = BilinearPredictor(out_feats, out_feats_mlp)

        self.DotProduct = DotProductPredictor() # Reemplaza Bilinear
        self.drop = nn.Dropout(drop)

    def encode(self, x, edge_index):
        h = self.drop(F.relu(self.conv1(x, edge_index)))
        return self.conv2(h, edge_index)

    def decodeMLP(self, edge_index, h, edge_attr=None):
        return self.MLP(edge_index, h, edge_attr)

    def decodeBilinear(self, edge_index, h):
        return self.Bilinear(edge_index, h)

    def decodeDotProduct(self, edge_index, h):
        return self.DotProduct(edge_index, h)

    def forward(self, x, edge_index):
        """Forward para regresión de atributos de nodos.
        Retorna predicciones para cada nodo.
        Para link prediction, usar: encode()
        """
        h = self.encode(x, edge_index)  # Embeddings
        return self.regressor(h)  # Predicción de atributo por nodo


################################################################################################################
## CON BATCHES
################################################################################################################

# En PyG tanto LinkNeighborLoader (Neighbor Sampling) como ClusterLoader (ClusterGCN)
# entregan un Batch con `x` y `edge_index` locales, por lo que encode() recibe
# siempre (x, edge_index): no hace falta distinguir bloques de subgrafo.


class GCNSampler2L(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, out_feats_mlp=1):
        super().__init__()
        self.conv1 = GCNConv(in_feats,  hidden_feats)
        self.conv2 = GCNConv(hidden_feats, out_feats)
        self.MLP   = MLPPredictor(out_feats, out_feats_mlp)

    def encode(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return h

    def decodeMLP(self, edge_index, h, edge_attr=None):
        return self.MLP(edge_index, h, edge_attr)

    def forward(self, x, edge_index):
        return self.encode(x, edge_index)


class GraphSAGESample2L(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, out_feats_mlp=1, aggregator='mean'):
        super().__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats, aggr=aggregator)
        self.conv2 = SAGEConv(hidden_feats, out_feats, aggr=aggregator)
        self.MLP   = MLPPredictor(out_feats, out_feats_mlp)


    def encode(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return h

    def decodeMLP(self, edge_index, h, edge_attr=None):
        return self.MLP(edge_index, h, edge_attr)

    def forward(self, x, edge_index):
        return self.encode(x, edge_index)


class GATSample2L(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, out_feats_mlp=1, num_heads=4):
        super().__init__()
        # conv1 devuelve (N, hidden*heads) (concat=True); conv2 recibe esa dimensión.
        # La salida final es (N, out*heads), igual que el flatten(1) de la versión DGL.
        self.conv1 = GATConv(in_feats, hidden_feats, heads=num_heads)
        self.conv2 = GATConv(hidden_feats * num_heads, out_feats, heads=num_heads)
        self.MLP   = MLPPredictor(out_feats * num_heads, out_feats_mlp)
        self.num_heads = num_heads

    def encode(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return h


    def decodeMLP(self, edge_index, h, edge_attr=None):
        return self.MLP(edge_index, h, edge_attr)

    def forward(self, x, edge_index):
        return self.encode(x, edge_index)
