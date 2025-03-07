import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

class DotPredictor(nn.Module):
    """
    calcula una puntuación para cada arista en el grafo utilizando el producto punto entre 
    las características de los nodos conectados por la arista.

    Métodos
    --------
    forward(g, h):
        Calcula la puntuación de cada arista en el grafo `g` mediante el producto punto
        entre las características de los nodos fuente y destino.

    Parámetros
    -----------
    g : dgl.DGLGraph
        El grafo en el que se calcularán las puntuaciones de las aristas.
    h : torch.Tensor
        Tensor que representa las características de los nodos del grafo. 
        Debe tener una dimensión compatible con el producto punto.

    Retorna
    --------
    torch.Tensor
        Un tensor con las puntuaciones de las aristas del grafo. Cada valor representa la 
        similitud entre los nodos conectados mediante el producto punto.
    """
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Calcula la puntuación de cada arista usando el producto punto entre
            # las características de los nodos de origen y destino.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v genera un vector de un solo elemento por cada arista,
            # por lo que se necesita hacer un squeeze.
            return g.edata['score'][:, 0]

class MLPPredictor(nn.Module):
    """
    Un predictor de relaciones basado en una red neuronal de perceptrón multicapa (MLP).  
    Este modelo toma los embeddings de los nodos y predice la relación entre ellos.

    Atributos:
    ----------
    W1 : nn.Linear  
        Capa lineal que transforma la concatenación de los embeddings de los nodos de entrada.  
    W2 : nn.Linear  
        Capa lineal que produce la puntuación final de la relación entre los nodos.

    Métodos:
    --------
    apply_edges(edges):  
        Aplica la red MLP a los pares de nodos conectados por aristas, calculando un puntaje para cada arista.  
    forward(g, h):  
        Propaga los embeddings a través de la red y predice los puntajes de las aristas.
    """
    def __init__(self, h_feats,out_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, out_feats)

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']
