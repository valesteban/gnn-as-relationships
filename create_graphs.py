import os
from modules.graph import *


def create_graph(graph_type, dataset_path, relationships_file, features_file, feature_list=None, remove_degree=None, debug=False):
    """
    Crea un grafo con las configuraciones dadas.

    Parameters:
        graph_type (str): Tipo de grafo a crear (DiGraph, MultiDiGraph, Graph).
        dataset_path (str): Ruta donde se guardarán los archivos generados.
        relationships_file (str): Ruta al archivo que contiene la lista de aristas.
        features_file (str): Ruta al archivo de características de los nodos.
        feature_list (list, optional): Lista de prefijos de características a incluir. Si es None, se incluyen todas las características. Por defecto es None.
        remove_degree (int, optional): Número de iteraciones para eliminar nodos de grado 1. Por defecto es None.

    Returns:
        Graph: Objeto de la clase Graph con el grafo creado y configurado.
    """

    # Creamos el directorio si no existe
    print('[PATH CREATE]',dataset_path)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    graph = Graph(dataset_path, debug=debug)

    # Creamos el grafo a partir de la lista de aristas de CAIDA AS Relationships
    graph.read_from_relationship_edgelist(relationships_file, graph_type)
    print("EDGES CREADOS ...")
    
    
    if features_file == 'node_degrees':
        # Se agregan como attr los grados in y out de los nodos
        graph.only_degree_features_nodes(filename_out="nodes.csv")
    else:
        # Se agregan todos los attr del archivo de atributos
        graph.features_nodes(features_file)
    print("FEATURES CREADOS ...")
    
    if remove_degree:
        for _ in range(remove_degree):
            graph.remove_nodes_degree(1)
    
    graph.create_meta_file()
    print("META CREADO ...")
    
    print("PROCESO COMPLETADO")

    return graph



# Definimos las rutas de los archivos
base_path = os.getcwd() + "/data/"
relationships_file = base_path + "CAIDA_AS_Relationships/Serial_2/20220701.as-rel2.txt.bz2"
features_file = base_path + "/node_features.csv"

# Definimos las listas de features
LIST_FEATURES_NO_CATEG = ['ASN', 'AS_rank_numberAsns', 'AS_rank_numberPrefixes', 'AS_rank_numberAddresses', 'AS_rank_total', 'AS_rank_customer',
                 'AS_rank_peer', 'AS_rank_provider', 'peeringDB_ix_count', 'peeringDB_fac_count', 'AS_hegemony', 'cti_top', 'cti_origin']

LIST_FEATURES_CATEG = ['AS_rank_continent',
                        'peeringDB_info_ratio',
                        'peeringDB_info_scope',
                        'peeringDB_info_type',
                        'peeringDB_policy_general'
                        'ASDB_C1L1']

list_feat = LIST_FEATURES_NO_CATEG + LIST_FEATURES_CATEG

# Casos

# Caso 1: Creación de un grafo dirigido (DiGraph)
path = base_path + "DGL_Graph/DiGraph_AllFeatures/"
create_graph("DiGraph", 
             path, 
             relationships_file, 
             features_file, 
             remove_degree=3)

# Caso 2: Creación de un grafo dirigido con multiples aristas por nodo (MultiDiGraph)
path = base_path + "DGL_Graph/MultiDiGraph_AllFeatures/"
create_graph("MultiDiGraph", 
             path, 
             relationships_file, 
             features_file, 
             remove_degree=3)

# Caso 3: Creación de un grafo dirigido (DiGraph) donde los atributos son unicamente el grado in y out de los nodos
path = base_path + "DGL_Graph/DiGraph_DegreeFeatures/"
create_graph("DiGraph", 
             path, 
             relationships_file, 
             features_file = 'node_degrees', 
             feature_list=list_feat, 
             remove_degree=3)

# Caso 4: Creación de un grafo dirigido con multiples aristas por nodo (MultiDiGraph) donde los atributos son unicamente el grado in y out de los nodos
path = base_path + "DGL_Graph/MultiDiGraph_DegreeFeatures/"
create_graph("MultiDiGraph", 
             path, 
             relationships_file, 
             features_file = 'node_degrees', 
             remove_degree=3)

