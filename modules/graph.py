import networkx as nx
import dgl
import pandas as pd
import numpy as  np
from random import sample
import bz2
import yaml
import os




class Graph:
    def __init__(self,dataset_graph_path ,max_paths,debug = False):
        """
        Constructor de la clase Graph.
        Inicializa la variable self.graph a None, que más tarde se utilizará para almacenar el grafo.
        """

        self.debug = debug

        self.nx_graph = None

        self.data_path = dataset_graph_path

        name_edges_file = ""
        name_nodes_file = ""
        self.max_paths = max_paths

    def create_graph_from_caida(self, filename:str,type:str, filename_out="edges.csv"):
        """
        Crea archivo edges.csv apartir de dataset CAIDA AS Relationships. 
        Lee un grafo desde un archivo de lista de aristas (edgelist) y lo almacena en self.graph.
        Este archivo consiste de src_id, dst_id y Relationship.
        Se ocupa luego para crear un grafo DGL.
        
        Parameters:
            filename (str): Ruta al archivo que contiene la lista de aristas.
            type (str): Tipo de grafo a crear (DiGraph, MultiDiGraph, Graph). 
            out_file (str): Nombre del archivo de salida.

        Returns:
            None
        """

        nx_graph = nx.Graph()


        with bz2.open(filename, "rb") as f:
            data = f.read()
            lines = data.decode().splitlines()

            tor_dataset = []
            labels = []
            
            for count, line in enumerate(lines):

                # Parar cuando se cumpla maximo de bgp paths indicado
                if count == self.max_paths:
                    break
                

                first_char = line[0]

                if first_char == "#":
                    continue
            
                line = line.split("|")

                # for nx
                src = int(line[0])
                dst = int(line[1])
                label = int(line[2])

                # Add Edge
                tor_dataset.append(np.asarray(line[:2]))      #seleccionamos los dos primeros elementos de la lista          

                if label == -1: #P2C 
                    # Cambiamos valor label a 1
                    labels.append(1)

                    if type == "MultiDiGraph":
                        # Agregamos relacion C2P 
                        tor_dataset.append(np.asarray(line[1::-1])) 
                        labels.append(2)
                
                else: # P2P
                    labels.append(label)

                    if type == "MultiDiGraph":
                        tor_dataset.append(np.asarray(line[1::-1])) 
                        labels.append(label)
        
        # Creamos DataFrame
        df_edges = pd.DataFrame(tor_dataset, columns=["src_id", "dst_id"])
        df_edges['Relationship'] = labels

        # Creamoss archivo edges.csv
        df_edges.to_csv(self.data_path+"edges.csv", index=False)
        self.name_edges_file = filename_out


        # Creamoss archivo edges.csv
        if type == "DiGraph":
            self.nx_graph = nx.from_pandas_edgelist(df_edges, "src_id", "dst_id", edge_attr=["Relationship"], create_using=nx.DiGraph())
        elif type == "MultiDiGraph":
            self.nx_graph = nx.from_pandas_edgelist(df_edges, "src_id", "dst_id", edge_attr=["Relationship"], create_using=nx.MultiDiGraph())
        else:
            self.nx_graph = nx.from_pandas_edgelist(df_edges, "src_id", "dst_id", edge_attr=["Relationship"], create_using=nx.Graph())
                
        if self.debug:
                print('[NX Graph]: ',self.nx_graph)

    def create_topology_from_ribs(self, rib_filename:str,type:str ,filename_out="edges.csv"):

        # Creamos un grafo dirigido
        if type == "DiGraph":
            self.nx_graph = nx.DiGraph()
        elif type == "MultiDiGraph":
            self.nx_graph = nx.MultiDiGraph()

        tor_dataset = []

        # Abrimos el archivo con las rutas BGP
        with open(rib_filename, "r") as archivo:
            for count,linea in enumerate(archivo):

                # Parar cuando se cumpla maximo de bgp paths indicado
                if count == self.max_paths:
                    break
                
                # Saltar líneas vacías
                if not linea.strip():
                    continue

                # Separar los números de AS y convertirlos a enteros
                as_list = linea.strip().split("|")
                as_list = [int(asn) for asn in as_list]

                # Recorrer la lista y agregar conexiones entre AS consecutivos
                for i in range(len(as_list) - 1):
                    origen = as_list[i]
                    destino = as_list[i + 1]
                    self.nx_graph.add_edge(origen, destino)
                    if type == "DiGraph":
                        tor_dataset.append(np.asarray([origen, destino]))
                    elif type == "MultiDiGraph":
                        tor_dataset.append(np.asarray([origen, destino]))
                        tor_dataset.append(np.asarray([destino, origen]))

        # Creamos DataFrame
        df_edges = pd.DataFrame(tor_dataset, columns=["src_id", "dst_id"])

        # Creamoss archivo edges.csv
        df_edges.to_csv(self.data_path+"edges.csv", index=False)
        self.name_edges_file = filename_out
        print("[ARCHIVO EDGES.CSV CREADO]")

        if self.debug:
                    print('[NX Graph]: ',self.nx_graph)
        

    def features_nodes(self,features_filename,filename_out="nodes.csv"):
        """
        Crea un archivo nodes.csv con los nodos id y sus características.
        Si no se especifica una lista de características, se incluyen todas.
        Ya existe nx_graph, sin attr de nodos.

        Parameters:
            features_filename (str): Ruta al archivo de características.
            list_feat (list or str, optional): Lista de prefijos de características a incluir. Si es "all", se incluyen todas las características. Por defecto es "all".
            normalize (bool, optional): Indica si se deben normalizar los valores no categóricos. Por defecto es False.
            filename_out (str, optional): Nombre del archivo de salida. Por defecto es "nodes.csv".

        Returns:
            None
        """
        
        # Lee el archivo de características
        features = pd.read_csv(features_filename)
        
        # Crea el archivo nodes.csv
        f = open(self.data_path + filename_out, "w")
        # Agrega los headers
        headers = "node_id,feat\n"
        f.write(headers)

        empty_info = 0
        # Por cada nodo en la topología, lo agrego en el archivo nodes.csv con sus features
        attrs = {}
        for node in self.nx_graph.nodes():
<<<<<<< HEAD
            try:
                # Filtra las filas correspondientes al nodo y obtiene los features seleccionados
                node_features = features.loc[features['ASN'] == int(node)].fillna(0).to_numpy()[0].tolist()[1:]
                
            except IndexError:
                node_features = [0] * len(features.columns[1:])  # Asignar ceros si no se encuentra el nodo
=======

            # Obtener la fila del nodo
            fila = features.loc[features['ASN'] == int(node)]
            
            # Si no se encuentra rellena con 0
            if fila.empty:
                empty_info += 1
                num_features = features.shape[1] - 1  # -1 porque la primera columna es ASN
                node_features = '0, ' [0] * num_features
            else:
                node_features = features.loc[features['ASN'] == int(node)].fillna(0).to_numpy()[0].tolist()[1:]
>>>>>>> dd938f7dad20d9d83f29b8a3e6b2679fecd09c32

            node_features = ', '.join([str(feature) for feature in node_features]) 
            w = f'{str(node)},"{node_features}"\n'
            f.write(w)
            # Guardamos los features como diccionario para agregarlos al grafo si se desea
            attrs[node] = {'features': node_features}
        
        f.close()
        self.name_nodes_file = filename_out
        print(f"[ARCHIVO NODES.CSV CREADO], {empty_info} nodos sin features")

        # Agregamos attrinbutos a los nodos 
        nx.set_node_attributes(self.nx_graph, attrs)

        if self.debug:
            print('[NX Graph]: ',self.nx_graph)

    def only_degree_features_nodes(self,filename_out="nodes.csv"):
        """
        Crea un archivo nodes.csv que consiste en node_id y attr correspondientes a node_degree_in y npde_degree_out.

        Parameters:
            features_filename (str): Ruta al archivo de características.
            filename_out (str, optional): Nombre del archivo de salida. Por defecto es "nodes.csv".

        Returns:
            None
        """

        # Creo archivo
        f = open(self.data_path + filename_out,"w")
        
        # Agregamos headers
        headers = "node_id,feat\n"
        f.write(headers)

        # Crear una lista para almacenar los datos
        data = []

        for node in self.nx_graph.nodes():
            in_degree = self.nx_graph.in_degree(node)
            out_degree = self.nx_graph.out_degree(node)

            # Normalizo los valores de los grados

            # Normalizacion paso 1: Transformación logarítmica 𝑥 → log(𝑥 + 1).
            in_degree_log = np.log(in_degree + 1)
            out_degree_log = np.log(out_degree + 1)

            data.append([node, in_degree_log, out_degree_log])

        # Convertimos la lista en un DataFrame para facilitar la normalización
        df = pd.DataFrame(data, columns=['node_id', 'in_degree', 'out_degree'])
        
        # Normalizacion paso 2: Normalización Max Abs Scaling
        max_abs_value = df[['in_degree', 'out_degree']].abs().max().max()
        df['in_degree'] = df['in_degree'] / max_abs_value
        df['out_degree'] = df['out_degree'] / max_abs_value

        # Guardar datos normalizados en nodes.csv
        with open(self.data_path + filename_out, "w") as f:
            # Agregamos headers
            headers = "node_id,feat\n"
            f.write(headers)

            # Iteramos sobre el DataFrame para escribir los datos
            for _, row in df.iterrows():
                node_features = f"{row['in_degree']}, {row['out_degree']}"
                f.write(f'{row["node_id"]},"{node_features}"\n')
        if self.debug:
            print(f"[SAVE IN: {self.data_path+'nodes.csv'}]")

        # Guardamos nombre de archivo de nodos
        self.name_nodes_file = filename_out

    def remove_nodes_degree(self, degree,filename_out="edges.csv"):
        """
        Elimina nodos de grado menor o igual a 'degree' del grafo y crea un nuevo archivo edges.csv con las aristas restantes.

        Parameters:
            degree (int): Grado máximo de los nodos a eliminar.
            filename_out (str, optional): Nombre del archivo de salida. Por defecto es "edges.csv".

        Returns:
            None
        """
        list_nodes_remove = []
        for node,deg in dict(self.nx_graph.degree()).items():
            if deg <= degree:
                list_nodes_remove.append(node)
        
        self.nx_graph.remove_nodes_from(list_nodes_remove)
        
        # Creamos el nuevo edges.csv
        edge_list =  self.nx_graph.edges.data()

        # Crear DataFrame con manejo de casos sin atributos de aristas
        df_edges = pd.DataFrame(
            [
                (u, v, data.get('Relationship', None))  # Usar .get() para manejar la ausencia del atributo
                for u, v, data in edge_list
            ],
            columns=['src_id', 'dst_id', 'Relationship']
        )
        df_edges.to_csv(self.data_path + filename_out, index=False)

        if self.debug:
            print('[NX Graph]: ',self.nx_graph)



    def create_meta_file(self):
        """
        Crea un archivo meta.yaml con la información del grafo.

        Parameters:
            graph_type (str): Tipo de grafo (DiGraph, MultiDiGraph).
            features_type (str): Tipo de características (AllFeatures, DegreeFeatures).

        Returns:
            None
        """
        
        dataset_name = os.path.basename(os.path.normpath(self.data_path))

        meta_data = {
            'dataset_name': dataset_name,
            'edge_data': [{'file_name': self.name_edges_file}],
            'node_data': [{'file_name': self.name_nodes_file}]
        }

        with open(os.path.join(self.data_path, 'meta.yaml'), 'w') as f:
            yaml.dump(meta_data, f)



def create_files(graph_type:str, dataset_graph_path:str,file:str, features_file:str='', from_caida:bool = False, feature_list=None, remove_degree=None, debug=False,max_paths=1000):
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
    if not os.path.exists(dataset_graph_path):
        os.makedirs(dataset_graph_path)
    print('[CaRPETA CREADA]: ',dataset_graph_path)


    graph = Graph(dataset_graph_path, max_paths ,debug=debug)

    print("[Creando topologia]")
    if from_caida == True:
        graph.create_graph_from_caida(filename=file,type=graph_type)
    else:
        graph.create_topology_from_ribs(rib_filename=file, type=graph_type)
    

    print("[Agregando attr]")
    if features_file == 'node_degrees': 
        print("[solo grados]")
        # Se agregan como attr los grados in y out de los nodos
        graph.only_degree_features_nodes(filename_out="nodes.csv")
    elif features_file != '':
        # Se agregan todos los attr del archivo de atributos
<<<<<<< HEAD
        graph.features_nodes(features_filename=features_file, filename_out="nodes.csv")
        print(graph.name_nodes_file,   graph.name_edges_file)
=======
        graph.features_nodes(features_file)
>>>>>>> dd938f7dad20d9d83f29b8a3e6b2679fecd09c32
    else:

        # Crear un DataFrame con los nodos
        df_nodos = pd.DataFrame({'src_id': list(graph.nx_graph.nodes())})

        # Guardar en un archivo CSV
        df_nodos.to_csv(dataset_graph_path+'nodes.csv', index=False)
    
    graph.create_meta_file()
    print("[META CREADO]")
    
    if remove_degree:
        for _ in range(remove_degree):
            graph.remove_nodes_degree(1)

    return graph
