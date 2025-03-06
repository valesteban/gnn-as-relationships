import networkx as nx
import pandas as pd
import numpy as  np
from random import sample
import bz2



class Graph:
    def __init__(self,path, debug = False):
        """
        Constructor de la clase Graph.
        Inicializa la variable self.graph a None, que más tarde se utilizará para almacenar el grafo.
        """

        self.debug = debug
        self.nx_graph = None
        
        self.dataset_dic = None

        self.path = path

    def read_from_relationship_edgelist(self, filename:str,type:str,out_file="edges.csv"):
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
            
            for line in lines:

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
        df_edges.to_csv(self.path+"edges.csv", index=False)
        if type == "DiGraph":
            self.nx_graph = nx.from_pandas_edgelist(df_edges, "src_id", "dst_id", edge_attr=["Relationship"], create_using=nx.DiGraph())
        elif type == "MultiDiGraph":
            self.nx_graph = nx.from_pandas_edgelist(df_edges, "src_id", "dst_id", edge_attr=["Relationship"], create_using=nx.MultiDiGraph())
        else:
            self.nx_graph = nx.from_pandas_edgelist(df_edges, "src_id", "dst_id", edge_attr=["Relationship"], create_using=nx.Graph())

    def features_nodes(self,features_filename, list_feat = "all",filename_out="nodes.csv"):
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
        no_cat_attr = []

        # Si se especifica una lista de features, selecciona solo esas columnas
        if list_feat != "all":

            # Crear una lista para guardar las columnas seleccionadas
            selected_columns = ['ASN']
            
            # Iterar sobre cada prefijo en list_feat (para representacion de hot hot de feat categoricos)
            for prefix in list_feat:
                # Filtrar columnas que empiezan con el prefijo
                matched_columns = [col for col in features.columns if col.startswith(prefix)]
                selected_columns.extend(matched_columns)

                # Agregamos valor no categorico (luego nomrlaizar)
                if prefix in features.columns:
                    no_cat_attr.append(prefix)
                
            # Filtrar el DataFrame para obtener solo columnas seleccionadas
            features = features[selected_columns]

        # Crea el archivo nodes.csv
        f = open(self.path + filename_out, "w")
        
        # Agrega los headers
        headers = "node_id,feat\n"
        f.write(headers)

        # Por cada nodo en la topología, lo agrego en el archivo nodes.csv con sus features
        for node in self.nx_graph.nodes():
            # Filtra las filas correspondientes al nodo y obtiene los features seleccionados
            node_features = features.loc[features['ASN'] == int(node)].fillna(0).to_numpy()[0].tolist()[1:]
            node_features = ', '.join([str(feature) for feature in node_features])
            w = f'{str(node)},"{node_features}"\n'
            f.write(w)
        
        f.close()        

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
        f = open(self.path + filename_out,"w")
        
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
        with open(self.path + filename_out, "w") as f:
            # Agregamos headers
            headers = "node_id,feat\n"
            f.write(headers)

            # Iteramos sobre el DataFrame para escribir los datos
            for _, row in df.iterrows():
                node_features = f"{row['in_degree']}, {row['out_degree']}"
                f.write(f'{row["node_id"]},"{node_features}"\n')
        if self.debug:
            print(f"[SAVE IN: {self.path+'nodes.csv'}]")


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
        df_edges = pd.DataFrame(
            [(u, v, data['Relationship']) for u, v, data in edge_list],
            columns=['src_id', 'dst_id', 'Relationship']
        )
        df_edges.to_csv(self.path + filename_out, index=False)

        if self.debug:
            print(self.nx_graph)

