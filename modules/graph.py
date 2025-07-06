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

    def create_graph_from_caida(self, filename:str, filename_out="edges.csv"):
        """
        Crea archivo edges.csv apartir de dataset CAIDA AS Relationships. 
        Lee un grafo desde un archivo de lista de aristas (edgelist) y lo almacena en self.graph.
        Este archivo consiste de src_id, dst_id y Relationship.
        Se ocupa luego para crear un grafo DGL.
        
        Parameters:
            filename (str): Ruta al archivo que contiene la lista de aristas.
            out_file (str): Nombre del archivo de salida.

        Returns:
            None
        """

        nx_graph = nx.DiGraph()


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

                    # Agregamos relacion C2P 
                    tor_dataset.append(np.asarray(line[1::-1])) 
                    labels.append(2)
                
                else: # P2P
                    labels.append(label)
                    tor_dataset.append(np.asarray(line[1::-1])) 
                    labels.append(label)
        
        # Creamos DataFrame
        df_edges = pd.DataFrame(tor_dataset, columns=["src_id", "dst_id"])
        df_edges['Relationship'] = labels

        # Creamoss archivo edges.csv
        df_edges.to_csv(self.data_path+"edges.csv", index=False)
        self.name_edges_file = filename_out


        # Creamoss archivo edges.csv
        self.nx_graph = nx.from_pandas_edgelist(df_edges, "src_id", "dst_id", edge_attr=["Relationship"], create_using=nx.DiGraph())
                
        if self.debug:
                print('[NX Graph]: ',self.nx_graph)

    def create_topology_from_ribs(self, rib_filename:str,filename_out="edges.csv"):

        # Creamos un grafo dirigido
        self.nx_graph = nx.DiGraph()

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

                    tor_dataset.append(np.asarray([origen, destino]))   
                    # tor_dataset.append(np.asarray([destino, origen]))  
                    

        # Creamos DataFrame
        df_edges = pd.DataFrame(tor_dataset, columns=["src_id", "dst_id"])

        # Creamoss archivo edges.csv
        df_edges.to_csv(self.data_path+"edges.csv", index=False)
        self.name_edges_file = filename_out

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
            try:
                # Filtra las filas correspondientes al nodo y obtiene los features seleccionados
                node_features = features.loc[features['ASN'] == int(node)].fillna(0).to_numpy()[0].tolist()[1:]
                
            except IndexError:
                node_features = [0] * len(features.columns[1:])  # Asignar ceros si no se encuentra el nodo

            node_features = ', '.join([str(feature) for feature in node_features]) 
            w = f'{str(node)},"{node_features}"\n'
            f.write(w)
            # Guardamos los features como diccionario para agregarlos al grafo si se desea
            attrs[node] = {'features': node_features}
        
        f.close()
        self.name_nodes_file = filename_out
        # print(f"[ARCHIVO NODES.CSV CREADO], {empty_info} nodos sin features")

        # Agregamos attrinbutos a los nodos 
        nx.set_node_attributes(self.nx_graph, attrs)

        if self.debug:
            print('[NX Graph]: ',self.nx_graph)

    def label_edges_caida(self,label_edges_file,filename_out="edges.csv"):


        nx_graph = self.nx_graph
        
        
        with bz2.open(label_edges_file, "rb") as f:
            data = f.read()
            lines = data.decode().splitlines()

            for count, line in enumerate(lines):                

                first_char = line[0]

                # Ignorar lineas comentarios
                if first_char == "#":
                    continue
            
                line = line.split("|")
                src = int(line[0])
                dst = int(line[1])
                label = int(line[2])

                if nx_graph.has_edge(src, dst):
                    if label == -1: #P2C
                        label = 1

                    nx_graph[src][dst]['Relationship'] = label

                if nx_graph.has_edge(dst, src):  
                    if label == -1:
                        label = 2
                    nx_graph[dst][src]['Relationship'] = label
                



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


    def remove_edges_with_no_label(self):
        # Paso 1: recolectar aristas sin el atributo 'Relationship'
        edges_to_remove = [
            (src, dst)
            for src, dst in self.nx_graph.edges()
            if 'Relationship' not in self.nx_graph[src][dst]
        ]

        # Paso 2: eliminarlas
        self.nx_graph.remove_edges_from(edges_to_remove)

        print(f"[INFO] Se eliminaron {len(edges_to_remove)} aristas sin 'Relationship'")


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
        for node,real_degree in dict(self.nx_graph.degree()).items():
            if real_degree <= degree:
                list_nodes_remove.append(node)
        
        self.nx_graph.remove_nodes_from(list_nodes_remove)
        
        # Creamos el nuevo edges.csv
        edge_list =  self.nx_graph.edges.data()

        # Crear DataFrame para guardar edges.csv
        df_edges = pd.DataFrame(
            [
                (u, v, data.get('Relationship', None))  # Usar .get() para manejar la ausencia del atributo
                for u, v, data in edge_list
            ],
            columns=['src_id', 'dst_id', 'Relationship']
        )


        df_edges.to_csv(self.data_path + filename_out, index=False)

        # Se lee archivo modes.csv
        df_nodes = pd.read_csv(self.data_path + self.name_nodes_file)

        # Se sacan los nodos que están en list_nodes_remove
        df_nodes_filtered = df_nodes[~df_nodes['node_id'].isin(list_nodes_remove)]

        # Se guarda el nuevo archivo nodes.csv
        df_nodes_filtered.to_csv(self.data_path + self.name_nodes_file, index=False)

        if self.debug:
            print('[NX Graph eliminando]: ',self.nx_graph) 



    def create_meta_file(self):
        
        dataset_name = os.path.basename(os.path.normpath(self.data_path))

        meta_data = {
            'dataset_name': dataset_name,
            'edge_data': [{'file_name': self.name_edges_file}],
            'node_data': [{'file_name': self.name_nodes_file}]
        }

        with open(os.path.join(self.data_path, 'meta.yaml'), 'w') as f:
            yaml.dump(meta_data, f)



def create_files(output_dir:str, 
                 rib_file:str,               
                 features_file:str='',   
                 from_caida:bool = False,   
                 label_edges_file='' , 
                 remove_degree=None, 
                 debug=False,
                 max_paths=1000):
    """
    Crea archivos nodes.csv y edges.csv para crear el dataset de un grafo en formato DGL.
    Los archivos se crean a partir de las configuraciones entregadas como parámetros.

    Parameters:
        output_dir (str): Ruta donde se guardarán los archivos generados.
        rib_file (str): Ruta al con las RIBs.
        features_file (str): Ruta al archivo de características de los nodos.
        from_caida (bool): Si es True, se crea el grafo a partir de un dataset CAIDA AS Relationships. Por defecto es False.
        label_edges_file (str): Ruta al archivo que contiene las etiquetas de las aristas. Por defecto es una cadena vacía.
        remove_degree (int, optional): Número de iteraciones para eliminar nodos de grado 1. Por defecto es None.
        debug (bool): Si es True, se imprimen mensajes de depuración. Por defecto es False.
        max_paths (int): Número máximo de rutas BGP a considerar. Por defecto es

    Returns:
        Graph: Objeto de la clase Graph con el grafo creado y configurado.

    """

    # Creamos el directorio si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('[CARPETA CREADA]: ',output_dir)

    # Creeamos instancia de la clase Graph
    graph = Graph(output_dir, max_paths ,debug=debug)

    # Creamos archivo edges.csv
    print(f"[Creando topologia desde {rib_file}]")
    if from_caida == True:
        graph.create_graph_from_caida(filename=rib_file)
    else:
        graph.create_topology_from_ribs(rib_filename=rib_file)
    
    # Creamos archivo nodes.csv
    print(f"[Agregando attr a nodos desde {features_file}]")
    if features_file == 'node_degrees': 
        # Se agregan como attr los grados in y out de los nodos
        graph.only_degree_features_nodes(filename_out="nodes.csv")
    elif features_file != '':
        # Se agregan todos los attr del archivo de atributos
        graph.features_nodes(features_filename=features_file, filename_out="nodes.csv")

    else:
        print("[No se agregan atributos a los nodos]")

        # Crear un DataFrame con los nodos
        df_nodos = pd.DataFrame({'src_id': list(graph.nx_graph.nodes())})

        # Guardar en un archivo CSV
        df_nodos.to_csv(output_dir + 'nodes.csv', index=False)
    
    if 'as-rel' in label_edges_file:
        print("[Etiquetando aristas con CAIDA]")
        graph.label_edges_caida(label_edges_file=label_edges_file, filename_out="edges.csv")
        # eliminar aristas sin label
        graph.remove_edges_with_no_label()

    


    graph.create_meta_file()
    print("[META CREADO]")
    if remove_degree:
        for i in range(remove_degree):
            graph.remove_nodes_degree(1)


    return graph
