import networkx as nx
import dgl
import pandas as pd
import numpy as  np
from random import sample
import bz2
import yaml
import os

from src.utils import month_number_to_name


class Graph:
    def __init__(self,
                 dataset_graph_path: str,
                 year: str,
                 max_paths: int,
                 atrr_file_path: str = "",
                 debug:bool = False):
        """
        Constructor de la clase Graph.
        Inicializa la variable self.graph a None, que más tarde se utilizará para almacenar el grafo.
        """

        # Ruta donde se guarda dataset DGL creado
        self.data_path = dataset_graph_path

        # Año de los grafos del dataset
        self.year = year

        # Ruta a carpeta archivos atributos nodos PeeringDB
        self.attr_path = atrr_file_path

        # Cantidad RIBs para crear grafo
        self.max_paths = max_paths

        # Debug mode
        self.debug = debug
        
        # Mapeo ASN → node_id y viceversa
        self.asn_to_id = {}  # {asn: node_id}
        self.id_to_asn = {}  # {node_id: asn}
        self.next_node_id = 0  # Contador para asignar IDs secuenciales

    def get_or_create_node_id(self, asn: int) -> int:
        """Obtiene el node_id para un ASN, o crea uno nuevo si no existe.
        
        Args:
            asn: Número de Sistema Autónomo
            
        Returns:
            node_id: Identificador interno del nodo
        """
        if asn not in self.asn_to_id:
            node_id = self.next_node_id
            self.asn_to_id[asn] = node_id
            self.id_to_asn[node_id] = asn
            self.next_node_id += 1
            return node_id
        return self.asn_to_id[asn]
    
    def get_asn_from_id(self, node_id: int) -> int:
        """Obtiene el ASN correspondiente a un node_id.
        
        Args:
            node_id: Identificador interno del nodo
            
        Returns:
            asn: Número de Sistema Autónomo
        """
        return self.id_to_asn.get(node_id, None)
    
    def save_asn_mapping(self, filename="asn_mapping.csv", node_ids=None):
        """Guarda el mapeo ASN → node_id en un archivo CSV.
        
        Args:
            filename: Nombre del archivo donde guardar el mapeo
            node_ids: Iterable opcional de node_id a incluir. Si es None,
                      se guardará el mapeo completo acumulado.
        """
        ruta_completa = os.path.join(self.data_path, filename)
        
        if node_ids is None:
            # Guardar mapeo completo
            node_items = sorted(self.id_to_asn.items())
            df_mapping = pd.DataFrame({
                'node_id': [k for k, _ in node_items],
                'asn': [v for _, v in node_items]
            })
        else:
            # Guardar solo el subconjunto de node_ids provisto (p. ej. nodos de un mes)
            filtered = [(nid, self.id_to_asn.get(int(nid))) for nid in sorted(map(int, node_ids))]
            df_mapping = pd.DataFrame({
                'node_id': [k for k, _ in filtered],
                'asn': [v for _, v in filtered]
            })
        
        df_mapping = df_mapping.sort_values('node_id')
        df_mapping.to_csv(ruta_completa, index=False)
        
        if self.debug:
            print(f"[MAPEO ASN GUARDADO]: {ruta_completa}")
    
    def load_asn_mapping(self, filename="asn_mapping.csv"):
        """Carga el mapeo ASN → node_id desde un archivo CSV.
        
        Args:
            filename: Nombre del archivo desde donde cargar el mapeo
        """
        ruta_completa = os.path.join(self.data_path, filename)
        if not os.path.exists(ruta_completa):
            raise FileNotFoundError(f"Archivo de mapeo no encontrado: {ruta_completa}")
        
        df_mapping = pd.read_csv(ruta_completa)
        self.asn_to_id = dict(zip(df_mapping['asn'], df_mapping['node_id']))
        self.id_to_asn = dict(zip(df_mapping['node_id'], df_mapping['asn']))
        self.next_node_id = max(self.id_to_asn.keys()) + 1 if self.id_to_asn else 0
        
        if self.debug:
            print(f"[MAPEO ASN CARGADO]: {ruta_completa}")
            print(f"[INFO] {len(self.asn_to_id)} ASNs mapeados")


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

            edge_list = []  # Lista de aristas con node_ids
            labels = []
            
            for count, line in enumerate(lines):

                # Parar cuando se cumpla maximo de bgp paths indicado
                if count == self.max_paths:
                    break
                
                first_char = line[0]
                if first_char == "#":
                    continue
            
                line = line.split("|")

                # Obtener ASNs
                src_asn = int(line[0])
                dst_asn = int(line[1])
                label = int(line[2])
                
                # Convertir ASN a node_id
                src_id = self.get_or_create_node_id(src_asn)
                dst_id = self.get_or_create_node_id(dst_asn)

                # Add Edge con node_ids
                edge_list.append([src_id, dst_id])

                if label == -1:  # Relación Provider-Customer en CAIDA
                    # Relación P2C: src es Provider de dst
                    labels.append(2)  # P2C = 2

                    # Agregamos relacion inversa C2P: dst es Customer de src
                    edge_list.append([dst_id, src_id]) 
                    labels.append(1)  # C2P = 1
                
                else: # P2P
                    labels.append(label)
                    edge_list.append([dst_id, src_id]) 
                    labels.append(label)
        
        # Creamos DataFrame con node_ids
        df_edges = pd.DataFrame(edge_list, columns=["src_id", "dst_id"])
        df_edges['Relationship'] = labels

        # Guardamos archivo edges.csv
        df_edges.to_csv(self.data_path+"edges.csv", index=False)
        self.name_edges_file = filename_out

        # Creamos grafo NetworkX con node_ids
        self.nx_graph = nx.from_pandas_edgelist(df_edges, "src_id", "dst_id", edge_attr=["Relationship"], create_using=nx.DiGraph())
                
        if self.debug:
            print('[NX GRAPH]: ',self.nx_graph)
            print(f'[MAPEO] Total ASNs mapeados: {len(self.asn_to_id)}')

    def create_topology_from_ribs(self, graph_id: str, rib_filename:str, filename_out: str ="edges.csv"):
        print("graph_id:",  graph_id)
        # 1.- Crear carpeta si no existe
        # --------------------------
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            print(f"[CARPETA CREADA]: {self.data_path}")

        # 2.- Crear grafo nx de mes correspondiente (usando node_ids)
        # --------------------------
        nx_graph = nx.DiGraph()

        # 3.- Obtener aristas desde RIBs
        # --------------------------
        edge_list = []
        with open(rib_filename, "r") as archivo:
            for count,linea in enumerate(archivo):

                # 3.1.- Si se cumple max_paths, parar
                if count == self.max_paths:
                    break
                
                # 3.2.- Saltar líneas vacías
                if not linea.strip():
                    continue

                # 3.3.- Separar los números de AS y convertirlos a enteros
                as_list = linea.strip().split("|")
                as_list = [int(asn) for asn in as_list]

                # 3.4.- Recorrer la lista y agregar conexiones entre AS consecutivos
                for i in range(len(as_list) - 1):
                    asn_origen = as_list[i]
                    asn_destino = as_list[i + 1]
                    
                    # Convertir ASN a node_id
                    origen_id = self.get_or_create_node_id(asn_origen)
                    destino_id = self.get_or_create_node_id(asn_destino)
                    
                    # Agregar arista al grafo NetworkX (con node_ids)
                    nx_graph.add_edge(origen_id, destino_id)

                    # 3.5.- Agregar arista a la lista (con node_ids)
                    edge_list.append([graph_id, origen_id, destino_id])   
                    

        # 4.- Crear DataFrame con aristas (node_ids)
        # --------------------------
        df_edges_new = pd.DataFrame(edge_list, columns=["graph_id","src_id", "dst_id"])

        # 5.- Guardar archivo edges.csv (append si existe)
        # --------------------------
        ruta_completa = os.path.join(self.data_path, filename_out)
        
        if os.path.exists(ruta_completa):
            # Si el archivo ya existe, leer y concatenar
            df_existente = pd.read_csv(ruta_completa)
            df_edges = pd.concat([df_existente, df_edges_new], ignore_index=True)
        else:
            # Si no existe, usar el nuevo DataFrame directamente
            df_edges = df_edges_new
        
        df_edges.to_csv(ruta_completa, index=False, header=True)

        # 6.- Guardar grafo en instancia y retornar
        # --------------------------
        self.nx_graph = nx_graph
        
        # 7.- Print debug info
        # --------------------------
        if self.debug:
            print(f'[NX GRAPH]: {nx_graph}')
            print(f"[SAVE IN]: {ruta_completa}")
            print(f'[MAPEO] Total ASNs mapeados: {len(self.asn_to_id)}')
        
        return nx_graph
        

    def features_nodes(self,features_filename,filename_out="nodes.csv"):
        """
        Crea un archivo nodes.csv con los nodos id, asn y sus características.
        Ya existe nx_graph, sin attr de nodos.

        Parameters:
            features_filename (str): Ruta al archivo de características.
            filename_out (str, optional): Nombre del archivo de salida. Por defecto es "nodes.csv".

        Returns:
            None
        """
        
        # Lee el archivo de características
        features = pd.read_csv(features_filename)
        
        # Crea el archivo nodes.csv
        ruta_completa = os.path.join(self.data_path, filename_out)
        f = open(ruta_completa, "w")
        # Agrega los headers (incluyendo asn)
        headers = "node_id,asn,feat\n"
        f.write(headers)

        empty_info = 0
        # Por cada nodo en la topología, lo agrego en el archivo nodes.csv con sus features
        attrs = {}
        for node_id in self.nx_graph.nodes():
            # Obtener el ASN correspondiente al node_id
            asn = self.get_asn_from_id(node_id)
            
            try:
                # Filtra las filas correspondientes al ASN y obtiene los features seleccionados
                node_features = features.loc[features['ASN'] == int(asn)].fillna(0).to_numpy()[0].tolist()[1:]
                
            except IndexError:
                node_features = [0] * len(features.columns[1:])  # Asignar ceros si no se encuentra el nodo
                empty_info += 1

            node_features_str = ', '.join([str(feature) for feature in node_features]) 
            w = f'{node_id},{asn},"{node_features_str}"\n'
            f.write(w)
            # Guardamos los features como diccionario para agregarlos al grafo si se desea
            attrs[node_id] = {'asn': asn, 'features': node_features_str}
        
        f.close()
        self.name_nodes_file = filename_out
        if self.debug:
            print(f"[ARCHIVO NODES.CSV CREADO], {empty_info} nodos sin features")

        # Agregamos attrinbutos a los nodos 
        nx.set_node_attributes(self.nx_graph, attrs)

        if self.debug:
            print('[NX GRAPH]: ',self.nx_graph)

    def label_edges_caida(self, label_edges_file, filename_out="edges.csv"):
        """Etiqueta aristas de un grafo NetworkX existente con relaciones CAIDA.
        
        Args:
            label_edges_file: Ruta al archivo CAIDA comprimido (.bz2)
            filename_out: Nombre de archivo (no usado actualmente)
        """
        # Validar que el archivo existe
        if not os.path.exists(label_edges_file):
            raise FileNotFoundError(f"Archivo CAIDA no encontrado: {label_edges_file}")

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
                src_asn = int(line[0])
                dst_asn = int(line[1])
                label = int(line[2])
                
                # Convertir ASN a node_id
                src_id = self.asn_to_id.get(src_asn)
                dst_id = self.asn_to_id.get(dst_asn)
                
                # Solo etiquetar si ambos ASN existen en el grafo
                if src_id is None or dst_id is None:
                    continue

                if nx_graph.has_edge(src_id, dst_id):
                    if label == -1:  # Relación Provider-Customer
                        label = 2  # P2C = 2 (src es Provider de dst)

                    nx_graph[src_id][dst_id]['Relationship'] = label

                if nx_graph.has_edge(dst_id, src_id):  
                    if label == -1:  # Relación Customer-Provider
                        label = 1  # C2P = 1 (dst es Customer de src)
                    nx_graph[dst_id][src_id]['Relationship'] = label
                



    def only_degree_features_nodes(self,filename_out="nodes.csv"):
        """
        Crea un archivo nodes.csv que consiste en node_id, asn y attr correspondientes a node_degree_in y node_degree_out.

        Parameters:
            filename_out (str, optional): Nombre del archivo de salida. Por defecto es "nodes.csv".

        Returns:
            None
        """

        # Creo archivo
        ruta_completa = os.path.join(self.data_path, filename_out)
        f = open(ruta_completa, "w")
        
        # Agregamos headers (incluyendo asn)
        headers = "node_id,asn,feat\n"
        f.write(headers)

        # Por cada nodo en la topología, lo agrego en el archivo nodes.csv con sus features
        attrs = {}
        for node_id in self.nx_graph.nodes():
            # Obtener el ASN correspondiente al node_id
            asn = self.get_asn_from_id(node_id)
            
            # Obtener grados de entrada y salida
            in_degree = self.nx_graph.in_degree(node_id)
            out_degree = self.nx_graph.out_degree(node_id)
            
            # Escribir en el archivo: node_id, asn, [in_degree, out_degree]
            w = f'{node_id},{asn},"{in_degree},{out_degree}"\n'
            f.write(w)
            
            # Guardamos los features como diccionario para agregarlos al grafo si se desea
            attrs[node_id] = {'asn': asn, 'in_degree': in_degree, 'out_degree': out_degree}
        
        f.close()
        self.name_nodes_file = filename_out

        # Agregamos attrinbutos a los nodos 
        nx.set_node_attributes(self.nx_graph, attrs)

        if self.debug:
            print('[ARCHIVO NODES.CSV CREADO CON GRADOS]')
            print('[NX GRAPH]: ',self.nx_graph)
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
        ruta_completa = os.path.join(self.data_path, filename_out)
        with open(ruta_completa, "w") as f:
            # Agregamos headers
            headers = "node_id,feat\n"
            f.write(headers)

            # Iteramos sobre el DataFrame para escribir los datos
            for _, row in df.iterrows():
                node_features = f"{row['in_degree']}, {row['out_degree']}"
                f.write(f'{row["node_id"]},"{node_features}"\n')
        if self.debug:
            print(f"[SAVE IN]: {ruta_completa}")

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


        df_edges.to_csv(os.path.join(self.data_path, filename_out), index=False)

        # Se lee archivo nodes.csv
        df_nodes = pd.read_csv(os.path.join(self.data_path, self.name_nodes_file))

        # Se sacan los nodos que están en list_nodes_remove
        df_nodes_filtered = df_nodes[~df_nodes['node_id'].isin(list_nodes_remove)]

        # Se guarda el nuevo archivo nodes.csv
        df_nodes_filtered.to_csv(os.path.join(self.data_path, self.name_nodes_file), index=False)

        if self.debug:
            print('[NX GRAPH ELIMINANDO]: ',self.nx_graph) 



    def create_meta_file(self, name_edges_file: str | None = None, name_nodes_file: str | None = None):
        """
        ./mini_multi_dataset/
            |-- meta.yaml
            |-- nodes.csv
            |-- edges.csv
            |-- graphs.csv
        
        Crea archivo meta.yaml de la forma:

        
        """
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path, exist_ok=True)
            print(f"[CARPETA CREADA]: {self.data_path}")

        contenido = f"""dataset_name: graphs_{self.year}
edge_data:
- file_name: {name_edges_file}
node_data:
- file_name: {name_nodes_file}
graph_data:
  file_name: graphs.csv
"""

        # Crear el archivo
        with open(os.path.join(self.data_path, "meta.yaml"), 'w') as f:
            f.write(contenido)

        print(f"[ARCHIVO META CREADO]: {self.data_path}/meta.yaml")

    def create_graphs_file(self, num_months: int = 12):

        # Se tienen num_months cantidad de grafos con IDs 0 a num_months
        graph_ids = pd.DataFrame({'graph_id': list(range(1, num_months + 1))})

        # Guardar el archivo
        ruta_completa = os.path.join(self.data_path, "graphs.csv")
        graph_ids.to_csv(ruta_completa, index=False)

        print(f"[ARCHIVO GRAPHS.CSV CREADO]: {ruta_completa}")

    def create_edges_file(self, rib_filename: str, graph_id: str ,filename_out="edges.csv"):
  
        # Creamos un grafo dirigido
        self.nx_graph = nx.DiGraph()

        tor_dataset = []

        # Abrimos el archivo con las rutas BGP
        with open(rib_filename, "r") as archivo:
            for count, linea in enumerate(archivo):
                # Parar cuando se cumpla maximo de bgp paths indicado
                if count == self.max_paths:
                    break

                if not linea.strip():
                    continue  # Saltar líneas vacías

                # Separar los números de AS y convertirlos a enteros
                as_list = linea.strip().split("|")
                as_list = [int(asn) for asn in as_list]

                # Agregar conexiones entre AS consecutivos
                for i in range(len(as_list) - 1):
                    origen = as_list[i]
                    destino = as_list[i + 1]
                    self.nx_graph.add_edge(origen, destino)

                    # Agregar arista en ambos sentidos (si es necesario)
                    tor_dataset.append([graph_id, origen, destino])   # feat = 1.0
                    tor_dataset.append([graph_id, destino, origen])   # feat = 1.0

        # Crear DataFrame
        df_edges = pd.DataFrame(tor_dataset, columns=["graph_id", "src_id", "dst_id"])

        # Ruta de salida
        ruta_completa = os.path.join(self.data_path , filename_out)

        # Si el archivo ya existe, lo leemos y concatenamos
        if os.path.exists(ruta_completa):
            df_existente = pd.read_csv(ruta_completa)
            df_final = pd.concat([df_existente, df_edges], ignore_index=True)
        else:
            df_final = df_edges

        # Guardar el archivo actualizado
        df_final.to_csv(ruta_completa, index=False)

        print("[ARCHIVO EDGES.CSV CREADO O ACTUALIZADO]")

        if self.debug:
            print(f'[NX Graph]: {self.nx_graph} ')

        return self.nx_graph
    

    def create_peeringdb_attr(self, nx_graph ,graph_id: str,filename_out="nodes.csv"):
                        
        # 1.- Definir ruta archivo nodos
        # --------------------------
        ruta_completa = os.path.join(self.data_path, filename_out)

        # 1.1.- Si existe el archivo, se abre en modo append ("a"), si no, se crea y se escriben los headers
        write_headers = not os.path.isfile(ruta_completa)
        f = open(ruta_completa, "a")
        if write_headers:
            print(f"Creando archivo: {ruta_completa}")
            f.write("graph_id,node_id,asn,feat\n")  # ✨ Agregada columna asn

        # 2.- Leer atributos 
        # --------------------------
        features = pd.read_csv(self.attr_path+ "peeringdb/attr/" + f"peeringdb_as_attr_{self.year}_{month_number_to_name(graph_id)}.csv")
        attrs = {}

        # 2.1.- Obtener nodos únicos del grafo
        unique_nodes = list(set(nx_graph.nodes()))
        
        # 2.2.- Por cada nodo único en el grafo, buscar sus features y escribir en el archivo
        for node_id in unique_nodes:
            # ✨ Obtener ASN correspondiente al node_id
            asn = self.get_asn_from_id(node_id)
            
            try:
                # Buscar fila correspondiente al ASN (no al node_id)
                node_features = features.loc[features['ASN'] == int(asn)].fillna(0).to_numpy()[0].tolist()[1:]
            except IndexError:
                node_features = [0] * (features.shape[1] - 1)

            node_features_str = ', '.join([str(feat) for feat in node_features])
            line = f'{graph_id},{node_id},{asn},"{node_features_str}"\n'  # ✨ Incluye asn
            f.write(line)

            attrs[node_id] = {'asn': asn, 'features': node_features}  # ✨ Incluye asn

        f.close()

        # 3.- Agregar atributos al grafo si se desea
        # --------------------------
        nx.set_node_attributes(nx_graph, attrs)

        if self.debug:
            print('[ARCHIVO ACTUALIZADO]:', ruta_completa)
            print('[NX GRAPH]:', nx_graph)
        
        return nx_graph




    def create_degree_attr(self, nx_graph ,graph_id: str, filename_out="nodes.csv"):
        """
        Calcula atributos de in/out-degree log-escala y los guarda/actualiza
        en nodes.csv con formato:
            graph_id,node_id,asn,feat
        donde feat = "in_norm, out_norm".
        Usa self.nx_graph que debe haber sido creado previamente.
        """

        # 1.- Recolectar datos solo de nodos únicos
        # --------------------------
        unique_nodes = list(set(nx_graph.nodes()))
        data = []
        
        for node_id in unique_nodes:
            # Obtener ASN correspondiente
            asn = self.get_asn_from_id(node_id)
            
            # 1.1 Calcular grados
            in_deg  = nx_graph.in_degree(node_id)
            out_deg = nx_graph.out_degree(node_id)
            
            # 1.2 Normalización logarítmica (maneja la distribución de grados)
            in_log  = np.log1p(in_deg)       # log(x + 1)
            out_log = np.log1p(out_deg)

            data.append([graph_id, node_id, asn, in_log, out_log])

        # 2.- Crear DataFrame 
        # --------------------------
        df = pd.DataFrame(
            data,
            columns=["graph_id", "node_id", "asn", "in_degree", "out_degree"],
        )

        # 3.- Max-Abs scaling (normaliza a un rango común [-1, 1])
        # --------------------------
        max_val = df[["in_degree", "out_degree"]].abs().to_numpy().max()
        if max_val != 0:
            df[["in_degree", "out_degree"]] /= max_val

        # 4.- Formatear columna feat 
        # --------------------------
        df["feat"] = df.apply(lambda r: f'{r.in_degree}, {r.out_degree}', axis=1)
        df_out = df[["graph_id", "node_id", "asn", "feat"]]

        # 5) Escribir/Guardar  ------------------------------------------------
        ruta_csv = os.path.join(self.data_path, filename_out)
        write_hdr = not os.path.isfile(ruta_csv)          # ¿existe?
        mode = "a" if not write_hdr else "w"              # 'a' = append, 'w' = crear

        with open(ruta_csv, mode) as f:
            if write_hdr:
                f.write("graph_id,node_id,asn,feat\n")
            df_out.to_csv(f, header=False, index=False, lineterminator="\n")

        # 6.- Añadir atributos al grafo 
        # --------------------------
        attrs = {int(r.node_id): {"asn": r.asn, "features": [r.in_degree, r.out_degree]}
                for r in df.itertuples()}
        
        nx.set_node_attributes(nx_graph, attrs)

        if self.debug:
            print(f"[ATTRS CREADOS]: {len(attrs)} nodos")


    def create_random_attr(self, nx_graph, graph_id: str, num_features: int = 69, filename_out="nodes.csv"):
        """
        Genera atributos aleatorios para cada nodo del grafo.
        Crea num_features atributos aleatorios normalizados entre 0 y 1.
        
        Parameters:
            nx_graph: Grafo NetworkX
            graph_id: ID del grafo (mes)
            num_features: Número de features aleatorios a generar (default: 69)
            filename_out: Nombre del archivo de salida
        
        Returns:
            nx_graph con atributos agregados
        """
        
        # 1.- Definir ruta archivo nodos
        # ---------------------------------------------
        ruta_completa = os.path.join(self.data_path, filename_out)

        # 1.1.- Si existe el archivo, se abre en modo append ("a"), si no, se crea y se escriben los headers
        write_headers = not os.path.isfile(ruta_completa)
        f = open(ruta_completa, "a")
        if write_headers:
            print(f"Creando archivo: {ruta_completa}")
            f.write("graph_id,node_id,asn,feat\n")  # ✨ Agregada columna asn

        # 2.- Generar atributos aleatorios para cada nodo único
        # --------------------------
        attrs = {}
        unique_nodes = list(set(nx_graph.nodes()))
        
        for node_id in unique_nodes:
            # ✨ Obtener ASN correspondiente al node_id
            asn = self.get_asn_from_id(node_id)
            
            # Generar num_features valores aleatorios entre 0 y 1
            random_features = np.random.rand(num_features)
            
            # Convertir a string para guardar en CSV
            node_features_str = ', '.join([str(feat) for feat in random_features])
            line = f'{graph_id},{node_id},{asn},"{node_features_str}"\n'  # ✨ Incluye asn
            f.write(line)
            
            # Guardar en diccionario de atributos
            attrs[node_id] = {'asn': asn, 'features': random_features.tolist()}  # ✨ Incluye asn

        f.close()

        # 3.- Agregar atributos al grafo
        # --------------------------
        nx.set_node_attributes(nx_graph, attrs)

        if self.debug:
            print(f'[ARCHIVO ACTUALIZADO]: {ruta_completa}')
            print(f'[ATTRS ALEATORIOS CREADOS]: {len(attrs)} nodos con {num_features} features cada uno')
            print('[NX GRAPH]:', nx_graph)
        
        return nx_graph


def create_graphs_from_ribs(data_path:str, 
                            year:str, 
                            max_num_routes:int,
                            attr:str= "peeringdb",
                            caida_file:str="",):

    # 1.- Crear instancia de Graph con ruta específica según tipo de atributo
    # -------------------------- 
    graph = Graph(dataset_graph_path=data_path + f"dgl_graph/{year}/{attr}/", 
              year=year, 
              max_paths=max_num_routes, 
              atrr_file_path=data_path,
              debug=True)
    

    
    name_edges_file = "edges.csv"
    name_nodes_file = "nodes.csv"

    # 2.- Crear archivos meta.yaml 
    # --------------------------
    graph.create_meta_file(name_edges_file=name_edges_file, name_nodes_file=name_nodes_file)

    # 3.- Eliminar archivos previos si existen
    # --------------------------
    path_edges = data_path + f"dgl_graph/{year}/{attr}/edges.csv"
    path_nodes = data_path + f"dgl_graph/{year}/{attr}/nodes.csv"
    path_graphs = data_path + f"dgl_graph/{year}/{attr}/graphs.csv"

    if os.path.exists(path_edges):
        os.remove(path_edges)
        print(f"[ELIMINADO]: {path_edges}")

    if os.path.exists(path_nodes):
        os.remove(path_nodes)
        print(f"[ELIMINADO]: {path_nodes}")
        
    if os.path.exists(path_graphs):
        os.remove(path_graphs)
        print(f"[ELIMINADO]: {path_graphs}")

    # 4.- Crear archivos edges.csv y nodes.csv para cada mes
    # --------------------------
    processed_months = []  # Lista para guardar los meses que se procesaron exitosamente
    
    for number_month in range(1, 12 + 1):

        path_ribs = data_path + f"RIBs/sanitized_rib_{month_number_to_name(number_month)}_{year}.txt"

        print(f"\n[CREANDO GRAFO MES]: {month_number_to_name(number_month)}---------------------------------")

        

        # 4.1.- Ver si existe el archivo RIB
        # --------------------------
        if not os.path.exists(path_ribs):
            print("[NO EXISTE T_T]")
        else:
            print("[SI EXISTE]")
            # 4.2.- Crear/Agregar edges.csv a mes correspondiente
            # --------------------------
            print("numero:", str(number_month))
            nx_graph = graph.create_topology_from_ribs(
                rib_filename = path_ribs, 
                graph_id = str(number_month) ,
                filename_out ="edges.csv")
            
            # 4.3.-Crear/Agregar nodes.csv a mes correspondiente
            if attr == "degree":
                print("[AGREGANDO  FEATURES A NODOS]")
                graph.create_degree_attr(nx_graph,
                                    graph_id = str(number_month) ,
                                    filename_out ="nodes.csv")
            elif attr == "random":
                print("[AGREGANDO FEATURES ALEATORIOS A NODOS (69 features)]")
                graph.create_random_attr(nx_graph,
                                        graph_id = str(number_month),
                                        num_features = 69,
                                        filename_out = "nodes.csv")
            
            else:
                print("[AGREGANDO TODAS FEATURES A NODOS DESDE PEERINGDB]")
                graph.create_peeringdb_attr(nx_graph ,
                                            graph_id = str(number_month) ,
                                            filename_out="nodes.csv")
            
            # Agregar el mes a la lista de procesados
            processed_months.append(number_month)
            
            print("[GRAFO NX]", nx_graph)
            
            # Guardar snapshot del mapeo para este mes (solo nodos que aparecen en el grafo de mes)
            try:
                month_name = month_number_to_name(number_month)
                month_map_filename = f"asn_mapping_{month_name}.csv"
                # Guardamos en la misma carpeta data_path + dgl_graph/{year}/{attr}/
                graph.save_asn_mapping(filename=month_map_filename, node_ids=list(nx_graph.nodes()))
            except Exception as e:
                print(f"[WARNING] No se pudo guardar asn_mapping para {number_month}: {e}")

    # 5.- Crear archivo graphs.csv con SOLO los meses que se procesaron
    # --------------------------
    if len(processed_months) > 0:
        graph.create_graphs_file(num_months=len(processed_months))
        print(f"\n[GRAPHS.CSV CREADO] con {len(processed_months)} grafos: {processed_months}")
    else:
        print("\n[ADVERTENCIA] No se procesó ningún mes. No se creó graphs.csv")
    
    # 6.- Guardar mapeo ASN → node_id
    # --------------------------
    graph.save_asn_mapping()
    print(f"\n[MAPEO ASN GUARDADO] Total de ASNs: {len(graph.asn_to_id)}")
    
    return graph