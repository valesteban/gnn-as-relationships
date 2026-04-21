
from collections import defaultdict


"""
Recibe archivo de texto con rutas BGP (AS_PATH) y genera dos archivos CSV:
- nodes_rib_20240701.csv: Contiene los nodos (AS) con su peso (cantidad de veces que aparece en las rutas).
- edges_rib_20240701.csv: Contiene las conexiones entre AS (aristas) con su peso (cantidad de veces que

FORMATO: Archivo RIBs de entrada:
---------------------------------------
15169|1239|701
15169|2914|3356|174

FORMATO: Archivo nodes_rib_20240701.csv
---------------------------------------
node_id | asn | |weight
---------------------------------------
0 | 15169 | 2
1 | 1239 | 1
...

FORMATO: Archivo edges_rib_20240701.csv
---------------------------------------
src_id | dst_id | src_asn | |dst_asn,weight
---------------------------------------
0 | 1 | 15169 | 1239 | 1
0 | 2 | 15169 | 701 | 1
...

"""



def create_graph_edges(rib_file: str, data_path: str):
    # Generar nombre de salida: edges_rib_20240701.csv
    edges_filename = f"edges_{rib_file.split('.')[0]}.csv"
    nodes_filename = f"nodes_{rib_file.split('.')[0]}.csv"
    
    dict_edges = defaultdict(int)
    dict_nodes = defaultdict(int)
    
    print(f"Procesando {rib_file}...")

    with open(data_path + rib_file, "r", encoding="utf-8") as archivo:
        for linea in archivo:
            as_path = linea.strip().split("|")
            
            if len(as_path) <= 1:
                continue

            for asn in as_path:
                dict_nodes[asn] += 1
            
            for i in range(len(as_path) - 1):
                as1 = as_path[i] 
                as2 = as_path[i+1]
                
                # Evitar self-loops (AS1 -> AS1)
                if as1 == as2:
                    continue
                
                # Guardar el enlace (puedes ordenar as1 y as2 si el grafo es no dirigido)
                # edge = tuple(sorted((as1, as2))) 
                dict_edges[(as1, as2)] += 1

        # Mapeao ASN - ID
        # -----------------------
        sorted_asns = sorted(dict_nodes.keys())
        asn_to_id = {asn: i for i, asn in enumerate(sorted_asns)}


        # Guardar nodos en CSV
        # -----------------------
        with open(data_path + nodes_filename, "w") as f_nodes:
            f_nodes.write("node_id,asn,weight\n")
            for asn, weight in dict_nodes.items():
                f_nodes.write(f"{asn_to_id[asn]},{asn},{weight}\n")
        
        print(f"Archivo de nodos creado: {data_path + nodes_filename}")
    
        # Guardar aristas en CSV 
        # -----------------------
        with open(data_path + edges_filename, "w") as f_edges:
            f_edges.write("src_id,dst_id,src_asn,dst_asn,weight\n")
            for (src_asn, dst_asn), weight in dict_edges.items():
                src_id = asn_to_id[src_asn]
                dst_id = asn_to_id[dst_asn]
                
                f_edges.write(f"{src_id},{dst_id},{src_asn},{dst_asn},{weight}\n")
                
        print(f"Archivo de aristas creado: {data_path + edges_filename}")
        return edges_filename
if __name__ == '__main__':




    list_ribs_files = ["rib_20260301_0000_to_20260301_0100.txt",
                       "rib_20260301_0000_to_20260301_0200.txt",
                       "rib_20260301_0000_to_20260301_0400.txt",
                       "rib_20260301_0000_to_20260301_0600.txt",
                       "rib_20260301_0000_to_20260301_1200.txt",
                        "rib_20260301_0000_to_20260302_0000.txt"]
    
    for rib_file in list_ribs_files:
        print("#####################################################")
        print(f"CREANDO ARCHIVO edges para {rib_file}")
        print("#####################################################")

        
        create_graph_edges(rib_file, data_path = "/media/vale/KINGSTON/TESIS/data_2026/")
