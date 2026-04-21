""""
1.- Importar archivo: nodes.csv y edges.csv -> Creando DataFrames de Pandas
2.- Agregar atributos a los nodos:
    - Atributos centralidad:
        - Degree centrality
        - Betweenness centrality
        - Closeness centrality
        - Eigenvector centrality
    - Atributos PeeringDB:
        - ASNs participantes en IXPs
        - ASNs con facilities
        - ASNs con presencia en múltiples países
3.- Agregar atributos a las aristas:
    - Atributos de peso:
    - AS CAIDA Relationships
4.- Guardar archivos actualizados: 
    - nodes_with_attributes.csv 
    - edges_with_attributes.csv


    

FORMATO: Archivo nodes.csv
---------------------------------------
node_id | asn | |weight
---------------------------------------
0 | 15169 | 2
1 | 1239 | 1
...

FORMATO: Archivo edges.csv
---------------------------------------
src_id | dst_id | src_asn | |dst_asn,weight
---------------------------------------
0 | 1 | 15169 | 1239 | 1
0 | 2 | 15169 | 701 | 1
...

"""
import sys
import os
import bz2

# Directorio raíz del proyecto al path de Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import json
import networkx as nx
import numpy as np
from collections import defaultdict
from src.peeringdb import download_peeringdb_file

def _minmax_scale(series: pd.Series) -> pd.Series:
    """Escala una serie a [0,1]. Si no hay variación, retorna ceros."""
    s = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(np.float64)
    min_v = s.min()
    max_v = s.max()
    if max_v - min_v == 0:
        return pd.Series(np.zeros(len(s), dtype=np.float32), index=s.index)
    scaled = (s - min_v) / (max_v - min_v)
    # clip antes del cast a float32 para evitar que el redondeo supere 1.0
    return scaled.clip(0.0, 1.0).astype(np.float32)

def _log_minmax_scale(series: pd.Series) -> pd.Series:
    """Aplica log(x+1) para reducir variabilidad, luego escala a [0,1]."""
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    log_s = np.log1p(s.clip(lower=0))
    return _minmax_scale(log_s)

def _check_range(col_name: str, series: pd.Series) -> None:
    """Lanza un error si algún valor está fuera de [0, 1]."""
    lo, hi = float(series.min()), float(series.max())
    if lo < 0.0 or hi > 1.0:
        raise ValueError(
            f"[NORM ERROR] '{col_name}' fuera de [0,1]: min={lo:.6f}, max={hi:.6f}"
        )

def add_atributes_nodes(nodes_csv: str, edges_csv: str, year: str, month: str, output_dir: str):

    # 1.- Cargar datos: nodos y aristas
    ###################################
    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)
    # Snapshot para no tocar atributos previos del usuario
    original_node_cols = set(nodes_df.columns)

    # 2.- Agregar atributos de centralidad
    ##################################
    edge_weight = 'weight' if 'weight' in edges_df.columns else None
    G = nx.from_pandas_edgelist(
        edges_df,
        source='src_id',
        target='dst_id',
        edge_attr=edge_weight,
        create_using=nx.DiGraph()
    )

    # Nota: se calculan sobre node_id para mantener consistencia con el mapeo interno.
    if 'node_id' in nodes_df.columns:
        print("[1/3] Calculando centralidades de nodos...")

        # 1) Degree centrality — log(x+1) + min-max [0,1]
        deg = nx.degree_centrality(G)
        raw = nodes_df['node_id'].map(deg).fillna(0.0)
        nodes_df['degree_centrality'] = _log_minmax_scale(raw)
        _check_range('degree_centrality', nodes_df['degree_centrality'])

        # 2) Betweenness centrality — log(x+1) + min-max [0,1]
        n_nodes = max(1, G.number_of_nodes())
        k_sample = min(100, n_nodes)
        try:
            bet = nx.betweenness_centrality(G, k=k_sample, seed=42)
        except TypeError:
            bet = nx.betweenness_centrality(G, k=k_sample)
        raw = nodes_df['node_id'].map(bet).fillna(0.0)
        nodes_df['betweenness_centrality'] = _log_minmax_scale(raw)
        _check_range('betweenness_centrality', nodes_df['betweenness_centrality'])

        # 3) Eigenvector centrality — log(x+1) + min-max [0,1]
        eig = {}
        try:
            eig = nx.eigenvector_centrality(G, max_iter=1000, weight=edge_weight)
        except Exception:
            try:
                eig = nx.eigenvector_centrality(G.to_undirected(), max_iter=1000, weight=edge_weight)
            except Exception:
                print("[WARN] eigenvector_centrality no convergió; se asignan ceros.")
                eig = {}
        raw = nodes_df['node_id'].map(eig).fillna(0.0)
        nodes_df['eigenvector_centrality'] = _log_minmax_scale(raw)
        _check_range('eigenvector_centrality', nodes_df['eigenvector_centrality'])

        # 4) PageRank — log(x+1) + min-max [0,1]
        try:
            pr = nx.pagerank(G, weight=edge_weight)
        except Exception:
            pr = nx.pagerank(G)
        raw = nodes_df['node_id'].map(pr).fillna(0.0)
        nodes_df['PageRank'] = _log_minmax_scale(raw)
        _check_range('PageRank', nodes_df['PageRank'])

        created_degree_cols = [
            'degree_centrality',
            'betweenness_centrality',
            'eigenvector_centrality',
            'PageRank'
        ]
    else:
        created_degree_cols = []
        print("[WARN] No existe columna node_id; se omiten centralidades.")

    # 3.-Procesar datos de PEERINGDB para agregar atributos a los nodos
    ###################################
    pdb_path = os.path.join(output_dir, f"peeringdb_2_dump_{year}_{month}_01.json")
    if not os.path.exists(pdb_path):
        print(f"[INFO] Descargando PeeringDB...")
        download_peeringdb_file(pdb_path, year=year, month=month)

    with open(pdb_path, "r", encoding="utf-8") as f:
        pdb_raw = json.load(f)

    net_df = pd.DataFrame(pdb_raw['net']['data']).set_index('asn')

    # Normalizar tipo de ASN para evitar mapeos fallidos
    nodes_df['asn'] = pd.to_numeric(nodes_df['asn'], errors='coerce')
    net_df.index = pd.to_numeric(net_df.index, errors='coerce')

    print(f"[2/3] Extrayendo atributos de PeeringDB...")
    print("[INFO] Se mantiene mapeo node_id != asn (atributos por ASN, IDs internos por node_id).")

    # 4.- Atributos Numéricos PeeringDB: log(x+1) + min-max [0,1]
    num_cols = ['ix_count', 'fac_count', 'info_prefixes4', 'info_prefixes6']
    created_numeric_cols = []
    created_final_num_cols = []

    for col in num_cols:
        if col in net_df.columns:
            vals = pd.to_numeric(nodes_df['asn'].map(net_df[col]), errors='coerce').fillna(0.0)
            nodes_df[col] = _log_minmax_scale(vals)
            _check_range(col, nodes_df[col])
            created_numeric_cols.append(col)
            if col not in original_node_cols:
                created_final_num_cols.append(col)

    # 4.1.- Normalizar weight de nodos: log(x+1) + min-max [0,1]
    if 'weight' in nodes_df.columns:
        nodes_df['weight'] = _log_minmax_scale(nodes_df['weight'])
        _check_range('weight', nodes_df['weight'])
        created_numeric_cols.append('weight')
        if 'weight' not in original_node_cols:
            created_final_num_cols.append('weight')

    # 5.- Atributos Categóricos: Transforma a One-Hot vectors
    cat_cols = [
        'policy_general',
        'policy_locations',
        'policy_contracts',
        'info_traffic',
        'info_scope',
        'info_type',
        'info_ratio'
    ]
    created_onehot_cols = []
    created_new_onehot_cols = []
    for col in cat_cols:
        if col in net_df.columns:
            s_cat = nodes_df['asn'].map(net_df[col])
            # Limpiar valores vacíos y faltantes
            s_cat = s_cat.astype('string').str.strip().replace({'': 'Unknown'}).fillna('Unknown')
            # One-hot explícito en 0/1
            dummies = pd.get_dummies(s_cat, prefix=col, dtype=np.uint8)
            for dcol in dummies.columns:
                if dcol in nodes_df.columns:
                    # Si ya existe (atributo previo), conservarlo y fusionar por OR lógico
                    old_vals = pd.to_numeric(nodes_df[dcol], errors='coerce').fillna(0).astype(np.uint8)
                    nodes_df[dcol] = np.maximum(old_vals, dummies[dcol].astype(np.uint8))
                else:
                    nodes_df[dcol] = dummies[dcol].astype(np.uint8)
                    if dcol not in original_node_cols:
                        created_new_onehot_cols.append(dcol)
                created_onehot_cols.append(dcol)


    # Integrar centralidades al conteo de columnas numéricas
    if created_degree_cols:
        created_numeric_cols.extend(created_degree_cols)
        created_final_num_cols.extend([c for c in created_degree_cols if c not in original_node_cols])

    print(f"[INFO] Columnas numéricas agregadas/actualizadas: {len(set(created_numeric_cols))}")
    print(f"[INFO] Columnas one-hot agregadas/actualizadas: {len(set(created_onehot_cols))}")

    # 7.- Limpieza: eliminar columnas nuevas sin información (varianza 0)
    dropped_cols = []
    zero_var_candidates = set(created_final_num_cols + created_new_onehot_cols)
    zero_var_cols = []
    for col in sorted(zero_var_candidates):
        if col not in nodes_df.columns:
            continue
        s = nodes_df[col]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            if s.nunique(dropna=False) <= 1:
                zero_var_cols.append(col)
    if zero_var_cols:
        nodes_df = nodes_df.drop(columns=zero_var_cols, errors='ignore')
        dropped_cols.extend(zero_var_cols)

    if dropped_cols:
        print(f"[INFO] Columnas eliminadas post-normalización: {len(set(dropped_cols))}")
    else:
        print("[INFO] No se eliminaron columnas en limpieza post-normalización.")

    # 8.- Guardar Nodos (sin eliminar atributos previos)
    #################################
    out_nodes = nodes_csv.replace(".csv", "_enriched_tesis.csv")
    nodes_df.to_csv(out_nodes, index=False)
    print(f"[OK] Nodos guardados en: {out_nodes}")
    return nodes_df

def add_atributes_edges(edges_csv: str, caida_rel_path: str = None, output_dir: str = None):
    """
    Formato CAIDA[cite: 89]:
    <provider-as>|<customer-as>|-1
    <peer-as>|<peer-as>|0

    Formato tesis (columna relationship):
    0 = P2P
    1 = C2P
    2 = P2C
   -1 = Sin etiqueta / desconocido
    """
    # 1.- Cargar aristas
    ##################################
    edges_df = pd.read_csv(edges_csv)

    # 2.- Resolver ruta del archivo de relaciones CAIDA:
    #   - Acepta ruta absoluta
    #   - Acepta ruta relativa (con respecto a output_dir o cwd)
    #   - Si no existe, deja relaciones como -1 (sin etiqueta)
    caida_file = None
    if caida_rel_path:
        candidates = []
        if os.path.isabs(caida_rel_path):
            candidates.append(caida_rel_path)
        else:
            candidates.append(caida_rel_path)
            if output_dir:
                candidates.append(os.path.join(output_dir, caida_rel_path))

        for path in candidates:
            if os.path.exists(path):
                caida_file = path
                break

    if caida_file:
        print(f"[INFO] Usando archivo CAIDA: {caida_file}")

        # Formato CAIDA: provider|customer|-1 o peer|peer|0
        # Mapeo tesis: 0=P2P, 1=C2P, 2=P2C, -1=desconocido
        caida_data = []
        opener = bz2.open if caida_file.endswith(".bz2") else open

        with opener(caida_file, 'rt', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                p = line.split('|')
                if len(p) < 3:
                    continue

                try:
                    as1, as2, rel = int(p[0]), int(p[1]), int(p[2])
                except ValueError:
                    continue

                if rel == -1:
                    # as1(provider) -> as2(customer) = P2C(2)
                    caida_data.append({'src_asn': as1, 'dst_asn': as2, 'relationship': 2})
                    # inversa: customer -> provider = C2P(1)
                    caida_data.append({'src_asn': as2, 'dst_asn': as1, 'relationship': 1})
                elif rel == 0:
                    # P2P es simétrica en ambas direcciones
                    caida_data.append({'src_asn': as1, 'dst_asn': as2, 'relationship': 0})
                    caida_data.append({'src_asn': as2, 'dst_asn': as1, 'relationship': 0})
                else:
                    # Etiqueta inesperada en CAIDA, se ignora
                    continue

        rel_df = pd.DataFrame(caida_data)
        rel_df = rel_df.drop_duplicates(subset=['src_asn', 'dst_asn'], keep='first')
        rel_df = rel_df.rename(columns={'relationship': 'relationship_caida'})
        edges_df = edges_df.merge(rel_df, on=['src_asn', 'dst_asn'], how='left')

        caida_rel = pd.to_numeric(edges_df['relationship_caida'], errors='coerce')
        if 'relationship' in edges_df.columns:
            # Mantener relaciones ya calculadas y solo completar faltantes/desconocidas
            prev_rel = pd.to_numeric(edges_df['relationship'], errors='coerce')
            fill_mask = prev_rel.isna() | (prev_rel == -1) | (~prev_rel.isin([0, 1, 2, -1]))
            prev_rel.loc[fill_mask] = caida_rel.loc[fill_mask]
            edges_df['relationship'] = prev_rel.fillna(-1).astype('int8')
        else:
            edges_df['relationship'] = caida_rel.fillna(-1).astype('int8')

        edges_df = edges_df.drop(columns=['relationship_caida'], errors='ignore')
    else:
        if 'relationship' in edges_df.columns:
            print("[WARN] No se encontró archivo CAIDA, se conserva relationship existente.")
            edges_df['relationship'] = pd.to_numeric(edges_df['relationship'], errors='coerce').fillna(-1).astype('int8')
        else:
            print("[WARN] No se encontró archivo CAIDA, se asigna relationship=-1 (sin etiqueta).")
            edges_df['relationship'] = -1

    # 3.- Garantizar arista inversa para relaciones conocidas:
    #     0 (P2P) -> 0
    #     1 (C2P) -> 2 (P2C)
    #     2 (P2C) -> 1 (C2P)
    #     -1 se mantiene como desconocida (no se fuerza inversa)
    rel_num = pd.to_numeric(edges_df['relationship'], errors='coerce').fillna(-1).astype('int16')
    edges_df['relationship'] = rel_num

    known_mask = edges_df['relationship'].isin([0, 1, 2])
    reverse_df = edges_df.loc[known_mask].copy()

    # Intercambiar orientación de la arista (por ID y ASN si están disponibles)
    if {'src_id', 'dst_id'}.issubset(reverse_df.columns):
        reverse_df[['src_id', 'dst_id']] = reverse_df[['dst_id', 'src_id']].to_numpy()
    if {'src_asn', 'dst_asn'}.issubset(reverse_df.columns):
        reverse_df[['src_asn', 'dst_asn']] = reverse_df[['dst_asn', 'src_asn']].to_numpy()

    reverse_df['relationship'] = reverse_df['relationship'].map({0: 0, 1: 2, 2: 1}).astype('int16')

    # Definir clave de arista para deduplicar sin perder filas ya existentes
    if {'src_id', 'dst_id'}.issubset(edges_df.columns):
        edge_key_cols = ['src_id', 'dst_id']
    elif {'src_asn', 'dst_asn'}.issubset(edges_df.columns):
        edge_key_cols = ['src_asn', 'dst_asn']
    else:
        edge_key_cols = None

    if edge_key_cols is not None:
        existing_keys = set(map(tuple, edges_df[edge_key_cols].to_numpy()))
        reverse_keys = set(map(tuple, reverse_df[edge_key_cols].to_numpy()))

        # Conteo informativo: cuántas aristas inversas faltaban
        missing_reverse = reverse_keys - existing_keys

        combined = pd.concat(
            [
                edges_df.assign(_src_priority=0),
                reverse_df.assign(_src_priority=1)
            ],
            ignore_index=True
        )
        combined['_rel_known'] = combined['relationship'].isin([0, 1, 2]).astype('int8')

        # Priorización:
        # 1) mantener relaciones conocidas (0/1/2) sobre desconocidas
        # 2) ante empate, conservar fila original antes que la generada
        combined = combined.sort_values(
            by=edge_key_cols + ['_rel_known', '_src_priority'],
            ascending=[True] * len(edge_key_cols) + [False, True]
        )

        edges_df = combined.drop_duplicates(subset=edge_key_cols, keep='first')
        edges_df = edges_df.drop(columns=['_src_priority', '_rel_known'], errors='ignore')

        print(f"[INFO] Aristas inversas agregadas: {len(missing_reverse)}")
    else:
        # Fallback conservador si no hay columnas para identificar orientación
        print("[WARN] No se encontraron columnas src/dst para forzar aristas inversas.")

    edges_df['relationship'] = pd.to_numeric(edges_df['relationship'], errors='coerce').fillna(-1).astype('int8')

    # 4.- Normalizar weight de aristas: log(x+1) + min-max [0,1]
    if 'weight' in edges_df.columns:
        edges_df['weight'] = _log_minmax_scale(edges_df['weight']).astype(np.float32)

    out_edges = edges_csv.replace(".csv", "_enriched_tesis.csv")
    edges_df.to_csv(out_edges, index=False)
    print(f"[OK] Aristas guardadas en: {out_edges}")




if __name__ == '__main__':

    output_dir = "/media/vale/KINGSTON/TESIS/data_2026/"
    nodes_csv = output_dir + "nodes_rib_20260301_0000_to_20260302_0000.csv"
    edges_csv = output_dir + "edges_rib_20260301_0000_to_20260302_0000.csv"

    peeringdb_file = "/media/vale/KINGSTON/TESIS/data_2026/peeringdb_2_dump_2026_03_01.json"
    year = "2026"
    month = "03"

    caida_relationships_file = "/media/vale/KINGSTON/TESIS/data_2026/20260301.as-rel.txt.bz2"
        

    print("#####################################################")
    print(f"AGREGANDO ATRIBUTOS ")
    print("#####################################################")


    # 1.- Agregar atributos a nodos
    add_atributes_nodes(nodes_csv,
                        edges_csv, 
                        year, 
                        month, 
                        output_dir
                        )

    # 2.- Agregar atributos a aristas
    add_atributes_edges(edges_csv, caida_relationships_file,output_dir)
   



