from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple
import urllib.request
import sys

import requests

# Permite ejecutar este script como:
#   python3 scrpts/3_filter_by_country.py
# y aun así resolver imports de `src.*`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lacnic import *
from src.peeringdb import *
from src.topology_filter import *

def save_topology_csv(
    nodes: Dict[str, int],
    edges: Dict[Tuple[str, str], int],
    output_dir: str,
    output_prefix: str,
) -> Tuple[str, str]:
    """
    Guarda:
    - nodes_<output_prefix>.csv
    - edges_<output_prefix>.csv
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    nodes_file = output_path / f"nodes_{output_prefix}.csv"
    edges_file = output_path / f"edges_{output_prefix}.csv"

    sorted_asns = sorted(nodes.keys(), key=lambda x: int(x))
    asn_to_id = {asn: i for i, asn in enumerate(sorted_asns)}

    with open(nodes_file, "w", encoding="utf-8", newline="") as f_nodes:
        writer = csv.writer(f_nodes)
        writer.writerow(["node_id", "asn", "weight"])

        for asn in sorted_asns:
            writer.writerow([asn_to_id[asn], asn, nodes[asn]])

    with open(edges_file, "w", encoding="utf-8", newline="") as f_edges:
        writer = csv.writer(f_edges)
        writer.writerow(["src_id", "dst_id", "src_asn", "dst_asn", "weight"])

        for (src_asn, dst_asn), weight in edges.items():
            writer.writerow([
                asn_to_id[src_asn],
                asn_to_id[dst_asn],
                src_asn,
                dst_asn,
                weight
            ])

    print(f"[OK] Archivo de nodos filtrado creado: {nodes_file}")
    print(f"[OK] Archivo de aristas filtrado creado: {edges_file}")

    return str(nodes_file), str(edges_file)

def expand_one_hop_neighbors(dict_edges: dict, seed_asns: set[str]) -> set[str]:
    keep_asns = set(seed_asns)

    for (src_asn, dst_asn) in dict_edges.keys():
        if src_asn in seed_asns or dst_asn in seed_asns:
            keep_asns.add(src_asn)
            keep_asns.add(dst_asn)

    return keep_asns

def filter_country_topology(
    nodes_csv: str,
    edges_csv: str,
    lacnic_file: str,
    output_dir: str,
    country_code: str = "CL",
    include_ixp_participants: bool = True,
    include_facility_asns: bool = True,
    include_neighbors: bool = True,
) -> Tuple[str, str]:
    """
    Pipeline completo:
    1) carga topología base
    2) arma seed ASNs:
       - ASNs del país desde LACNIC
       - + ASNs en IXPs del país (opcional)
       - + ASNs en facilities del país (opcional)
    3) agrega vecinos 1-hop (opcional)
    4) guarda topología filtrada
    """
    print("-----------------------------------------------------")
    print(f"[INFO] Filtrando topología para país: {country_code}")
    print(f"[INFO] Nodes CSV: {nodes_csv}")
    print(f"[INFO] Edges CSV: {edges_csv}")
    print("-----------------------------------------------------")

    # Topología base
    nodes = load_nodes_csv(nodes_csv)
    edges = load_edges_csv(edges_csv)

    print(f"[INFO] Nodos originales: {len(nodes)}")
    print(f"[INFO] Aristas originales: {len(edges)}")

    # Semilla base: país desde LACNIC
    seed_asns = load_country_asns_from_lacnic(lacnic_file, country_code=country_code)

    # Agregar IXPs
    if include_ixp_participants:
        try:
            ixp_asns = get_ixp_participant_asns(country_code=country_code)
            seed_asns |= ixp_asns
        except Exception as e:
            print(f"[WARN] No se pudieron obtener participantes de IXPs: {e}")

    # Agregar facilities
    if include_facility_asns:
        try:
            facility_asns = get_facility_asns(country_code=country_code)
            seed_asns |= facility_asns
        except Exception as e:
            print(f"[WARN] No se pudieron obtener ASNs en facilities: {e}")

    print(f"[INFO] Total seed ASNs: {len(seed_asns)}")

    # Expandir vecinos inmediatos
    if include_neighbors:
        keep_asns = expand_one_hop_neighbors(edges, seed_asns)
        print(f"[INFO] Total ASNs luego de agregar vecinos 1-hop: {len(keep_asns)}")
    else:
        keep_asns = set(seed_asns)

    # Filtrar
    filtered_nodes, filtered_edges = filter_topology(nodes, edges, keep_asns)

    print(f"[INFO] Nodos filtrados: {len(filtered_nodes)}")
    print(f"[INFO] Aristas filtradas: {len(filtered_edges)}")

    # Nombre de salida
    # nodes_rib_20260301_0000_to_20260301_0100.csv
    # -> cl_rib_20260301_0000_to_20260301_0100
    nodes_stem = Path(nodes_csv).stem
    if nodes_stem.startswith("nodes_"):
        base_name = nodes_stem[len("nodes_"):]
    else:
        base_name = nodes_stem

    output_prefix = f"{country_code.lower()}_{base_name}"

    return save_topology_csv(
        nodes=filtered_nodes,
        edges=filtered_edges,
        output_dir=output_dir,
        output_prefix=output_prefix,
    )



if __name__ == "__main__":
    print("#######################")
    print("FILTER COUNTRY TOPOLOGY")
    print("#######################")

    data_path = "/media/vale/KINGSTON/TESIS/data_2026/"
    output_dir = data_path

    lacnic_file = str(Path(data_path) / "delegated-lacnic-extended-latest")
    country_code = "CL"

    # mismos RIBs que usaste antes
    list_ribs_files = [
        "rib_20260301_0000_to_20260301_0100.txt",
        "rib_20260301_0000_to_20260301_0200.txt",
        "rib_20260301_0000_to_20260301_0400.txt",
        "rib_20260301_0000_to_20260301_0600.txt",
        "rib_20260301_0000_to_20260301_1200.txt",
        "rib_20260301_0000_to_20260302_0000.txt",
    ]

    for rib_file in list_ribs_files:
        rib_stem = Path(rib_file).stem

        nodes_csv = str(Path(data_path) / f"nodes_{rib_stem}.csv")
        edges_csv = str(Path(data_path) / f"edges_{rib_stem}.csv")

        print("#####################################################")
        print(f"FILTRANDO TOPOLOGÍA PARA {rib_file}")
        print("#####################################################")

        if not Path(nodes_csv).exists():
            print(f"[WARN] No existe {nodes_csv}. Se omite.")
            continue

        if not Path(edges_csv).exists():
            print(f"[WARN] No existe {edges_csv}. Se omite.")
            continue

        try:
            filter_country_topology(
                nodes_csv=nodes_csv,
                edges_csv=edges_csv,
                lacnic_file=lacnic_file,
                output_dir=output_dir,
                country_code=country_code,
                include_ixp_participants=True,
                include_facility_asns=True,
                include_neighbors=True,
            )
        except Exception as e:
            print(f"[ERROR] Falló el filtrado para {rib_file}: {e}")