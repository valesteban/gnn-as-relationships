from typing import Dict, Set, Tuple
from collections import defaultdict
import csv

def load_nodes_csv(nodes_csv: str) -> Dict[str, int]:
    """
    Lee nodes_*.csv y retorna:
    {asn: weight}
    """
    nodes: Dict[str, int] = {}

    with open(nodes_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            asn = row["asn"].strip()
            weight = int(row["weight"])
            nodes[asn] = weight

    return nodes


def load_edges_csv(edges_csv: str) -> Dict[Tuple[str, str], int]:
    """
    Lee edges_*.csv y retorna:
    {(src_asn, dst_asn): weight}

    Si accidentalmente hubiera duplicados en el CSV, los acumula.
    """
    edges: Dict[Tuple[str, str], int] = defaultdict(int)

    with open(edges_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            src_asn = row["src_asn"].strip()
            dst_asn = row["dst_asn"].strip()
            weight = int(row["weight"])

            edges[(src_asn, dst_asn)] += weight

    return dict(edges)

def filter_topology(dict_nodes: dict, dict_edges: dict, keep_asns: set[str]):
    filtered_nodes = {
        asn: weight
        for asn, weight in dict_nodes.items()
        if asn in keep_asns
    }

    filtered_edges = {
        (src_asn, dst_asn): weight
        for (src_asn, dst_asn), weight in dict_edges.items()
        if src_asn in keep_asns and dst_asn in keep_asns
    }

    return filtered_nodes, filtered_edges