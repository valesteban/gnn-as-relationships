"""METIS para `ClusterData` (Caso 3, ClusterGCN) en Windows.

En Windows ni `torch_sparse` ("Not compiled with METIS support") ni `pyg_lib`
("METIS not yet supported on Windows") traen METIS, así que `ClusterData` falla.
`activar_metis_windows(dll)` reemplaza `ClusterData._metis` por una llamada directa a
`METIS_PartGraphKway` (o `METIS_PartGraphRecursive` si `recursive=True`) de una
`metis.dll` (METIS 5.1.0, índices de 32 bits), con las mismas opciones por defecto que
usa `torch_sparse` en Linux. En Linux, o si PyG ya tiene METIS, no hace nada.

`metis.dll` se genera desde la biblioteca estática de conda-forge (metis 5.1.0, win-64);
ver README, sección "Caso 3 en Windows".
"""
import ctypes
from pathlib import Path

import torch
from torch_geometric.loader import ClusterData
from torch_geometric.utils import remove_self_loops, sort_edge_index
from torch_geometric.utils.sparse import index2ptr

_IDX = ctypes.c_int32
_PIDX = ctypes.POINTER(_IDX)


def pyg_tiene_metis():
    """True si torch_sparse o pyg_lib pueden particionar con METIS en este sistema."""
    ptr, idx = torch.tensor([0, 1, 2]), torch.tensor([1, 0])
    try:
        torch.ops.torch_sparse.partition(ptr, idx, None, 2, False)
        return True
    except Exception:
        pass
    try:
        import pyg_lib
        pyg_lib.partition.metis(ptr, idx, 2)
        return True
    except Exception:
        return False


def _ptr(t):
    return ctypes.cast(t.data_ptr(), _PIDX)


def activar_metis_windows(dll_path):
    """Usa `dll_path` para el METIS de ClusterData si PyG no lo tiene. Devuelve True si lo activó."""
    if pyg_tiene_metis():
        return False
    dll_path = Path(dll_path)
    if not dll_path.exists():
        raise FileNotFoundError(f"No existe metis.dll: {dll_path} (ver README, 'Caso 3 en Windows')")
    lib = ctypes.CDLL(str(dll_path))
    for f in (lib.METIS_PartGraphKway, lib.METIS_PartGraphRecursive):
        f.restype = ctypes.c_int

    def _metis(self, edge_index, num_nodes):
        # METIS espera un grafo simétrico sin self-loops, en formato CSR.
        ei, _ = remove_self_loops(edge_index.cpu())
        row, col = sort_edge_index(ei, num_nodes=num_nodes)
        xadj = index2ptr(row, size=num_nodes).to(torch.int32).contiguous()
        adjncy = col.to(torch.int32).contiguous()
        part = torch.zeros(num_nodes, dtype=torch.int32)

        nvtxs, ncon, nparts, objval = _IDX(num_nodes), _IDX(1), _IDX(self.num_parts), _IDX(0)
        fn = lib.METIS_PartGraphRecursive if self.recursive else lib.METIS_PartGraphKway
        ret = fn(ctypes.byref(nvtxs), ctypes.byref(ncon), _ptr(xadj), _ptr(adjncy),
                 None, None, None, ctypes.byref(nparts), None, None, None,
                 ctypes.byref(objval), _ptr(part))
        if ret != 1:  # METIS_OK
            raise RuntimeError(f"METIS devolvió el código {ret}")
        return part.to(torch.long).to(edge_index.device)

    ClusterData._metis = _metis
    return True
