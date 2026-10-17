"""Corre los dos notebooks de entrenamiento sobre todos los grafos de un experimento.

Para cada grafo de <experimento>/grafos/<año>/<mes>/<AAAAMMDDTHHMM>/ que tenga
as_features.csv y gnn_edges.csv (generados por as-level-internet-graph):

1. Preparación:
   - descarga el snapshot CAIDA del mes si falta (CAIDA_DIR/AAAAMM01.as-rel.txt.bz2);
   - etiqueta las aristas (gnn_edges_caida.csv, GNN.label_edges_from_caida);
   - crea el dataset ToR del clasificador ANN (GNN.build_tor_dataset).
2. AS_relationship_inference_end_to_end.ipynb: un caso por ejecución
   (caso0 … caso4, comparacion).
3. AS_relationship_inference_por_partes.ipynb: embeddings (caso0 … caso4,
   deepwalk, bgp2vec) y luego el clasificador ANN una vez por cada embedding
   generado en ese grafo.

Cada caso se ejecuta en un kernel nuevo con solo sus celdas (etiquetas "config"
y "<caso>" en los metadatos de las celdas) y los parámetros por variables de
entorno (GRAFO_DIR, CAIDA_DIR, SEED, RESULTS_ROOT, GNN_CONFIG, TOR_PATH,
EMBEDDINGS_PATH). Un caso que falla (p. ej. sin memoria) se registra y se sigue
con el siguiente.

Salidas:
- por grafo: GRAFO_DIR/results/{end_to_end, EnfoquePorPartes}/… (save_run y
  los guardados de los notebooks) y GRAFO_DIR/results/ejecucion/ con el
  notebook ejecutado de cada caso (.ipynb) y un marcador .ok si terminó bien;
- por experimento: <experimento>/resultados_gnn/ejecucion_<fecha>.csv (estado y
  duración de cada caso) y metricas.csv (todas las métricas de todos los grafos).

Uso (desde la raíz del repositorio):
    env310/bin/python scripts/correr_experimento.py /media/vale/KINGSTON/as-level-internet-graph/experimentos/experimento7
    env310/bin/python scripts/correr_experimento.py <experimento> --grafos 2025 2026 --notebooks end_to_end --casos-e2e caso0 caso1
    env310/bin/python scripts/correr_experimento.py <experimento> --rapido      # prueba de humo (resultados aparte)
    env310/bin/python scripts/correr_experimento.py <experimento> --reanudar    # salta los casos ya terminados
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import nbformat  # noqa: E402
from nbclient import NotebookClient  # noqa: E402
from nbclient.exceptions import CellExecutionError, DeadKernelError  # noqa: E402

from modules.gnn import GNN  # noqa: E402

NOTEBOOKS = {
    "end_to_end": REPO / "AS_relationship_inference_end_to_end.ipynb",
    "por_partes": REPO / "AS_relationship_inference_por_partes.ipynb",
}

CASOS_E2E = ["caso0", "caso1", "caso2", "caso3", "caso4", "comparacion"]
CASOS_PARTES = ["caso0", "caso1", "caso2", "caso3", "caso4", "deepwalk", "bgp2vec", "clasificador"]

CAIDA_URL = "https://publicdata.caida.org/datasets/as-relationships/serial-1/{nombre}"
import yaml  # noqa: E402

# --caida-dir por defecto: rutas.caida_dir de config.yaml (raíz del repo)
with open(REPO / "config.yaml", encoding="utf-8") as _f:
    CAIDA_DIR_DEFECTO = yaml.safe_load(_f)["rutas"]["caida_dir"]

# Configuración mínima para comprobar que todo corre (no para obtener resultados)
# CONFIG_RAPIDA = {
#     "hidden_dim": 16, "out_dim": 8, "epochs": 2, "patience": 2, "num_heads": 1,
#     "epochs_muestreo": 1, "deepwalk_epochs": 1, "bgp2vec_max_routes": 20_000,
#     "bgp2vec_count_routes": False, "ann_epochs": 2,
# }


def buscar_grafos(exp_dir, filtros):
    """Carpetas de grafo del experimento, filtradas por subcadena del nombre (p. ej. '2026')."""
    grafos = sorted(p for p in (exp_dir / "grafos").glob("*/*/*") if p.is_dir())
    if filtros:
        grafos = [g for g in grafos if any(f in g.name for f in filtros)]
    return grafos


def asegurar_caida(grafo_dir, caida_dir, descargar):
    """Devuelve el archivo CAIDA del mes del grafo; lo descarga si falta y descargar=True."""
    try:
        return GNN.caida_file_for_graph(grafo_dir, caida_dir)
    except FileNotFoundError:
        if not descargar:
            raise
    nombre = f"{grafo_dir.name[:6]}01.as-rel.txt.bz2"
    destino = Path(caida_dir) / nombre
    tmp = destino.with_suffix(".parcial")
    print(f"  [CAIDA] descargando {nombre} …")
    urllib.request.urlretrieve(CAIDA_URL.format(nombre=nombre), tmp)
    tmp.rename(destino)
    return destino


def preparar_grafo(grafo_dir, results_root, caida_dir, seed, descargar):
    """CAIDA + etiquetado + dataset ToR. Devuelve None si todo está listo o el motivo para omitir el grafo."""
    for f in ("as_features.csv", "gnn_edges.csv"):
        if not (grafo_dir / f).exists():
            return f"sin {f} (grafo no enriquecido)"

    edges_caida = grafo_dir / "gnn_edges_caida.csv"
    if not edges_caida.exists():
        try:
            caida_file = asegurar_caida(grafo_dir, caida_dir, descargar)
        except Exception as e:  # sin red, 404, o descarga desactivada
            return f"sin archivo CAIDA: {e}"
        GNN(debug=True).label_edges_from_caida(grafo_dir / "gnn_edges.csv", caida_file,
                                               output_csv=edges_caida)

    tor_dir = results_root / "results" / "EnfoquePorPartes" / "ToR"
    tor_resumen = tor_dir / "tor_resumen.json"
    hecho = tor_resumen.exists() and json.loads(tor_resumen.read_text()).get("seed") == seed
    if not hecho:
        resumen = GNN(debug=True).build_tor_dataset(edges_caida, tor_dir, test_size=0.2, seed=seed)
        tor_resumen.write_text(json.dumps(resumen, indent=2))
    return None


def ejecutar_notebook(nb_path, tags, env, salida):
    """Ejecuta solo las celdas con alguna de las etiquetas dadas, en un kernel nuevo.

    Guarda el notebook ejecutado (aunque falle) en salida. Devuelve (ok, mensaje_error).
    """
    nb = nbformat.read(nb_path, as_version=4)
    nb.cells = [c for c in nb.cells
                if c.cell_type == "code" and set(c.get("metadata", {}).get("tags", [])) & set(tags)]

    previo = {k: os.environ.get(k) for k in env}
    os.environ.update({k: str(v) for k, v in env.items()})
    # El kernel "python3" se lanza con el "python" del PATH: forzar el del entorno actual
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    try:
        cliente = NotebookClient(nb, timeout=None, kernel_name="python3",
                                 resources={"metadata": {"path": str(REPO)}})
        cliente.execute()
        return True, ""
    except (CellExecutionError, DeadKernelError) as e:
        mensaje = re.sub(r"\x1b\[[0-9;]*m", "", str(e)).strip().splitlines()
        return False, (mensaje[-1] if mensaje else type(e).__name__)[:300]
    finally:
        for k, v in previo.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        salida.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, salida)


def embeddings_del_grafo(results_root):
    """Embeddings generados para el grafo (se excluye Caso2, que es del grafo externo 2022)."""
    base = results_root / "results" / "EnfoquePorPartes"
    rutas = []
    for caso in ("Caso0", "Caso1", "Caso3", "Caso4"):
        rutas += sorted((base / caso).glob("embeddings_*.pt"))
    rutas += sorted((base / "Caso5").glob("embeddings_deepWalk_*.pt"))
    rutas += sorted((base / "Caso5").glob("bgp2vec.word2vec"))
    return rutas


def consolidar_metricas(grafos_roots, salida):
    """Junta los metrics.json de todos los grafos en un único CSV (última corrida de cada carpeta)."""
    filas = []
    for grafo_dir, results_root in grafos_roots:
        for notebook, patron in (("end_to_end", "results/end_to_end/*/*/metrics.json"),
                                 ("por_partes", "results/EnfoquePorPartes/clasificador_ann/*/metrics.json")):
            for f in sorted(results_root.glob(patron)):
                m = json.loads(f.read_text(encoding="utf-8"))
                filas.append({
                    "grafo": grafo_dir.name, "notebook": notebook, "caso": m["caso"],
                    "modelo": m["modelo"], "decoder": m["decoder"], "seed": m["seed"],
                    "config_id": m.get("config_id", ""), "clase": m.get("extra", {}).get("clase", ""),
                    "accuracy": round(m["accuracy"], 6), "macro_f1": round(m["macro_f1"], 6),
                    "weighted_f1": round(m["weighted_f1"], 6), "n_test": m["n_test"],
                    "tiempo_s": m.get("extra", {}).get("tiempo_s", ""),
                    "tiempo_total_s": m.get("extra", {}).get("tiempo_total_s",
                                                             m.get("extra", {}).get("tiempo_s", "")),
                    "epocas": m.get("extra", {}).get("epocas", ""),
                    "fecha": m["fecha"], "carpeta": str(f.parent),
                })
    if filas:
        with open(salida, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
            w.writeheader()
            w.writerows(filas)
    return len(filas)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("experimento", type=Path, help="carpeta experimentos/experimento<N>")
    ap.add_argument("--grafos", nargs="*", default=[],
                    help="solo grafos cuyo nombre contenga alguna de estas cadenas (p. ej. 2025 2026)")
    ap.add_argument("--notebooks", nargs="*", default=list(NOTEBOOKS), choices=list(NOTEBOOKS))
    ap.add_argument("--casos-e2e", nargs="*", default=CASOS_E2E, choices=CASOS_E2E)
    ap.add_argument("--casos-partes", nargs="*", default=CASOS_PARTES, choices=CASOS_PARTES)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--caida-dir", type=Path, default=Path(CAIDA_DIR_DEFECTO))
    ap.add_argument("--sin-descarga", action="store_true",
                    help="no descargar CAIDA: omitir los grafos sin archivo del mes")
    ap.add_argument("--config", type=json.loads, default={},
                    help='JSON que sobrescribe GLOBAL_CONFIG, p. ej. \'{"epochs": 100}\'')
    ap.add_argument("--rapido", action="store_true",
                    help="prueba de humo: hiperparámetros mínimos y resultados en GRAFO_DIR/prueba_rapida/")
    ap.add_argument("--reanudar", action="store_true", help="saltar los casos con marcador .ok")
    ap.add_argument("--solo-preparar", action="store_true",
                    help="solo CAIDA + etiquetado + dataset ToR, sin ejecutar notebooks")
    args = ap.parse_args(argv)

    exp_dir = args.experimento.resolve()
    if not (exp_dir / "grafos").is_dir():
        print(f"No es una carpeta de experimento (falta grafos/): {exp_dir}")
        return 2
    grafos = buscar_grafos(exp_dir, args.grafos)
    if not grafos:
        print("No hay grafos que coincidan.")
        return 2

    config = {**(CONFIG_RAPIDA if args.rapido else {}), **args.config}
    resumen_dir = exp_dir / ("resultados_gnn_prueba" if args.rapido else "resultados_gnn")
    resumen_dir.mkdir(parents=True, exist_ok=True)
    inicio = datetime.now()
    ejecucion_csv = resumen_dir / f"ejecucion_{inicio:%Y%m%d_%H%M%S}.csv"
    campos = ["grafo", "notebook", "caso", "embedding", "estado", "duracion_s", "detalle"]
    with open(ejecucion_csv, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(campos)

    def registrar(**fila):
        with open(ejecucion_csv, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=campos).writerow({k: fila.get(k, "") for k in campos})
        print(f"  → {fila['notebook']}/{fila['caso']} {fila.get('embedding', '')}: {fila['estado']}"
              f"{' (' + str(fila['duracion_s']) + ' s)' if fila.get('duracion_s') not in (None, '') else ''}"
              f"{'  ' + fila['detalle'] if fila.get('detalle') else ''}")

    print(f"Experimento: {exp_dir}  ({len(grafos)} grafos)")
    print(f"Semilla: {args.seed}  |  config: {config or 'la de los notebooks'}")
    print(f"Registro: {ejecucion_csv}")

    grafos_ok = []
    try:
        for n, grafo_dir in enumerate(grafos, 1):
            print(f"\n[{n}/{len(grafos)}] {grafo_dir.relative_to(exp_dir)}")
            results_root = grafo_dir / "prueba_rapida" if args.rapido else grafo_dir
            motivo = preparar_grafo(grafo_dir, results_root, args.caida_dir, args.seed,
                                    descargar=not args.sin_descarga)
            if motivo:
                registrar(grafo=grafo_dir.name, notebook="-", caso="-", estado="omitido", detalle=motivo)
                continue
            grafos_ok.append((grafo_dir, results_root))
            if args.solo_preparar:
                registrar(grafo=grafo_dir.name, notebook="-", caso="preparacion", estado="ok")
                continue

            env_base = {"GRAFO_DIR": grafo_dir, "CAIDA_DIR": args.caida_dir, "SEED": args.seed,
                        "RESULTS_ROOT": results_root, "GNN_CONFIG": json.dumps(config),
                        "TOR_PATH": f"{results_root}/results/EnfoquePorPartes/ToR/"}
            ejec_dir = results_root / "results" / "ejecucion"

            trabajos = []  # (notebook, caso, embedding o None)
            if "end_to_end" in args.notebooks:
                trabajos += [("end_to_end", c, None) for c in args.casos_e2e]
            if "por_partes" in args.notebooks:
                trabajos += [("por_partes", c, None) for c in args.casos_partes if c != "clasificador"]
                if "clasificador" in args.casos_partes:
                    trabajos.append(("por_partes", "clasificador", "*"))  # se expande al llegar

            while trabajos:
                notebook, caso, emb = trabajos.pop(0)
                if emb == "*":  # un clasificador por cada embedding que exista ahora
                    embs = embeddings_del_grafo(results_root)
                    if not embs:
                        registrar(grafo=grafo_dir.name, notebook=notebook, caso=caso,
                                  estado="omitido", detalle="no hay embeddings")
                    trabajos = [(notebook, caso, e) for e in embs] + trabajos
                    continue

                nombre = f"{notebook}_{caso}" + (f"_{emb.parent.name}_{emb.stem}" if emb else "")
                marcador = ejec_dir / f"{nombre}.ok"
                if args.reanudar and marcador.exists():
                    registrar(grafo=grafo_dir.name, notebook=notebook, caso=caso,
                              embedding=emb.name if emb else "", estado="ya hecho")
                    continue

                env = dict(env_base)
                if emb:
                    env["EMBEDDINGS_PATH"] = emb
                t0 = time.perf_counter()
                ok, error = ejecutar_notebook(NOTEBOOKS[notebook], ["config", caso], env,
                                              ejec_dir / f"{nombre}.ipynb")
                duracion = round(time.perf_counter() - t0, 1)
                if ok:
                    marcador.write_text(json.dumps({"seed": args.seed, "config": config,
                                                    "fecha": datetime.now().isoformat(timespec="seconds")}))
                elif marcador.exists():
                    marcador.unlink()
                registrar(grafo=grafo_dir.name, notebook=notebook, caso=caso,
                          embedding=emb.name if emb else "", estado="ok" if ok else "error",
                          duracion_s=duracion, detalle=error)
    except KeyboardInterrupt:
        print("\nInterrumpido: se consolidan las métricas obtenidas hasta ahora.")

    n_metricas = consolidar_metricas(grafos_ok, resumen_dir / "metricas.csv")
    print(f"\nMétricas consolidadas: {n_metricas} filas en {resumen_dir / 'metricas.csv'}")
    print(f"Registro de ejecución: {ejecucion_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
