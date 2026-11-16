"""Corre varias pruebas (hiperparámetros, semillas, split) sobre los notebooks de entrenamiento.

Lee un archivo de pruebas (p. ej. pruebas.yaml) que define variaciones sobre config.yaml con
claves con puntos. Cada prueba tiene `cambios` (se aplican a todas sus variantes) y `barrido`
(producto cartesiano: una variante por combinación). Por cada variante y notebook:

1. escribe <salida>/<prueba>/<variante>/config.yaml con los cambios aplicados y con
   experimento.nombre = <nombre base>/pruebas/<prueba>/<variante> (resultados separados por
   variante, para que save_run no sobrescriba entre variantes que difieren fuera de
   global_config_*, como comunes.split o comunes.validacion);
2. ejecuta el notebook en un kernel nuevo, cambiando solo en memoria la línea CONFIG_FILE = ...
   (el .ipynb del repositorio no se modifica);
3. guarda el notebook ejecutado (también si falla) y una fila en <salida>/estado.csv.

Al final junta los resumen.csv de las variantes terminadas y muestra media ± std de accuracy y
macro-F1 por (prueba, variante sin semilla, enfoque, caso, modelo, decoder); los folds de K-fold
se agrupan igual que las semillas. La tabla se guarda en <salida>/resumen_pruebas.csv.

El notebook por partes solo implementa holdout: con comunes.validacion.tipo = kfold no se
ejecuta y queda como "omitido".

Uso (desde la raíz del repositorio):
    env312/bin/python scripts/correr_pruebas.py pruebas.yaml
    env312/bin/python scripts/correr_pruebas.py pruebas.yaml --dry-run
    env312/bin/python scripts/correr_pruebas.py pruebas.yaml --solo semillas splits
    env312/bin/python scripts/correr_pruebas.py pruebas.yaml --reanudar
"""
import argparse
import copy
import csv
import itertools
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent

NOTEBOOKS = {
    "end_to_end": REPO / "AS_relationship_inference_end_to_end.ipynb",
    "por_partes": REPO / "AS_relationship_inference_por_partes.ipynb",
}
# Subcarpeta de resultados (bajo <salida>/<nombre>/) que escribe cada notebook con save_run
ENFOQUE = {"end_to_end": "end_to_end", "por_partes": "por_etapas"}

CAMPOS_ESTADO = ["fecha", "prueba", "variante", "notebook", "estado", "duracion_s",
                 "error", "config", "resultados", "notebook_ejecutado"]


# --------------------------------------------------------------------------- config
def obtener(cfg, clave):
    """Valor de una clave con puntos; KeyError si no existe en la config."""
    nodo = cfg
    for parte in clave.split("."):
        if not isinstance(nodo, dict) or parte not in nodo:
            raise KeyError(clave)
        nodo = nodo[parte]
    return nodo


def asignar(cfg, clave, valor):
    """Asigna una clave con puntos que ya existe en la config (atrapa errores de tipeo)."""
    obtener(cfg, clave)
    *ruta, ultima = clave.split(".")
    nodo = cfg
    for parte in ruta:
        nodo = nodo[parte]
    nodo[ultima] = copy.deepcopy(valor)


def validar(cfg, notebooks, donde):
    """Mismos criterios que split_edges_classification / split_edges_kfold y la celda 12."""
    split = cfg["comunes"]["split"]
    val = cfg["comunes"].get("validacion") or {"tipo": "holdout"}
    tipo = val.get("tipo")
    tr, va = float(split.get("train_size", 0.7)), float(split.get("val_size", 0.15))
    if split.get("balance_mode", "proportional") not in {"proportional", "strict_equal"}:
        raise ValueError(f"{donde}: balance_mode inválido: {split.get('balance_mode')!r}")
    if not 0 <= va < 1:
        raise ValueError(f"{donde}: split.val_size debe estar en [0, 1). Recibido: {va}")
    if tipo not in {"holdout", "kfold"}:
        raise ValueError(f"{donde}: validacion.tipo inválido: {tipo!r}. Usa 'holdout' o 'kfold'")
    if tipo == "kfold" and int(val.get("k", 5)) < 2:
        raise ValueError(f"{donde}: validacion.k debe ser >= 2. Recibido: {val.get('k')}")
    # holdout usa train_size; por partes siempre usa holdout
    if (tipo == "holdout" or "por_partes" in notebooks) and not tr + va < 1.0:
        raise ValueError(f"{donde}: train_size + val_size debe ser < 1. Recibido: {tr} + {va}")


def tipo_validacion(cfg):
    return (cfg["comunes"].get("validacion") or {"tipo": "holdout"}).get("tipo")


# --------------------------------------------------------------------------- variantes
def texto_valor(valor):
    if isinstance(valor, dict):
        return "-".join(texto_valor(v) for v in valor.values())
    if isinstance(valor, (list, tuple)):
        return "-".join(texto_valor(v) for v in valor)
    return re.sub(r"[^A-Za-z0-9.+]", "", str(valor))


def nombres_cortos(claves):
    """Último componente de cada clave; la clave completa si dos coinciden."""
    ultimos = [c.split(".")[-1] for c in claves]
    return {c: (u if ultimos.count(u) == 1 else c) for c, u in zip(claves, ultimos)}


def efectivo(cfg, valores):
    """Valores del barrido que de verdad cambian la ejecución (kfold ignora train_size)."""
    ef = copy.deepcopy(valores)
    if tipo_validacion(cfg) == "kfold":
        for clave, v in ef.items():
            if clave == "comunes.split" and isinstance(v, dict):
                v.pop("train_size", None)
            elif clave == "comunes.split.train_size":
                ef[clave] = None
    return json.dumps(ef, sort_keys=True)


def expandir(prueba, base):
    """Lista de variantes (dicts) de una prueba y avisos de combinaciones colapsadas."""
    nombre = prueba["nombre"]
    if not re.fullmatch(r"[A-Za-z0-9_.+-]+", nombre):
        raise ValueError(f"prueba {nombre!r}: usa solo letras, números, _ . + -")
    notebooks = prueba.get("notebooks", ["end_to_end"])
    for nb in notebooks:
        if nb not in NOTEBOOKS:
            raise ValueError(f"prueba {nombre}: notebook desconocido {nb!r} (usa {list(NOTEBOOKS)})")
    cambios = prueba.get("cambios") or {}
    barrido = prueba.get("barrido") or {}
    for clave, lista in barrido.items():
        if not isinstance(lista, list) or not lista:
            raise ValueError(f"prueba {nombre}: barrido.{clave} debe ser una lista no vacía")
    cortos = nombres_cortos(list(barrido))
    base_nombre = str(base["experimento"]["nombre"])

    variantes, vistos, avisos = [], {}, []
    for combo in itertools.product(*barrido.values()):
        valores = dict(zip(barrido, combo))
        cfg = copy.deepcopy(base)
        for clave, valor in {**cambios, **valores}.items():
            try:
                asignar(cfg, clave, valor)
            except KeyError:
                raise KeyError(f"prueba {nombre}: la clave {clave!r} no existe en la config base") from None
        partes = [f"{cortos[c]}-{texto_valor(v)}" for c, v in valores.items()]
        var = f"{nombre}__{'_'.join(partes)}" if partes else nombre
        validar(cfg, notebooks, var)

        clave_ef = efectivo(cfg, valores)
        if clave_ef in vistos:
            avisos.append(f"{var}: igual a {vistos[clave_ef]} (con kfold no se usa train_size); se omite")
            continue
        vistos[clave_ef] = var

        if "experimento.nombre" not in cambios and "experimento.nombre" not in valores:
            cfg["experimento"]["nombre"] = f"{base_nombre}/pruebas/{nombre}/{var}"
        sin_semilla = {c: v for c, v in valores.items() if c != "comunes.seed"}
        variantes.append({
            "prueba": nombre, "variante": var, "notebooks": notebooks, "cfg": cfg,
            "valores": valores,
            "grupo": "_".join(f"{cortos[c]}-{texto_valor(v)}" for c, v in sin_semilla.items()) or "-",
        })
    return variantes, avisos


# --------------------------------------------------------------------------- ejecución
def ejecutar_notebook(nb_path, config_path, salida_nb):
    """Ejecuta el notebook con CONFIG_FILE apuntando a config_path. Devuelve (estado, error)."""
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError, DeadKernelError

    nb = nbformat.read(nb_path, as_version=4)
    patron = re.compile(r"^CONFIG_FILE\s*=.*$", re.M)
    celdas = [c for c in nb.cells if c.cell_type == "code" and patron.search(c.source)]
    if len(celdas) != 1:
        raise RuntimeError(f"{nb_path.name}: se esperaba 1 celda con 'CONFIG_FILE = ...', hay {len(celdas)}")
    celdas[0].source = patron.sub(f"CONFIG_FILE = {str(config_path)!r}", celdas[0].source, count=1)

    cliente = NotebookClient(nb, timeout=None, kernel_name="python3",
                             resources={"metadata": {"path": str(REPO)}})
    estado, error = "ok", ""
    try:
        cliente.execute()
    except CellExecutionError as e:
        texto = re.sub(r"\x1b\[[0-9;]*m", "", str(e)).strip()   # sin códigos de color ANSI
        estado, error = "error", (texto.splitlines()[-1] if texto else repr(e))[:500]
    except DeadKernelError as e:
        estado, error = "error", f"kernel muerto (¿memoria?): {e}"[:500]
    finally:
        nbformat.write(nb, salida_nb)
    return estado, error


def leer_estado(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def registrar(path, fila):
    nuevo = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_ESTADO)
        if nuevo:
            w.writeheader()
        w.writerow(fila)


def resultados_dir(cfg, nb):
    return Path(cfg["experimento"]["salida"]) / str(cfg["experimento"]["nombre"]) / ENFOQUE[nb]


# --------------------------------------------------------------------------- resumen
def resumir(variantes, estado_path, salida):
    import pandas as pd

    ok = {(f["variante"], f["notebook"]) for f in leer_estado(estado_path) if f["estado"] == "ok"}
    tablas = []
    for v in variantes:
        for nb in v["notebooks"]:
            res = resultados_dir(v["cfg"], nb) / "resumen.csv"
            if (v["variante"], nb) not in ok or not res.exists():
                continue
            df = pd.read_csv(res)
            df["prueba"], df["variante"], df["grupo"] = v["prueba"], v["variante"], v["grupo"]
            df["enfoque"] = ENFOQUE[nb]
            df["validacion"] = texto_valor(v["cfg"]["comunes"].get("validacion") or {"tipo": "holdout"})
            tablas.append(df)
    if not tablas:
        print("\n[Resumen] No hay variantes terminadas con resumen.csv.")
        return
    df = pd.concat(tablas, ignore_index=True)
    df["caso_base"] = df["caso"].astype(str).str.replace(r"_kfold\d+/fold\d+$", "", regex=True)
    tabla = (df.groupby(["prueba", "grupo", "enfoque", "validacion", "caso_base", "modelo", "decoder"])
               .agg(n=("macro_f1", "size"),
                    semillas=("seed", lambda s: " ".join(map(str, sorted(set(s))))),
                    accuracy_media=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
                    macro_f1_media=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"))
               .reset_index().round(4))
    tabla.to_csv(salida / "resumen_pruebas.csv", index=False)
    with pd.option_context("display.max_rows", None, "display.width", 200, "display.max_columns", None):
        print("\n[Resumen media ± std por semilla y fold]")
        print(tabla.to_string(index=False))
    print(f"\nGuardado en {salida / 'resumen_pruebas.csv'}")


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pruebas", type=Path, help="archivo YAML de pruebas (p. ej. pruebas.yaml)")
    ap.add_argument("--solo", nargs="+", metavar="PRUEBA", help="correr solo estas pruebas")
    ap.add_argument("--reanudar", action="store_true", help="saltar variantes que ya terminaron ok")
    ap.add_argument("--dry-run", action="store_true", help="solo listar variantes, sin ejecutar")
    args = ap.parse_args(argv)

    with open(args.pruebas, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    base_path = Path(spec.get("base", "config.yaml"))
    base_path = base_path if base_path.is_absolute() else (args.pruebas.resolve().parent / base_path)
    with open(base_path, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    salida = Path(spec["salida"])

    pruebas = spec.get("pruebas") or []
    nombres = [p["nombre"] for p in pruebas]
    if len(set(nombres)) != len(nombres):
        raise ValueError(f"nombres de prueba repetidos: {nombres}")
    if args.solo:
        faltan = set(args.solo) - set(nombres)
        if faltan:
            raise ValueError(f"--solo: pruebas inexistentes {sorted(faltan)} (hay {nombres})")
        pruebas = [p for p in pruebas if p["nombre"] in args.solo]

    # Expandir y validar todo antes de ejecutar nada
    variantes = []
    for p in pruebas:
        vs, avisos = expandir(p, base)
        variantes += vs
        print(f"\n[Prueba {p['nombre']}] {len(vs)} variante(s), notebooks: {p.get('notebooks', ['end_to_end'])}")
        if p.get("cambios"):
            print(f"  cambios: {p['cambios']}")
        for v in vs:
            omit = " (por_partes omitido: kfold)" if ("por_partes" in v["notebooks"]
                                                       and tipo_validacion(v["cfg"]) == "kfold") else ""
            print(f"  - {v['variante']}{omit}")
            for c, val in v["valores"].items():
                print(f"      {c} = {val}")
        for a in avisos:
            print(f"  [aviso] {a}")
    n_ej = sum(len(v["notebooks"]) for v in variantes)
    print(f"\nTotal: {len(variantes)} variantes, {n_ej} ejecuciones de notebook. Salida: {salida}")
    if args.dry_run:
        return

    # El grafo de cada variante debe existir (p. ej. disco externo montado) antes de entrenar
    faltan = sorted({str(Path(v["cfg"]["rutas"]["grafo_dir"]) / a) for v in variantes
                     for a in ("as_features.csv", "gnn_edges_caida.csv")
                     if not (Path(v["cfg"]["rutas"]["grafo_dir"]) / a).exists()})
    if faltan:
        raise FileNotFoundError("No se encuentran los archivos del grafo (¿disco montado?):\n  "
                                + "\n  ".join(faltan))

    salida.mkdir(parents=True, exist_ok=True)
    estado_path = salida / "estado.csv"
    hechos = {(f["variante"], f["notebook"]) for f in leer_estado(estado_path) if f["estado"] == "ok"}

    i = 0
    for v in variantes:
        dir_var = salida / v["prueba"] / v["variante"]
        dir_var.mkdir(parents=True, exist_ok=True)
        config_path = dir_var / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(v["cfg"], f, allow_unicode=True, sort_keys=False)

        for nb in v["notebooks"]:
            i += 1
            fila = {"fecha": datetime.now().isoformat(timespec="seconds"), "prueba": v["prueba"],
                    "variante": v["variante"], "notebook": nb, "config": str(config_path),
                    "resultados": str(resultados_dir(v["cfg"], nb)), "error": "",
                    "duracion_s": 0, "notebook_ejecutado": ""}
            if args.reanudar and (v["variante"], nb) in hechos:
                print(f"[{i}/{n_ej}] {v['variante']} · {nb}: ya terminado, se salta")
                continue
            if nb == "por_partes" and tipo_validacion(v["cfg"]) == "kfold":
                print(f"[{i}/{n_ej}] {v['variante']} · {nb}: omitido (por partes no implementa kfold)")
                registrar(estado_path, {**fila, "estado": "omitido",
                                        "error": "por partes no implementa kfold"})
                continue

            print(f"[{i}/{n_ej}] {v['variante']} · {nb}: ejecutando…", flush=True)
            salida_nb = dir_var / f"{nb}.ipynb"
            t0 = time.time()
            try:
                estado, error = ejecutar_notebook(NOTEBOOKS[nb], config_path, salida_nb)
            except Exception as e:  # noqa: BLE001 (falla al preparar o al arrancar el kernel)
                estado, error = "error", f"{type(e).__name__}: {e}"[:500]
            dur = round(time.time() - t0, 1)
            registrar(estado_path, {**fila, "estado": estado, "error": error, "duracion_s": dur,
                                    "notebook_ejecutado": str(salida_nb)})
            print(f"    -> {estado} en {dur:.0f} s" + (f": {error}" if error else ""), flush=True)

    resumir(variantes, estado_path, salida)


if __name__ == "__main__":
    sys.exit(main())
