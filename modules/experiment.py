"""Reproducibilidad y guardado de resultados de los entrenamientos.

- set_seed: fija todas las fuentes de aleatoriedad (random, numpy, torch, PyG)
  para que una misma celda dé siempre el mismo resultado, sin importar qué
  celdas se hayan ejecutado antes.
- save_run: guarda pesos, métricas e historial de un modelo entrenado en
  results_dir/<caso>/<modelo>_<decoder>_seed<seed>/ y agrega una fila a
  results_dir/resumen.csv.
"""
import csv
import hashlib
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch_geometric
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def set_seed(seed=42, deterministic=True):
    """Fija la semilla de random, numpy, torch (CPU y CUDA) y PyG.

    Con deterministic=True además pide a PyTorch algoritmos deterministas. En
    CPU los resultados son idénticos entre corridas; en GPU algunas operaciones
    scatter de PyG no tienen versión determinista y solo emiten un aviso
    (warn_only), así que pueden variar levemente.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch_geometric.seed_everything(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def _to_numpy(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def config_id(config=None, extra=None):
    """Identificador corto (6 caracteres) de la configuración de una corrida.

    Es el hash de los hiperparámetros (config) y de la clase del modelo
    (extra["clase"]); en por partes también de los del embedding
    (extra["config_embedding"], extra["clase_embedding"]). Mismos valores dan
    el mismo id, así que repetir una corrida sobrescribe su carpeta y variar
    hiperparámetros o capas crea una carpeta nueva.
    """
    extra = extra or {}
    partes = {"config": config or {}}
    for clave in ("clase", "config_embedding", "clase_embedding"):
        if extra.get(clave) is not None:
            partes[clave] = extra[clave]
    datos = json.dumps(partes, sort_keys=True, default=str)
    return hashlib.md5(datos.encode("utf-8")).hexdigest()[:6]


def save_run(results_dir, case, model_name, decoder, *, seed, y_true, y_pred,
             class_names, state_dict=None, history=None, config=None, extra=None):
    """Guarda los resultados de un modelo evaluado en test.

    Escribe en results_dir/<case>/<model_name>_<decoder>_seed<seed>_<config_id>/
    (config_id distingue hiperparámetros y capas distintos, ver config_id):
      - model.pth     pesos (state_dict), si se pasan
      - metrics.json  accuracy, macro-F1, F1 ponderado, reporte por clase,
                      matriz de confusión, semilla, config y extra
      - history.csv   curvas por época (dict de listas), si se pasa
    y agrega una fila a results_dir/resumen.csv. Devuelve la carpeta creada.
    """
    cid = config_id(config, extra)
    run_dir = Path(results_dir) / case / f"{model_name}_{decoder}_seed{seed}_{cid}"
    run_dir.mkdir(parents=True, exist_ok=True)

    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    labels = list(range(len(class_names)))

    metrics = {
        "caso": case,
        "modelo": model_name,
        "decoder": decoder,
        "seed": seed,
        "config_id": cid,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n_test": int(len(y_true)),
        "reporte": classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                         output_dict=True, zero_division=0),
        "matriz_confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "clases": list(class_names),
        "config": config or {},
        "extra": extra or {},
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    if state_dict is not None:
        torch.save({k: v.detach().cpu() for k, v in state_dict.items()}, run_dir / "model.pth")

    if history:
        n = max(len(v) for v in history.values())
        with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", *history.keys()])
            for i in range(n):
                writer.writerow([i + 1, *[v[i] if i < len(v) else "" for v in history.values()]])

    extra_m = metrics["extra"]
    tiempo_s = extra_m.get("tiempo_s")
    tiempo_total_s = extra_m.get("tiempo_total_s", tiempo_s)
    _append_resumen(Path(results_dir) / "resumen.csv", {
        "fecha": metrics["fecha"], "caso": case, "modelo": model_name, "decoder": decoder, "seed": seed,
        "config_id": cid,
        "accuracy": f"{metrics['accuracy']:.6f}", "macro_f1": f"{metrics['macro_f1']:.6f}",
        "tiempo_s": "" if tiempo_s is None else f"{tiempo_s:.1f}",
        "tiempo_total_s": "" if tiempo_total_s is None else f"{tiempo_total_s:.1f}",
        "epocas": extra_m.get("epocas", ""),
        "carpeta": str(run_dir),
    })

    print(f"[RESULTADOS] {run_dir}  (acc={metrics['accuracy']:.4f}, macro-F1={metrics['macro_f1']:.4f})")
    return run_dir


def _append_resumen(resumen, fila):
    """Agrega una fila a resumen.csv; si el archivo tiene columnas antiguas, lo reescribe con las nuevas."""
    campos = list(fila)
    previas = None
    if resumen.exists():
        with open(resumen, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != campos:
                previas = list(reader)
    if previas is not None or not resumen.exists():
        with open(resumen, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=campos, restval="", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(previas or [])
    with open(resumen, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=campos).writerow(fila)
