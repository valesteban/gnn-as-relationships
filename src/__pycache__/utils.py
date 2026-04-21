from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from collections import Counter
import torch





def month_number_to_name(mes: str | int) -> str:
    """Convierte número de mes a nombre.
    """
    if mes is None:
        raise ValueError("El mes no puede ser None")

    if isinstance(mes, int):
        mes_int = mes
    elif isinstance(mes, str):
        mes_str = mes.strip()
        if not mes_str:
            raise ValueError("El mes no puede ser un string vacío")
        try:
            mes_int = int(mes_str)
        except ValueError as e:
            raise ValueError(f"Mes inválido: {mes!r}") from e
    else:
        raise TypeError(f"Tipo inválido para mes: {type(mes).__name__}")

    # Validar rango antes de buscar en el diccionario
    if not 1 <= mes_int <= 12:
        raise ValueError(f"El mes debe estar entre 1 y 12, se recibió: {mes_int}")

    mapping = {
        1: "enero",
        2: "febrero",
        3: "marzo",
        4: "abril",
        5: "mayo",
        6: "junio",
        7: "julio",
        8: "agosto",
        9: "septiembre",
        10: "octubre",
        11: "noviembre",
        12: "diciembre",
    }

    return mapping[mes_int]