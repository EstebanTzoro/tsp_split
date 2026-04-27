# tsp/heuristicas.py
import random
import numpy as np
from tsp.distancias import distancia_euclidea_total


def _construir_matriz_distancias(nodos: list) -> np.ndarray:
    n = len(nodos)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(
                (nodos[i][1] - nodos[j][1]) ** 2 +
                (nodos[i][2] - nodos[j][2]) ** 2
            )
            dist[i][j] = d
            dist[j][i] = d
    return dist


def heuristica_nnh(nodos: list) -> tuple[list, float]:
    """
    Nearest Neighbor Heuristic.
    Construye la ruta empezando en el nodo 0 y siempre yendo al más cercano no visitado.
    Devuelve (ruta, costo).
    """
    n = len(nodos)
    dist = _construir_matriz_distancias(nodos)

    ruta_idx = [0]
    costo = 0.0
    no_visitados = set(range(n))
    no_visitados.remove(0)

    while no_visitados:
        actual = ruta_idx[-1]
        siguiente = min(no_visitados, key=lambda i: dist[actual][i])
        costo += dist[actual][siguiente]
        ruta_idx.append(siguiente)
        no_visitados.remove(siguiente)

    costo += dist[ruta_idx[-1]][ruta_idx[0]]
    ruta = [nodos[i] for i in ruta_idx]
    return ruta, costo


def heuristica_insercion(nodos: list) -> tuple[list, float]:
    """
    Best Insertion Heuristic.
    Inserta iterativamente el nodo que minimiza el incremento de costo.
    Devuelve (ruta, costo).
    """
    n = len(nodos)
    dist = _construir_matriz_distancias(nodos)

    ruta_idx = [0, 0]
    costo = 0.0

    while len(ruta_idx) < n + 1:
        mejor_incremento = float("inf")
        sel = None
        pos = None

        for i in range(n):
            if i not in ruta_idx:
                for p in range(1, len(ruta_idx)):
                    incremento = (
                        dist[ruta_idx[p - 1]][i]
                        + dist[i][ruta_idx[p]]
                        - dist[ruta_idx[p - 1]][ruta_idx[p]]
                    )
                    if incremento < mejor_incremento:
                        mejor_incremento = incremento
                        sel = i
                        pos = p

        costo += mejor_incremento
        ruta_idx.insert(pos, sel)

    ruta_idx = ruta_idx[:-1]
    ruta = [nodos[i] for i in ruta_idx]
    return ruta, costo


def heuristica_randomkeys(nodos: list, seed: int = None) -> tuple[list, float]:
    """
    Random Keys Heuristic.
    Asigna una clave aleatoria a cada nodo y ordena por ella.
    Devuelve (ruta, costo).
    """
    if seed is not None:
        random.seed(seed)

    n = len(nodos)
    keys = [(i, random.random()) for i in range(n)]
    keys.sort(key=lambda x: x[1])
    orden = [k[0] for k in keys]

    ruta = [nodos[i] for i in orden]
    costo = distancia_euclidea_total(ruta)
    return ruta, costo


# Registro central de heurísticas disponibles
HEURISTICAS = {
    "NNH": heuristica_nnh,
    "INSERCION": heuristica_insercion,
    "RANDOM_KEYS": heuristica_randomkeys,
}


def obtener_heuristica(nombre: str):
    """Devuelve la función de heurística por nombre. Lanza error si no existe."""
    if nombre not in HEURISTICAS:
        raise ValueError(
            f"Heurística '{nombre}' no válida. Disponibles: {list(HEURISTICAS.keys())}"
        )
    return HEURISTICAS[nombre]
