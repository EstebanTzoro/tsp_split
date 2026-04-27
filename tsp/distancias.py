# tsp/distancias.py
import math
import heapq


def distancia_euclidea(nodo1: tuple, nodo2: tuple) -> float:
    """Distancia euclidea entre dos nodos (id, x, y)."""
    return math.sqrt((nodo1[1] - nodo2[1]) ** 2 + (nodo1[2] - nodo2[2]) ** 2)


def distancia_euclidea_total(camino: list) -> float:
    """
    Distancia total de un camino cerrado.
    El camino es una lista de nodos (id, x, y).
    """
    total = 0.0
    n = len(camino)
    for i in range(n - 1):
        total += distancia_euclidea(camino[i], camino[i + 1])
    total += distancia_euclidea(camino[-1], camino[0])
    return total


def calcular_mst(nodos: list) -> float:
    """
    Calcula el costo del Árbol de Expansión Mínima (MST) usando Prim.
    Se usa como cota inferior para calcular el gap de calidad.
    """
    n = len(nodos)
    visitado = [False] * n
    min_heap = [(0.0, 0)]
    total = 0.0

    while min_heap:
        peso, u = heapq.heappop(min_heap)

        if visitado[u]:
            continue

        visitado[u] = True
        total += peso

        for v in range(n):
            if not visitado[v]:
                dist = distancia_euclidea(nodos[u], nodos[v])
                heapq.heappush(min_heap, (dist, v))

    return total
