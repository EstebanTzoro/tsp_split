# tsp/dos_opt.py
"""
Movimientos 2-Opt para TSP.

Incluye:
- Best Improvement: evalúa todos los vecinos y escoge el mejor
- First Improvement: acepta el primer vecino que mejore
"""
import time
from tsp.distancias import distancia_euclidea_total


def _aplicar_swap_2opt(ruta: list, i: int, j: int) -> list:
    """Construye nueva ruta invirtiendo el segmento [i, j]."""
    return ruta[:i] + ruta[i:j + 1][::-1] + ruta[j + 1:]


# =============================================================
# 2-OPT BEST IMPROVEMENT
# =============================================================
def aplicar_2opt_una_iteracion(
    ruta_actual: list,
    tolerancia: float = 1e-9,
    timeout: float = None,
    tiempo_inicio: float = None,
) -> tuple:
    """
    Busca el MEJOR vecino 2-opt en una pasada completa (Best Improvement).

    Si timeout y tiempo_inicio son provistos, corta la búsqueda a mitad
    de iteración cuando se excede el tiempo (devuelve la mejor encontrada hasta ese momento).

    Devuelve
    --------
    nueva_ruta      : list
    costo_actual    : float
    nuevo_costo     : float
    mejora          : float
    hubo_mejora     : bool
    corto_por_tiempo : bool
    """
    costo_actual = distancia_euclidea_total(ruta_actual)
    mejor_costo = costo_actual
    mejor_ruta = ruta_actual
    hubo_mejora = False
    corto_por_tiempo = False
    n = len(ruta_actual)

    for i in range(1, n - 2):
        # Verificar timeout entre vecinos externos
        if timeout is not None and tiempo_inicio is not None:
            if (time.perf_counter() - tiempo_inicio) >= timeout:
                corto_por_tiempo = True
                break

        for j in range(i + 1, n - 1):
            nueva = _aplicar_swap_2opt(ruta_actual, i, j)
            nuevo_costo = distancia_euclidea_total(nueva)
            if nuevo_costo < mejor_costo - tolerancia:
                mejor_costo = nuevo_costo
                mejor_ruta = nueva
                hubo_mejora = True

    mejora = costo_actual - mejor_costo
    return mejor_ruta, costo_actual, mejor_costo, mejora, hubo_mejora, corto_por_tiempo


# =============================================================
# 2-OPT FIRST IMPROVEMENT
# =============================================================
def aplicar_2opt_first_una_iteracion(
    ruta_actual: list,
    tolerancia: float = 1e-9,
    timeout: float = None,
    tiempo_inicio: float = None,
) -> tuple:
    """
    Busca el PRIMER vecino 2-opt que mejore (First Improvement).
    
    Mucho más rápido que Best Improvement, especialmente en instancias grandes.
    En cuanto encuentra un vecino mejor, lo acepta y termina la iteración.

    Devuelve
    --------
    nueva_ruta      : list
    costo_actual    : float
    nuevo_costo     : float
    mejora          : float
    hubo_mejora     : bool
    corto_por_tiempo : bool
    """
    costo_actual = distancia_euclidea_total(ruta_actual)
    n = len(ruta_actual)
    corto_por_tiempo = False

    for i in range(1, n - 2):
        # Verificar timeout entre vecinos externos
        if timeout is not None and tiempo_inicio is not None:
            if (time.perf_counter() - tiempo_inicio) >= timeout:
                corto_por_tiempo = True
                break

        for j in range(i + 1, n - 1):
            nueva = _aplicar_swap_2opt(ruta_actual, i, j)
            nuevo_costo = distancia_euclidea_total(nueva)
            if nuevo_costo < costo_actual - tolerancia:
                # ¡Primera mejora! Aceptar y salir
                mejora = costo_actual - nuevo_costo
                return nueva, costo_actual, nuevo_costo, mejora, True, False

    # No se encontró mejora en toda la pasada
    return ruta_actual, costo_actual, costo_actual, 0.0, False, corto_por_tiempo