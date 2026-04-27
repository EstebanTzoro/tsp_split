# tsp/vnd.py
"""
Variable Neighborhood Descent (VND) para TSP.

VND explora sistemáticamente diferentes vecindarios en un orden predefinido.
Si encuentra mejora en un vecindario, vuelve al primero (intensificación).
Si no encuentra mejora en ninguno, termina (óptimo local en todos los vecindarios).

Para este proyecto, usamos los movimientos M1-M4 del Split como vecindarios.
Cada Mk se aplica como búsqueda local exhaustiva sobre TODA la ruta usando
el Split DP con SOLO ese movimiento habilitado (más M0 que es el caso base).
"""
import time
import pandas as pd
from tsp.distancias import distancia_euclidea_total
from tsp.busqueda_local import busqueda_local_split


# =============================================================
# VND: Variable Neighborhood Descent
# =============================================================
def vnd_split(
    ruta_inicial: list,
    orden_vecindarios: list = None,
    max_iter_por_vecindario: int = 100,
    tolerancia: float = 1e-9,
    timeout: float = None,
    target_costo: float = None,
) -> tuple:
    """
    Variable Neighborhood Descent usando los movimientos del Split como vecindarios.

    Algoritmo VND clásico:
    1. k = 0 (índice del vecindario actual)
    2. Aplicar búsqueda local con vecindario N_k hasta convergencia
    3. Si hubo mejora respecto a la entrada de N_k:
         - Volver a k = 0 (intensificación)
       Si no hubo mejora:
         - k = k + 1 (siguiente vecindario)
    4. Repetir hasta agotar todos los vecindarios sin mejora

    Parámetros
    ----------
    ruta_inicial : list
        Solución inicial
    orden_vecindarios : list, opcional
        Orden de los vecindarios a explorar. Por defecto ["M1", "M2", "M3", "M4"]
    max_iter_por_vecindario : int
        Iteraciones máximas de búsqueda local por cada vecindario
    tolerancia : float
        Tolerancia para considerar mejora
    timeout : float, opcional
        Tiempo máximo total en segundos
    target_costo : float, opcional
        Si se alcanza este costo o menor, detiene VND

    Devuelve
    --------
    mejor_ruta, mejor_costo, df_resumen, razon_parada, tiempo_total, n_cambios_vecindario
    """
    if orden_vecindarios is None:
        orden_vecindarios = ["M1", "M2", "M3", "M4"]

    # Mapear nombre de vecindario a flags de Split
    def _flags_para_vecindario(nombre):
        flags = {"usar_m1": False, "usar_m2": False, "usar_m3": False, "usar_m4": False}
        if nombre == "M1":
            flags["usar_m1"] = True
        elif nombre == "M2":
            flags["usar_m2"] = True
        elif nombre == "M3":
            flags["usar_m3"] = True
        elif nombre == "M4":
            flags["usar_m4"] = True
        else:
            raise ValueError(f"Vecindario desconocido: {nombre}")
        return flags

    ruta_actual = ruta_inicial.copy()
    costo_actual = distancia_euclidea_total(ruta_actual)

    tiempo_inicio = time.perf_counter()
    razon_parada = "convergencia"

    historial = []
    n_cambios_vecindario = 0
    k = 0  # índice del vecindario actual

    while k < len(orden_vecindarios):
        # Verificar timeout global
        tiempo_transcurrido = time.perf_counter() - tiempo_inicio
        if timeout is not None and tiempo_transcurrido >= timeout:
            razon_parada = "timeout"
            break

        # Verificar target
        if target_costo is not None and costo_actual <= target_costo:
            razon_parada = "target_alcanzado"
            break

        vecindario = orden_vecindarios[k]
        flags = _flags_para_vecindario(vecindario)
        costo_antes = costo_actual

        # Calcular timeout restante para esta búsqueda local
        timeout_restante = None
        if timeout is not None:
            timeout_restante = max(0.001, timeout - tiempo_transcurrido)

        # Aplicar búsqueda local con SOLO este vecindario
        nueva_ruta, nuevo_costo, _, _, _, razon_bl, tiempo_bl = busqueda_local_split(
            ruta_actual,
            max_iter=max_iter_por_vecindario,
            tolerancia=tolerancia,
            timeout=timeout_restante,
            target_costo=target_costo,
            **flags,
        )

        mejora = costo_antes - nuevo_costo
        hubo_mejora = mejora > tolerancia

        historial.append({
            "vecindario": vecindario,
            "costo_entrada": costo_antes,
            "costo_salida": nuevo_costo,
            "mejora_abs": mejora,
            "mejora_pct": (mejora / costo_antes * 100) if costo_antes > 0 else 0,
            "hubo_mejora": hubo_mejora,
            "razon_bl": razon_bl,
            "tiempo_seg": tiempo_bl,
            "k_indice": k,
        })

        if hubo_mejora:
            # Aceptar y volver al primer vecindario
            ruta_actual = nueva_ruta
            costo_actual = nuevo_costo
            if k != 0:
                n_cambios_vecindario += 1
            k = 0
        else:
            # Pasar al siguiente vecindario
            k += 1
            n_cambios_vecindario += 1

        # Si la BL paró por timeout, terminar el VND
        if razon_bl == "timeout":
            razon_parada = "timeout"
            break

    tiempo_total = time.perf_counter() - tiempo_inicio
    df_historial = pd.DataFrame(historial)

    return (
        ruta_actual,
        costo_actual,
        df_historial,
        razon_parada,
        tiempo_total,
        n_cambios_vecindario,
    )