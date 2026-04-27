import time
import pandas as pd
from tsp.distancias import distancia_euclidea_total
from tsp.split import split_tsp_dp, reconstruir_camino, stats_mov_a_dataframe
from tsp.dos_opt import (
    aplicar_2opt_una_iteracion,
    aplicar_2opt_first_una_iteracion,
)


# =============================================================
# SPLIT — una iteración
# =============================================================
def aplicar_split_una_iteracion(
    ruta_actual: list,
    usar_m1: bool = True,
    usar_m2: bool = True,
    usar_m3: bool = True,
    usar_m4: bool = True,
) -> tuple:
    """
    Aplica una sola iteración del Split DP sobre la ruta dada.

    Devuelve
    --------
    nueva_ruta, costo_actual, nuevo_costo, mejora, df_stats, stats_mov
    """
    costo_actual = distancia_euclidea_total(ruta_actual)

    V, P, M, stats_mov, _ = split_tsp_dp(
        ruta_actual,
        usar_m1=usar_m1,
        usar_m2=usar_m2,
        usar_m3=usar_m3,
        usar_m4=usar_m4,
    )

    nueva_ruta, stats_mov = reconstruir_camino(ruta_actual, P, M, stats_mov)
    nuevo_costo = distancia_euclidea_total(nueva_ruta)
    mejora = costo_actual - nuevo_costo

    return (
        nueva_ruta,
        costo_actual,
        nuevo_costo,
        mejora,
        stats_mov_a_dataframe(stats_mov),
        stats_mov,
    )


# =============================================================
# SPLIT — búsqueda local iterativa completa
# =============================================================
def busqueda_local_split(
    ruta_inicial: list,
    max_iter: int = 100,
    tolerancia: float = 1e-9,
    timeout: float = None,
    target_costo: float = None,
    usar_m1: bool = True,
    usar_m2: bool = True,
    usar_m3: bool = True,
    usar_m4: bool = True,
) -> tuple:
    """
    Ejecuta Split de forma iterativa hasta que no haya mejora,
    se alcance max_iter, se alcance timeout, o se alcance target_costo.

    Parámetros
    ----------
    target_costo : float, opcional
        Si se alcanza este costo o uno menor, detiene la búsqueda (Time-to-Target)

    Devuelve
    --------
    mejor_ruta, mejor_costo, df_resumen_iteraciones, df_movimientos_iteraciones,
    df_movimientos_acumulado, razon_parada, tiempo_total
    """
    ruta_actual = ruta_inicial.copy()

    acumulado = {
        k: {"nombre": f"M{k}", "evaluado": 0, "mejora_vs_M0": 0,
            "ganador_local": 0, "usado_final": 0}
        for k in range(5)
    }

    resumen_iter = []
    detalle_mov = []

    tiempo_inicio_total = time.perf_counter()
    razon_parada = "max_iter"

    for iteracion in range(max_iter):
        if timeout is not None:
            tiempo_transcurrido = time.perf_counter() - tiempo_inicio_total
            if tiempo_transcurrido >= timeout:
                razon_parada = "timeout"
                break

        tiempo_inicio_iter = time.perf_counter()

        nueva_ruta, c_ini, c_fin, mejora, df_stats, stats_mov = aplicar_split_una_iteracion(
            ruta_actual,
            usar_m1=usar_m1,
            usar_m2=usar_m2,
            usar_m3=usar_m3,
            usar_m4=usar_m4,
        )

        tiempo_iter = time.perf_counter() - tiempo_inicio_iter

        resumen_iter.append({
            "iteracion": iteracion + 1,
            "costo_entrada": c_ini,
            "costo_salida": c_fin,
            "mejora_abs": mejora,
            "mejora_pct": (mejora / c_ini * 100) if c_ini > 0 else 0,
            "hubo_mejora": mejora > tolerancia,
            "tiempo_seg": tiempo_iter,
        })

        df_tmp = df_stats.copy()
        df_tmp["iteracion"] = iteracion + 1
        detalle_mov.append(df_tmp)

        for k in acumulado:
            acumulado[k]["evaluado"] += stats_mov[k]["evaluado"]
            acumulado[k]["mejora_vs_M0"] += stats_mov[k]["mejora_vs_M0"]
            acumulado[k]["ganador_local"] += stats_mov[k]["ganador_local"]
            acumulado[k]["usado_final"] += stats_mov[k]["usado_final"]

        # Verificar target (TTT)
        if target_costo is not None and c_fin <= target_costo:
            razon_parada = "target_alcanzado"
            ruta_actual = nueva_ruta
            break

        if mejora <= tolerancia:
            razon_parada = "convergencia"
            break

        ruta_actual = nueva_ruta

    tiempo_total = time.perf_counter() - tiempo_inicio_total

    mejor_ruta = ruta_actual
    mejor_costo = distancia_euclidea_total(mejor_ruta)

    df_resumen = pd.DataFrame(resumen_iter)
    df_mov_iter = pd.concat(detalle_mov, ignore_index=True) if detalle_mov else pd.DataFrame()
    df_mov_acum = pd.DataFrame([
        {
            "movimiento": acumulado[k]["nombre"],
            "evaluado": acumulado[k]["evaluado"],
            "mejora_vs_M0": acumulado[k]["mejora_vs_M0"],
            "ganador_local": acumulado[k]["ganador_local"],
            "usado_final": acumulado[k]["usado_final"],
        }
        for k in sorted(acumulado)
    ])

    return (
        mejor_ruta,
        mejor_costo,
        df_resumen,
        df_mov_iter,
        df_mov_acum,
        razon_parada,
        tiempo_total,
    )


# =============================================================
# 2-OPT BEST IMPROVEMENT — búsqueda local iterativa
# =============================================================
def busqueda_local_2opt(
    ruta_inicial: list,
    max_iter: int = 100,
    tolerancia: float = 1e-9,
    timeout: float = None,
    target_costo: float = None,
) -> tuple:
    """
    Ejecuta 2-Opt Best Improvement de forma iterativa hasta que no haya mejora,
    se alcance max_iter, se alcance timeout, o se alcance target_costo.

    Devuelve
    --------
    mejor_ruta, mejor_costo, df_resumen_iteraciones, razon_parada, tiempo_total
    """
    ruta_actual = ruta_inicial.copy()
    resumen_iter = []

    tiempo_inicio_total = time.perf_counter()
    razon_parada = "max_iter"

    for iteracion in range(max_iter):
        if timeout is not None:
            tiempo_transcurrido = time.perf_counter() - tiempo_inicio_total
            if tiempo_transcurrido >= timeout:
                razon_parada = "timeout"
                break

        tiempo_inicio_iter = time.perf_counter()

        nueva_ruta, c_ini, c_fin, mejora, hubo_mejora, corto_tiempo = aplicar_2opt_una_iteracion(
            ruta_actual,
            tolerancia=tolerancia,
            timeout=timeout,
            tiempo_inicio=tiempo_inicio_total,
        )

        tiempo_iter = time.perf_counter() - tiempo_inicio_iter

        resumen_iter.append({
            "iteracion": iteracion + 1,
            "costo_entrada": c_ini,
            "costo_salida": c_fin,
            "mejora_abs": mejora,
            "mejora_pct": (mejora / c_ini * 100) if c_ini > 0 else 0,
            "hubo_mejora": hubo_mejora,
            "tiempo_seg": tiempo_iter,
        })

        if corto_tiempo:
            razon_parada = "timeout"
            ruta_actual = nueva_ruta
            break

        if target_costo is not None and c_fin <= target_costo:
            razon_parada = "target_alcanzado"
            ruta_actual = nueva_ruta
            break

        if not hubo_mejora or mejora <= tolerancia:
            razon_parada = "convergencia"
            break

        ruta_actual = nueva_ruta

    tiempo_total = time.perf_counter() - tiempo_inicio_total

    mejor_ruta = ruta_actual
    mejor_costo = distancia_euclidea_total(mejor_ruta)
    df_resumen = pd.DataFrame(resumen_iter)

    return (
        mejor_ruta,
        mejor_costo,
        df_resumen,
        razon_parada,
        tiempo_total,
    )


# =============================================================
# 2-OPT FIRST IMPROVEMENT — búsqueda local iterativa
# =============================================================
def busqueda_local_2opt_first(
    ruta_inicial: list,
    max_iter: int = 100,
    tolerancia: float = 1e-9,
    timeout: float = None,
    target_costo: float = None,
) -> tuple:
    """
    Ejecuta 2-Opt First Improvement de forma iterativa hasta que no haya mejora,
    se alcance max_iter, se alcance timeout, o se alcance target_costo.

    Mucho más rápido que Best Improvement en instancias grandes.

    Devuelve
    --------
    mejor_ruta, mejor_costo, df_resumen_iteraciones, razon_parada, tiempo_total
    """
    ruta_actual = ruta_inicial.copy()
    resumen_iter = []

    tiempo_inicio_total = time.perf_counter()
    razon_parada = "max_iter"

    for iteracion in range(max_iter):
        if timeout is not None:
            tiempo_transcurrido = time.perf_counter() - tiempo_inicio_total
            if tiempo_transcurrido >= timeout:
                razon_parada = "timeout"
                break

        tiempo_inicio_iter = time.perf_counter()

        nueva_ruta, c_ini, c_fin, mejora, hubo_mejora, corto_tiempo = aplicar_2opt_first_una_iteracion(
            ruta_actual,
            tolerancia=tolerancia,
            timeout=timeout,
            tiempo_inicio=tiempo_inicio_total,
        )

        tiempo_iter = time.perf_counter() - tiempo_inicio_iter

        resumen_iter.append({
            "iteracion": iteracion + 1,
            "costo_entrada": c_ini,
            "costo_salida": c_fin,
            "mejora_abs": mejora,
            "mejora_pct": (mejora / c_ini * 100) if c_ini > 0 else 0,
            "hubo_mejora": hubo_mejora,
            "tiempo_seg": tiempo_iter,
        })

        if corto_tiempo:
            razon_parada = "timeout"
            ruta_actual = nueva_ruta
            break

        if target_costo is not None and c_fin <= target_costo:
            razon_parada = "target_alcanzado"
            ruta_actual = nueva_ruta
            break

        if not hubo_mejora or mejora <= tolerancia:
            razon_parada = "convergencia"
            break

        ruta_actual = nueva_ruta

    tiempo_total = time.perf_counter() - tiempo_inicio_total

    mejor_ruta = ruta_actual
    mejor_costo = distancia_euclidea_total(mejor_ruta)
    df_resumen = pd.DataFrame(resumen_iter)

    return (
        mejor_ruta,
        mejor_costo,
        df_resumen,
        razon_parada,
        tiempo_total,
    )