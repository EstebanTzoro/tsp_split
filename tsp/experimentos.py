# tsp/experimentos.py
import os
import time
import random
import pandas as pd

from tsp.io import leer_archivo_tsp, listar_tsp_en_carpeta
from tsp.distancias import distancia_euclidea_total, calcular_mst
from tsp.heuristicas import obtener_heuristica
from tsp.split import split_tsp_dp, reconstruir_camino, stats_mov_a_dataframe
from tsp.busqueda_local import (
    busqueda_local_split,
    busqueda_local_2opt,
    busqueda_local_2opt_first,
)
from tsp.perturbaciones import (
    perturbacion_double_bridge,
    perturbacion_multi_movimiento,
)
from tsp.vnd import vnd_split
from tsp.brkga import brkga
from tsp.visualizacion import (
    dashboard_diagnostico,
    dashboard_busqueda_local,
    dashboard_comparativo,
    dashboard_randomkeys_estabilidad,
    dashboard_perturbacion,
    dashboard_paper_busqueda_local,
    dashboard_paper_ils,
    dashboard_paper_brkga,
)


# =============================================================
# HELPERS
# =============================================================

def _generar_configuraciones_mov() -> list[dict]:
    configs = []
    for m1 in [False, True]:
        for m2 in [False, True]:
            for m3 in [False, True]:
                for m4 in [False, True]:
                    configs.append({
                        "usar_m1": m1, "usar_m2": m2,
                        "usar_m3": m3, "usar_m4": m4,
                        "config": f"M1{int(m1)}_M2{int(m2)}_M3{int(m3)}_M4{int(m4)}",
                    })
    return configs


def _construir_ruta_inicial(nodos, heuristica, seed_randomkeys):
    fn = obtener_heuristica(heuristica)
    if heuristica == "RANDOM_KEYS":
        return fn(nodos, seed=seed_randomkeys)
    return fn(nodos)


def _guardar_excel(ruta_excel: str, hojas: dict):
    with pd.ExcelWriter(ruta_excel, engine="openpyxl") as writer:
        for nombre, df in hojas.items():
            if df is not None and not df.empty:
                df.to_excel(writer, sheet_name=nombre, index=False)


def _rango_nodos(n: int) -> str:
    if n <= 50:       return "01_tiny (<=50)"
    elif n <= 150:    return "02_small (51-150)"
    elif n <= 500:    return "03_medium (151-500)"
    elif n <= 1000:   return "04_large (501-1000)"
    else:             return "05_xlarge (>1000)"


def _verificar_instancia(ruta: str) -> tuple[bool, str]:
    """Verifica que la instancia sea EUC_2D. Devuelve (es_valida, tipo)."""
    try:
        with open(ruta, "r") as f:
            for line in f:
                if "EDGE_WEIGHT_TYPE" in line:
                    tipo = line.split(":")[1].strip()
                    return tipo == "EUC_2D", tipo
    except Exception:
        pass
    return False, "DESCONOCIDO"


# =============================================================
# EXPERIMENTO 1: DIAGNÓSTICO DE MOVIMIENTOS
# =============================================================

def experimento_diagnostico(
    carpeta_entrada: str,
    carpeta_salida: str = "resultados/diagnostico",
    correr_todas_configuraciones: bool = True,
    solo_config: dict = None,
) -> tuple:
    """
    Evalúa el Split DP puro sobre todas las instancias EUC_2D.
    Omite automáticamente instancias con otros tipos de distancia.

    Genera: diagnostico.xlsx + graficos/
    Devuelve: df_resultados, df_movimientos, df_resumen_config
    """
    os.makedirs(carpeta_salida, exist_ok=True)
    archivos = listar_tsp_en_carpeta(carpeta_entrada)
    configuraciones = _generar_configuraciones_mov() if correr_todas_configuraciones else [solo_config]

    print(f"\n{'='*55}")
    print("DIAGNÓSTICO DE MOVIMIENTOS")
    print(f"Instancias encontradas: {len(archivos)} | Configuraciones: {len(configuraciones)}")
    print(f"{'='*55}\n")

    filas_resultados  = []
    filas_movimientos = []

    for config in configuraciones:
        print(f"\n--- CONFIG: {config['config']} ---")
        for ruta in archivos:
            nombre = os.path.splitext(os.path.basename(ruta))[0]
            valida, tipo = _verificar_instancia(ruta)
            if not valida:
                print(f"  OMITIDA {nombre} — tipo {tipo} (requiere EUC_2D)")
                continue
            try:
                nodos = leer_archivo_tsp(ruta)
                costo_inicial = distancia_euclidea_total(nodos)
                V, P, M, stats_mov, _ = split_tsp_dp(
                    nodos,
                    usar_m1=config["usar_m1"], usar_m2=config["usar_m2"],
                    usar_m3=config["usar_m3"], usar_m4=config["usar_m4"],
                )
                camino, stats_mov = reconstruir_camino(nodos, P, M, stats_mov)
                costo_final = distancia_euclidea_total(camino)
                mejora_pct  = (costo_inicial - costo_final) / costo_inicial * 100 if costo_inicial > 0 else 0
                lb_mst      = calcular_mst(nodos)
                gap_pct     = (costo_final - lb_mst) / lb_mst * 100 if lb_mst > 0 else 0

                filas_resultados.append({
                    "instancia": nombre, "n_nodos": len(nodos),
                    "rango_nodos": _rango_nodos(len(nodos)), "config": config["config"],
                    "costo_inicial": round(costo_inicial, 4), "costo_final": round(costo_final, 4),
                    "mejora_pct": round(mejora_pct, 4), "cota_mst": round(lb_mst, 4),
                    "gap_pct": round(gap_pct, 4),
                })
                for k in sorted(stats_mov):
                    filas_movimientos.append({
                        "instancia": nombre, "n_nodos": len(nodos),
                        "rango_nodos": _rango_nodos(len(nodos)), "config": config["config"],
                        "movimiento": stats_mov[k]["nombre"],
                        "evaluado": stats_mov[k]["evaluado"],
                        "mejora_vs_M0": stats_mov[k]["mejora_vs_M0"],
                        "ganador_local": stats_mov[k]["ganador_local"],
                        "usado_final": stats_mov[k]["usado_final"],
                    })
                print(f"  {nombre:<20} n={len(nodos):>5} | fin={costo_final:>12.2f} | mejora={mejora_pct:>7.2f}% | gap={gap_pct:>7.2f}%")
            except Exception as e:
                print(f"  ERROR {nombre}: {e}")

    df_resultados  = pd.DataFrame(filas_resultados)
    df_movimientos = pd.DataFrame(filas_movimientos)
    df_resumen_config = (
        df_resultados.groupby("config", as_index=False)[["mejora_pct", "gap_pct"]]
        .mean().round(4).sort_values("gap_pct")
    ) if not df_resultados.empty else pd.DataFrame()

    _guardar_excel(os.path.join(carpeta_salida, "diagnostico.xlsx"), {
        "resultados": df_resultados, "movimientos": df_movimientos,
        "resumen_config": df_resumen_config,
    })
    print(f"\n✓ Excel guardado en: {carpeta_salida}/diagnostico.xlsx")
    dashboard_diagnostico(df_resultados, df_movimientos, os.path.join(carpeta_salida, "graficos"))
    return df_resultados, df_movimientos, df_resumen_config


# =============================================================
# EXPERIMENTO 2: BÚSQUEDA LOCAL CON HEURÍSTICAS
# =============================================================

def experimento_busqueda_local(
    carpeta_entrada: str,
    carpeta_salida: str = "resultados/busqueda_local",
    heuristicas_a_probar: list = None,
    correr_todas_configuraciones: bool = True,
    solo_config: dict = None,
    max_iter: int = 100,
    tolerancia: float = 1e-9,
    timeout: float = None,
    seed_randomkeys: int = 11,
) -> tuple:
    """
    Búsqueda local iterativa con Split.
    Omite instancias no EUC_2D automáticamente.
    Registra razón de parada: convergencia | max_iter | timeout.

    Parámetros
    ----------
    timeout : float o None — segundos máximos por instancia

    Genera: busqueda_local.xlsx + graficos/
    Devuelve: df_resultados, df_movimientos, df_convergencia, df_resumen_config
    """
    os.makedirs(carpeta_salida, exist_ok=True)
    archivos = listar_tsp_en_carpeta(carpeta_entrada)
    heuristicas_a_probar = heuristicas_a_probar or ["NNH", "INSERCION", "RANDOM_KEYS"]
    configuraciones = _generar_configuraciones_mov() if correr_todas_configuraciones else [solo_config]

    print(f"\n{'='*55}")
    print("BÚSQUEDA LOCAL CON HEURÍSTICAS")
    print(f"Instancias: {len(archivos)} | Heurísticas: {len(heuristicas_a_probar)} | Configs: {len(configuraciones)}")
    if timeout:
        print(f"Timeout por corrida: {timeout}s")
    print(f"{'='*55}\n")

    filas_resultados   = []
    filas_movimientos  = []
    filas_convergencia = []

    for config in configuraciones:
        print(f"\n===== CONFIG: {config['config']} =====")
        for heuristica in heuristicas_a_probar:
            for ruta in archivos:
                nombre = os.path.splitext(os.path.basename(ruta))[0]
                valida, tipo = _verificar_instancia(ruta)
                if not valida:
                    print(f"  OMITIDA {nombre} — tipo {tipo}")
                    continue

                try:
                    nodos = leer_archivo_tsp(ruta)
                    ruta_ini, heur_stats = _construir_ruta_inicial(nodos, heuristica, seed_randomkeys)
                    costo_inicial = distancia_euclidea_total(ruta_ini)
                    lb_mst = calcular_mst(nodos)

                    mejor_ruta, mejor_costo, df_res_iter, df_mov_iter, df_mov_acum, razon_parada, tiempo_total = busqueda_local_split(
                        ruta_ini,
                        max_iter=max_iter,
                        tolerancia=tolerancia,
                        timeout=timeout,
                        usar_m1=config["usar_m1"],
                        usar_m2=config["usar_m2"],
                        usar_m3=config["usar_m3"],
                        usar_m4=config["usar_m4"],
                    )

                    n_iter = len(df_res_iter)
                    mejora_pct = (costo_inicial - mejor_costo) / costo_inicial * 100 if costo_inicial > 0 else 0
                    gap_pct    = (mejor_costo - lb_mst) / lb_mst * 100 if lb_mst > 0 else 0

                    filas_resultados.append({
                        "instancia": nombre, "n_nodos": len(nodos),
                        "rango_nodos": _rango_nodos(len(nodos)), "heuristica": heuristica,
                        "config": config["config"], "costo_inicial": round(costo_inicial, 4),
                        "costo_final": round(mejor_costo, 4), "mejora_abs": round(costo_inicial - mejor_costo, 4),
                        "mejora_pct": round(mejora_pct, 4), "cota_mst": round(lb_mst, 4),
                        "gap_pct": round(gap_pct, 4), "iteraciones_ejecutadas": n_iter,
                        "tiempo_seg": round(tiempo_total, 4), "razon_parada": razon_parada,
                    })

                    for _, row in df_mov_acum.iterrows():
                        filas_movimientos.append({
                            "instancia": nombre, "n_nodos": len(nodos),
                            "rango_nodos": _rango_nodos(len(nodos)), "heuristica": heuristica,
                            "config": config["config"], "movimiento": row["movimiento"],
                            "evaluado": row["evaluado"], "mejora_vs_M0": row["mejora_vs_M0"],
                            "ganador_local": row["ganador_local"], "usado_final": row["usado_final"],
                        })

                    for _, row in df_res_iter.iterrows():
                        filas_convergencia.append({
                            "instancia": nombre, "n_nodos": len(nodos),
                            "rango_nodos": _rango_nodos(len(nodos)), "heuristica": heuristica,
                            "config": config["config"], "iteracion": row["iteracion"],
                            "costo_salida": round(row["costo_salida"], 4),
                            "mejora_pct": round(row["mejora_pct"], 4),
                            "tiempo_seg": row.get("tiempo_seg", None),
                        })

                    print(
                        f"  {nombre:<18} {heuristica:<12} {config['config']:<16} | "
                        f"iters={n_iter:>3} t={tiempo_total:>7.2f}s [{razon_parada:<13}] | gap={gap_pct:>6.2f}%"
                    )

                except Exception as e:
                    print(f"  ERROR {nombre} ({heuristica}): {e}")

    df_resultados   = pd.DataFrame(filas_resultados)
    df_movimientos  = pd.DataFrame(filas_movimientos)
    df_convergencia = pd.DataFrame(filas_convergencia)
    df_resumen_config = (
        df_resultados.groupby(["heuristica", "config"], as_index=False)
        [["mejora_pct", "gap_pct", "iteraciones_ejecutadas"]].mean().round(4)
        .sort_values(["heuristica", "gap_pct"])
    ) if not df_resultados.empty else pd.DataFrame()

    _guardar_excel(os.path.join(carpeta_salida, "busqueda_local.xlsx"), {
        "resultados": df_resultados, "resumen_config": df_resumen_config,
        "movimientos": df_movimientos, "convergencia": df_convergencia,
    })
    print(f"\n✓ Excel guardado en: {carpeta_salida}/busqueda_local.xlsx")

    if not df_resultados.empty:
        instancia_ejemplo = df_resultados.sort_values("mejora_pct", ascending=False).iloc[0]["instancia"]
        dashboard_busqueda_local(
            df_resumen=df_resultados, df_iteraciones=df_convergencia,
            df_movimientos_agregado=df_movimientos, df_resumen_config=df_resumen_config,
            carpeta_salida=os.path.join(carpeta_salida, "graficos"), instancia_ejemplo=instancia_ejemplo,
        )

    return df_resultados, df_movimientos, df_convergencia, df_resumen_config


# =============================================================
# EXPERIMENTO 3: COMPARATIVO SPLIT vs 2-OPT
# =============================================================

def experimento_comparativo(
    carpeta_entrada: str,
    carpeta_salida: str = "resultados/comparativo",
    heuristicas_a_probar: list = None,
    correr_todas_configuraciones: bool = True,
    solo_config: dict = None,
    max_iter: int = 100,
    tolerancia: float = 1e-9,
    timeout: float = None,
    seed_randomkeys: int = 11,
) -> tuple:
    """
    Compara Split vs 2-Opt sobre las mismas instancias y heurísticas.
    Omite instancias no EUC_2D automáticamente.

    Genera: comparativo.xlsx + graficos/
    Devuelve: df_resultados, df_resumen, df_por_tamano, df_convergencia_agg
    """
    os.makedirs(carpeta_salida, exist_ok=True)
    archivos = listar_tsp_en_carpeta(carpeta_entrada)
    heuristicas_a_probar = heuristicas_a_probar or ["NNH", "INSERCION", "RANDOM_KEYS"]
    configuraciones = _generar_configuraciones_mov() if correr_todas_configuraciones else [solo_config]

    print(f"\n{'='*55}")
    print("COMPARATIVO SPLIT vs 2-OPT")
    print(f"Instancias: {len(archivos)} | Heurísticas: {len(heuristicas_a_probar)} | Configs Split: {len(configuraciones)}")
    if timeout:
        print(f"Timeout por corrida: {timeout}s")
    print(f"{'='*55}\n")

    filas_resultados   = []
    filas_convergencia = []

    for config in configuraciones:
        print(f"\n===== CONFIG SPLIT: {config['config']} =====")
        for heuristica in heuristicas_a_probar:
            for ruta in archivos:
                nombre = os.path.splitext(os.path.basename(ruta))[0]
                valida, tipo = _verificar_instancia(ruta)
                if not valida:
                    print(f"  OMITIDA {nombre} — tipo {tipo}")
                    continue

                try:
                    nodos = leer_archivo_tsp(ruta)
                    ruta_ini, _ = _construir_ruta_inicial(nodos, heuristica, seed_randomkeys)
                    costo_inicial = distancia_euclidea_total(ruta_ini)
                    lb_mst = calcular_mst(nodos)

                    # --- SPLIT ---
                    mejor_ruta_split, mejor_costo_split, df_res_split, _, _, razon_split, tiempo_split = busqueda_local_split(
                        ruta_ini[:],
                        max_iter=max_iter,
                        tolerancia=tolerancia,
                        timeout=timeout,
                        usar_m1=config["usar_m1"],
                        usar_m2=config["usar_m2"],
                        usar_m3=config["usar_m3"],
                        usar_m4=config["usar_m4"],
                    )
                    n_iter_split = len(df_res_split)
                    mejora_pct_split = (costo_inicial - mejor_costo_split) / costo_inicial * 100 if costo_inicial > 0 else 0
                    gap_pct_split    = (mejor_costo_split - lb_mst) / lb_mst * 100 if lb_mst > 0 else 0

                    # --- 2-OPT ---
                    mejor_ruta_2opt, mejor_costo_2opt, df_res_2opt, razon_2opt, tiempo_2opt = busqueda_local_2opt(
                        ruta_ini[:],
                        max_iter=max_iter,
                        tolerancia=tolerancia,
                        timeout=timeout,
                    )
                    n_iter_2opt = len(df_res_2opt)
                    mejora_pct_2opt = (costo_inicial - mejor_costo_2opt) / costo_inicial * 100 if costo_inicial > 0 else 0
                    gap_pct_2opt    = (mejor_costo_2opt - lb_mst) / lb_mst * 100 if lb_mst > 0 else 0

                    ventaja_split = mejor_costo_2opt - mejor_costo_split
                    ventaja_pct   = (ventaja_split / mejor_costo_2opt * 100) if mejor_costo_2opt > 0 else 0
                    gana_split    = 1 if mejor_costo_split < mejor_costo_2opt else 0
                    gana_2opt     = 1 if mejor_costo_2opt < mejor_costo_split else 0
                    empate        = 1 if abs(mejor_costo_split - mejor_costo_2opt) < 1e-6 else 0

                    filas_resultados.append({
                        "instancia": nombre, "n_nodos": len(nodos),
                        "rango_nodos": _rango_nodos(len(nodos)), "heuristica": heuristica,
                        "config_split": config["config"], "costo_inicial": round(costo_inicial, 4),
                        "costo_split": round(mejor_costo_split, 4), "costo_2opt": round(mejor_costo_2opt, 4),
                        "mejora_pct_split": round(mejora_pct_split, 4), "mejora_pct_2opt": round(mejora_pct_2opt, 4),
                        "gap_pct_split": round(gap_pct_split, 4), "gap_pct_2opt": round(gap_pct_2opt, 4),
                        "ventaja_split": round(ventaja_split, 4), "ventaja_pct": round(ventaja_pct, 4),
                        "gana_split": gana_split, "gana_2opt": gana_2opt, "empate": empate,
                        "tiempo_split_seg": round(tiempo_split, 4), "tiempo_2opt_seg": round(tiempo_2opt, 4),
                        "iters_split": n_iter_split, "iters_2opt": n_iter_2opt,
                        "razon_split": razon_split, "razon_2opt": razon_2opt,
                        "cota_mst": round(lb_mst, 4),
                    })

                    # Convergencia
                    for _, row in df_res_split.iterrows():
                        filas_convergencia.append({
                            "instancia": nombre, "n_nodos": len(nodos),
                            "rango_nodos": _rango_nodos(len(nodos)), "heuristica": heuristica,
                            "config_split": config["config"], "metodo": "SPLIT",
                            "iteracion": row["iteracion"], "costo_salida": round(row["costo_salida"], 4),
                            "mejora_pct": round(row["mejora_pct"], 4),
                        })
                    for _, row in df_res_2opt.iterrows():
                        filas_convergencia.append({
                            "instancia": nombre, "n_nodos": len(nodos),
                            "rango_nodos": _rango_nodos(len(nodos)), "heuristica": heuristica,
                            "config_split": config["config"], "metodo": "2OPT",
                            "iteracion": row["iteracion"], "costo_salida": round(row["costo_salida"], 4),
                            "mejora_pct": round(row["mejora_pct"], 4),
                        })

                    print(
                        f"  {nombre:<18} {heuristica:<12} | Split: gap={gap_pct_split:>6.2f}% | "
                        f"2opt: gap={gap_pct_2opt:>6.2f}% | ventaja: {ventaja_pct:>+6.2f}%"
                    )

                except Exception as e:
                    print(f"  ERROR {nombre} ({heuristica}): {e}")

    df_resultados = pd.DataFrame(filas_resultados)

    df_por_tamano = (
        df_resultados.groupby("rango_nodos", as_index=False)
        [["gap_pct_split", "gap_pct_2opt", "ventaja_split", "gana_split", "gana_2opt", "empate",
          "tiempo_split_seg", "tiempo_2opt_seg", "iters_split", "iters_2opt"]].mean().round(4)
        .sort_values("rango_nodos")
    ) if not df_resultados.empty else pd.DataFrame()

    df_resumen = (
        df_resultados.groupby(["heuristica", "config_split"], as_index=False)
        [["gap_pct_split", "gap_pct_2opt", "mejora_pct_split", "mejora_pct_2opt",
          "ventaja_split", "gana_split", "gana_2opt", "empate",
          "tiempo_split_seg", "tiempo_2opt_seg", "iters_split", "iters_2opt"]].mean().round(4)
        .sort_values(["heuristica", "gap_pct_split"])
    ) if not df_resultados.empty else pd.DataFrame()

    df_convergencia = pd.DataFrame(filas_convergencia)
    df_convergencia_agg = (
        df_convergencia.groupby(["heuristica", "config_split", "metodo", "iteracion"], as_index=False)
        [["costo_salida", "mejora_pct"]].mean().round(4)
        .sort_values(["heuristica", "config_split", "metodo", "iteracion"])
    ) if not df_convergencia.empty else pd.DataFrame()

    _guardar_excel(os.path.join(carpeta_salida, "comparativo.xlsx"), {
        "resultados": df_resultados, "por_tamano": df_por_tamano,
        "convergencia": df_convergencia_agg, "resumen": df_resumen,
    })
    print(f"\n✓ Excel guardado en: {carpeta_salida}/comparativo.xlsx")

    if not df_resultados.empty:
        dashboard_comparativo(
            df_comparativo=df_resultados, df_iter_agregadas=df_convergencia_agg,
            carpeta_salida=os.path.join(carpeta_salida, "graficos"),
        )

    return df_resultados, df_resumen, df_por_tamano, df_convergencia_agg

# =============================================================
# EXPERIMENTO 4: RANDOM_KEYS MULTISEMILLA
# =============================================================

def experimento_randomkeys_estabilidad(
    carpeta_entrada: str,
    carpeta_salida: str = "resultados/randomkeys_estabilidad",
    correr_todas_configuraciones: bool = True,
    solo_config: dict = None,
    max_iter: int = 100,
    tolerancia: float = 1e-9,
    timeout: float = None,
    semilla_inicio: int = 11,
    n_semillas: int = 20,
) -> tuple:
    """
    Evalúa la estabilidad de RANDOM_KEYS usando múltiples semillas.

    Para cada instancia EUC_2D, para cada semilla y para cada configuración
    de movimientos, construye una ruta inicial con RANDOM_KEYS y aplica
    búsqueda local iterativa con Split.

    Genera:
      - randomkeys_estabilidad.xlsx
      - graficos/

    Devuelve:
      df_resultados, df_resumen_config, df_convergencia_agg
    """
    os.makedirs(carpeta_salida, exist_ok=True)
    archivos = listar_tsp_en_carpeta(carpeta_entrada)
    configuraciones = _generar_configuraciones_mov() if correr_todas_configuraciones else [solo_config]
    semillas = list(range(semilla_inicio, semilla_inicio + n_semillas))

    print(f"\n{'='*55}")
    print("RANDOM_KEYS MULTISEMILLA")
    print(f"Instancias: {len(archivos)} | Configs: {len(configuraciones)} | Semillas: {len(semillas)}")
    if timeout:
        print(f"Timeout por corrida: {timeout}s")
    print(f"{'='*55}\n")

    filas_resultados = []
    filas_convergencia = []

    for config in configuraciones:
        print(f"\n===== CONFIG: {config['config']} =====")
        for ruta in archivos:
            nombre = os.path.splitext(os.path.basename(ruta))[0]
            valida, tipo = _verificar_instancia(ruta)
            if not valida:
                print(f"  OMITIDA {nombre} — tipo {tipo}")
                continue

            try:
                nodos = leer_archivo_tsp(ruta)
                lb_mst = calcular_mst(nodos)

                for semilla in semillas:
                    ruta_ini, _ = _construir_ruta_inicial(nodos, "RANDOM_KEYS", semilla)
                    costo_inicial = distancia_euclidea_total(ruta_ini)

                    mejor_ruta, mejor_costo, df_res_iter, df_mov_iter, df_mov_acum, razon_parada, tiempo_total = busqueda_local_split(
                        ruta_ini,
                        max_iter=max_iter,
                        tolerancia=tolerancia,
                        timeout=timeout,
                        usar_m1=config["usar_m1"],
                        usar_m2=config["usar_m2"],
                        usar_m3=config["usar_m3"],
                        usar_m4=config["usar_m4"],
                    )

                    n_iter = len(df_res_iter)
                    mejora_pct = (costo_inicial - mejor_costo) / costo_inicial * 100 if costo_inicial > 0 else 0
                    gap_pct = (mejor_costo - lb_mst) / lb_mst * 100 if lb_mst > 0 else 0

                    filas_resultados.append({
                        "instancia": nombre,
                        "n_nodos": len(nodos),
                        "rango_nodos": _rango_nodos(len(nodos)),
                        "heuristica": "RANDOM_KEYS",
                        "semilla": semilla,
                        "config": config["config"],
                        "costo_inicial": round(costo_inicial, 4),
                        "costo_final": round(mejor_costo, 4),
                        "mejora_abs": round(costo_inicial - mejor_costo, 4),
                        "mejora_pct": round(mejora_pct, 4),
                        "cota_mst": round(lb_mst, 4),
                        "gap_pct": round(gap_pct, 4),
                        "iteraciones_ejecutadas": n_iter,
                        "tiempo_seg": round(tiempo_total, 4),
                        "razon_parada": razon_parada,
                    })

                    for _, row in df_res_iter.iterrows():
                        filas_convergencia.append({
                            "instancia": nombre,
                            "n_nodos": len(nodos),
                            "rango_nodos": _rango_nodos(len(nodos)),
                            "heuristica": "RANDOM_KEYS",
                            "semilla": semilla,
                            "config": config["config"],
                            "iteracion": row["iteracion"],
                            "costo_salida": round(row["costo_salida"], 4),
                            "mejora_pct": round(row["mejora_pct"], 4),
                            "tiempo_seg": row.get("tiempo_seg", None),
                        })

                    print(
                        f"  {nombre:<18} seed={semilla:<4} {config['config']:<16} "
                        f"| iters={n_iter:>3} t={tiempo_total:>7.2f}s [{razon_parada:<13}] "
                        f"| gap={gap_pct:>6.2f}%"
                    )

            except Exception as e:
                print(f"  ERROR {nombre}: {e}")

    df_resultados = pd.DataFrame(filas_resultados)
    df_convergencia = pd.DataFrame(filas_convergencia)

    df_resumen_config = (
        df_resultados.groupby("config", as_index=False)
        [["gap_pct", "mejora_pct", "iteraciones_ejecutadas", "tiempo_seg"]]
        .agg(["mean", "std", "median"])
        .round(4)
    ) if not df_resultados.empty else pd.DataFrame()

    if not df_resumen_config.empty:
        df_resumen_config.columns = [
            "config" if c[0] == "config" else f"{c[0]}_{c[1]}"
            for c in df_resumen_config.columns.to_flat_index()
        ]

    df_resumen_instancia = (
        df_resultados.groupby(["instancia", "config"], as_index=False)
        [["gap_pct", "mejora_pct", "iteraciones_ejecutadas", "tiempo_seg"]]
        .mean().round(4)
    ) if not df_resultados.empty else pd.DataFrame()

    df_convergencia_agg = (
        df_convergencia.groupby(["config", "iteracion"], as_index=False)
        [["costo_salida", "mejora_pct"]]
        .mean().round(4)
        .sort_values(["config", "iteracion"])
    ) if not df_convergencia.empty else pd.DataFrame()

    _guardar_excel(os.path.join(carpeta_salida, "randomkeys_estabilidad.xlsx"), {
        "resultados": df_resultados,
        "resumen_config": df_resumen_config,
        "resumen_instancia": df_resumen_instancia,
        "convergencia": df_convergencia_agg,
    })
    print(f"\n✓ Excel guardado en: {carpeta_salida}/randomkeys_estabilidad.xlsx")

    if not df_resultados.empty:
        dashboard_randomkeys_estabilidad(
            df_resultados=df_resultados,
            df_convergencia=df_convergencia_agg,
            carpeta_salida=os.path.join(carpeta_salida, "graficos"),
        )

    return df_resultados, df_resumen_config, df_convergencia_agg


# =============================================================
# EXPERIMENTO 5: BÚSQUEDA LOCAL CON PERTURBACIÓN ALEATORIA
# =============================================================

def _perturbacion_aleatoria(ruta: list, n_movimientos: int, movimientos_habilitados: list, seed: int = None) -> list:
    """
    Perturba una ruta aplicando n_movimientos aleatorios.
    
    Para cada movimiento:
    1. Escoge aleatoriamente i y j (i < j)
    2. Escoge aleatoriamente un movimiento de la lista de movimientos habilitados
    3. Aplica el movimiento entre i y j
    
    Parámetros
    ----------
    ruta : list
        Secuencia de nodos a perturbar
    n_movimientos : int
        Número de movimientos aleatorios a aplicar
    movimientos_habilitados : list
        Lista de movimientos disponibles, e.g., ["M0", "M1"]
    seed : int, opcional
        Semilla para reproducibilidad
        
    Devuelve
    --------
    list : ruta perturbada
    """
    if seed is not None:
        random.seed(seed)
    
    ruta_perturbada = ruta[:]
    n = len(ruta_perturbada)
    
    for _ in range(n_movimientos):
        # Escoger aleatoriamente i y j tal que i < j
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        
        # Escoger aleatoriamente un movimiento
        mov = random.choice(movimientos_habilitados)
        
        # Aplicar el movimiento
        segmento = ruta_perturbada[i:j+1]
        
        if mov == "M0":
            # M0: mantener orden original (no hacer nada)
            pass
        elif mov == "M1":
            # M1: invertir el segmento
            segmento = segmento[::-1]
            ruta_perturbada[i:j+1] = segmento
        # Aquí se pueden agregar más movimientos si se habilitan
        # elif mov == "M2":
        #     ...
    
    return ruta_perturbada


def experimento_perturbacion(
    carpeta_entrada: str,
    carpeta_salida: str = "resultados/perturbacion",
    heuristicas_a_probar: list = None,
    solo_config: dict = None,
    max_iter_bl: int = 100,
    max_iter_perturb: int = 10,
    n_movimientos_perturb: int = 5,
    tolerancia: float = 1e-9,
    timeout: float = None,
    seed_randomkeys: int = 11,
) -> tuple:
    """
    Búsqueda local con perturbación aleatoria.
    
    Algoritmo:
    1. Construir ruta inicial con heurística
    2. Aplicar búsqueda local con Split hasta convergencia
    3. Si no se alcanzó max_iter_perturb:
       a. Perturbar la mejor ruta encontrada
       b. Aplicar búsqueda local nuevamente
       c. Si mejora, actualizar mejor solución
       d. Repetir
    
    Parámetros
    ----------
    max_iter_bl : int
        Iteraciones máximas de búsqueda local por cada fase
    max_iter_perturb : int
        Número de perturbaciones a realizar
    n_movimientos_perturb : int
        Número de movimientos aleatorios por perturbación
        
    Genera: perturbacion.xlsx + graficos/
    Devuelve: df_resultados, df_convergencia, df_perturbaciones, df_resumen_config
    """
    os.makedirs(carpeta_salida, exist_ok=True)
    archivos = listar_tsp_en_carpeta(carpeta_entrada)
    heuristicas_a_probar = heuristicas_a_probar or ["NNH", "INSERCION", "RANDOM_KEYS"]
    
    # Para perturbación solo usamos la configuración especificada
    if solo_config is None:
        raise ValueError("experimento_perturbacion requiere especificar solo_config en el YAML")
    
    # Determinar qué movimientos están habilitados
    movimientos_habilitados = ["M0"]  # M0 siempre está
    if solo_config.get("usar_m1", False):
        movimientos_habilitados.append("M1")
    # Aquí se pueden agregar M2, M3, M4 si se habilitan en el futuro

    print(f"\n{'='*55}")
    print("BÚSQUEDA LOCAL CON PERTURBACIÓN ALEATORIA")
    print(f"Instancias: {len(archivos)} | Heurísticas: {len(heuristicas_a_probar)}")
    print(f"Config: {solo_config['config']} | Movimientos habilitados: {movimientos_habilitados}")
    print(f"Max iter BL: {max_iter_bl} | Max perturbaciones: {max_iter_perturb}")
    print(f"Movimientos por perturbación: {n_movimientos_perturb}")
    if timeout:
        print(f"Timeout por corrida: {timeout}s")
    print(f"{'='*55}\n")

    filas_resultados = []
    filas_convergencia = []
    filas_perturbaciones = []

    for heuristica in heuristicas_a_probar:
        for ruta in archivos:
            nombre = os.path.splitext(os.path.basename(ruta))[0]
            valida, tipo = _verificar_instancia(ruta)
            if not valida:
                print(f"  OMITIDA {nombre} — tipo {tipo}")
                continue

            try:
                nodos = leer_archivo_tsp(ruta)
                ruta_ini, _ = _construir_ruta_inicial(nodos, heuristica, seed_randomkeys)
                costo_inicial = distancia_euclidea_total(ruta_ini)
                lb_mst = calcular_mst(nodos)

                tiempo_inicio_total = time.time()
                
                # Primera búsqueda local
                mejor_ruta_global = ruta_ini[:]
                mejor_costo_global = costo_inicial
                
                mejor_ruta, mejor_costo, df_res_iter, _, _, razon, tiempo_bl = busqueda_local_split(
                    ruta_ini,
                    max_iter=max_iter_bl,
                    tolerancia=tolerancia,
                    timeout=timeout,
                    usar_m1=solo_config["usar_m1"],
                    usar_m2=solo_config.get("usar_m2", False),
                    usar_m3=solo_config.get("usar_m3", False),
                    usar_m4=solo_config.get("usar_m4", False),
                )
                
                mejor_ruta_global = mejor_ruta[:]
                mejor_costo_global = mejor_costo
                
                # Guardar convergencia inicial (fase 0)
                for _, row in df_res_iter.iterrows():
                    filas_convergencia.append({
                        "instancia": nombre,
                        "n_nodos": len(nodos),
                        "rango_nodos": _rango_nodos(len(nodos)),
                        "heuristica": heuristica,
                        "config": solo_config["config"],
                        "fase_perturbacion": 0,
                        "iteracion": row["iteracion"],
                        "costo_salida": round(row["costo_salida"], 4),
                        "mejora_pct": round(row["mejora_pct"], 4),
                    })
                
                # Ciclo de perturbaciones
                n_perturbaciones_aplicadas = 0
                mejoras_por_perturbacion = 0
                
                for fase_perturb in range(1, max_iter_perturb + 1):
                    # Verificar timeout
                    tiempo_transcurrido = time.time() - tiempo_inicio_total
                    if timeout and tiempo_transcurrido > timeout:
                        razon = "timeout"
                        break
                    
                    # Perturbar la mejor solución actual
                    ruta_perturbada = _perturbacion_aleatoria(
                        mejor_ruta_global,
                        n_movimientos=n_movimientos_perturb,
                        movimientos_habilitados=movimientos_habilitados,
                        seed=seed_randomkeys + fase_perturb  # Semilla diferente por fase
                    )
                    costo_perturbado = distancia_euclidea_total(ruta_perturbada)
                    
                    # Aplicar búsqueda local desde la ruta perturbada
                    ruta_bl, costo_bl, df_res_iter, _, _, razon_bl, tiempo_bl = busqueda_local_split(
                        ruta_perturbada,
                        max_iter=max_iter_bl,
                        tolerancia=tolerancia,
                        timeout=timeout - tiempo_transcurrido if timeout else None,
                        usar_m1=solo_config["usar_m1"],
                        usar_m2=solo_config.get("usar_m2", False),
                        usar_m3=solo_config.get("usar_m3", False),
                        usar_m4=solo_config.get("usar_m4", False),
                    )
                    
                    n_perturbaciones_aplicadas += 1
                    mejoro = costo_bl < mejor_costo_global
                    
                    if mejoro:
                        mejoras_por_perturbacion += 1
                        mejor_ruta_global = ruta_bl[:]
                        mejor_costo_global = costo_bl
                    
                    # Registrar info de perturbación
                    filas_perturbaciones.append({
                        "instancia": nombre,
                        "n_nodos": len(nodos),
                        "heuristica": heuristica,
                        "config": solo_config["config"],
                        "fase_perturbacion": fase_perturb,
                        "costo_antes_perturbar": round(costo_perturbado, 4),
                        "costo_despues_bl": round(costo_bl, 4),
                        "costo_mejor_global": round(mejor_costo_global, 4),
                        "mejoro_global": mejoro,
                        "iteraciones_bl": len(df_res_iter),
                        "tiempo_bl_seg": round(tiempo_bl, 4),
                    })
                    
                    # Guardar convergencia de esta fase
                    for _, row in df_res_iter.iterrows():
                        filas_convergencia.append({
                            "instancia": nombre,
                            "n_nodos": len(nodos),
                            "rango_nodos": _rango_nodos(len(nodos)),
                            "heuristica": heuristica,
                            "config": solo_config["config"],
                            "fase_perturbacion": fase_perturb,
                            "iteracion": row["iteracion"],
                            "costo_salida": round(row["costo_salida"], 4),
                            "mejora_pct": round(row["mejora_pct"], 4),
                        })

                tiempo_total = time.time() - tiempo_inicio_total
                mejora_pct_final = (costo_inicial - mejor_costo_global) / costo_inicial * 100 if costo_inicial > 0 else 0
                gap_pct_final = (mejor_costo_global - lb_mst) / lb_mst * 100 if lb_mst > 0 else 0

                filas_resultados.append({
                    "instancia": nombre,
                    "n_nodos": len(nodos),
                    "rango_nodos": _rango_nodos(len(nodos)),
                    "heuristica": heuristica,
                    "config": solo_config["config"],
                    "costo_inicial": round(costo_inicial, 4),
                    "costo_final": round(mejor_costo_global, 4),
                    "mejora_abs": round(costo_inicial - mejor_costo_global, 4),
                    "mejora_pct": round(mejora_pct_final, 4),
                    "cota_mst": round(lb_mst, 4),
                    "gap_pct": round(gap_pct_final, 4),
                    "n_perturbaciones": n_perturbaciones_aplicadas,
                    "n_mejoras": mejoras_por_perturbacion,
                    "tiempo_total_seg": round(tiempo_total, 4),
                    "razon_parada": razon,
                })

                print(
                    f"  {nombre:<18} {heuristica:<12} | perturb={n_perturbaciones_aplicadas:>2} "
                    f"mejoras={mejoras_por_perturbacion:>2} | gap={gap_pct_final:>6.2f}% t={tiempo_total:>7.2f}s"
                )

            except Exception as e:
                print(f"  ERROR {nombre} ({heuristica}): {e}")

    df_resultados = pd.DataFrame(filas_resultados)
    df_convergencia = pd.DataFrame(filas_convergencia)
    df_perturbaciones = pd.DataFrame(filas_perturbaciones)
    
    df_resumen_config = (
        df_resultados.groupby(["heuristica", "config"], as_index=False)
        [["mejora_pct", "gap_pct", "n_perturbaciones", "n_mejoras", "tiempo_total_seg"]]
        .mean().round(4)
        .sort_values(["heuristica", "gap_pct"])
    ) if not df_resultados.empty else pd.DataFrame()

    _guardar_excel(os.path.join(carpeta_salida, "perturbacion.xlsx"), {
        "resultados": df_resultados,
        "resumen_config": df_resumen_config,
        "perturbaciones": df_perturbaciones,
        "convergencia": df_convergencia,
    })
    print(f"\n✓ Excel guardado en: {carpeta_salida}/perturbacion.xlsx")

    if not df_resultados.empty:
        dashboard_perturbacion(
            df_resultados=df_resultados,
            df_convergencia=df_convergencia,
            df_perturbaciones=df_perturbaciones,
            carpeta_salida=os.path.join(carpeta_salida, "graficos"),
        )

    return df_resultados, df_convergencia, df_perturbaciones, df_resumen_config

# =============================================================
# EXPERIMENTO PAPER 1: BÚSQUEDA LOCAL COMPARATIVA
# =============================================================

def _ejecutar_metodo_busqueda_local(
    metodo: str,
    ruta_inicial: list,
    max_iter: int,
    tolerancia: float,
    timeout: float = None,
    target_costo: float = None,
) -> dict:
    """
    Ejecuta un método de búsqueda local sobre una ruta inicial.

    Métodos soportados:
        - "2OPT_BEST"  : 2-opt Best Improvement
        - "2OPT_FIRST" : 2-opt First Improvement
        - "SPLIT_M01"  : Split con solo M0 y M1
        - "SPLIT_FULL" : Split con todos los movimientos (M0-M4)
        - "VND"        : Variable Neighborhood Descent (M1-M4)

    Devuelve dict con: ruta, costo, df_iter, razon_parada, tiempo_seg
    """
    if metodo == "2OPT_BEST":
        ruta, costo, df_iter, razon, t = busqueda_local_2opt(
            ruta_inicial, max_iter=max_iter, tolerancia=tolerancia,
            timeout=timeout, target_costo=target_costo,
        )
        return {"ruta": ruta, "costo": costo, "df_iter": df_iter,
                "razon_parada": razon, "tiempo_seg": t}

    elif metodo == "2OPT_FIRST":
        ruta, costo, df_iter, razon, t = busqueda_local_2opt_first(
            ruta_inicial, max_iter=max_iter, tolerancia=tolerancia,
            timeout=timeout, target_costo=target_costo,
        )
        return {"ruta": ruta, "costo": costo, "df_iter": df_iter,
                "razon_parada": razon, "tiempo_seg": t}

    elif metodo == "SPLIT_M01":
        ruta, costo, df_iter, _, _, razon, t = busqueda_local_split(
            ruta_inicial, max_iter=max_iter, tolerancia=tolerancia,
            timeout=timeout, target_costo=target_costo,
            usar_m1=True, usar_m2=False, usar_m3=False, usar_m4=False,
        )
        return {"ruta": ruta, "costo": costo, "df_iter": df_iter,
                "razon_parada": razon, "tiempo_seg": t}

    elif metodo == "SPLIT_FULL":
        ruta, costo, df_iter, _, _, razon, t = busqueda_local_split(
            ruta_inicial, max_iter=max_iter, tolerancia=tolerancia,
            timeout=timeout, target_costo=target_costo,
            usar_m1=True, usar_m2=True, usar_m3=True, usar_m4=True,
        )
        return {"ruta": ruta, "costo": costo, "df_iter": df_iter,
                "razon_parada": razon, "tiempo_seg": t}

    elif metodo == "VND":
        ruta, costo, df_iter, razon, t, _ = vnd_split(
            ruta_inicial, max_iter_por_vecindario=max_iter,
            tolerancia=tolerancia, timeout=timeout, target_costo=target_costo,
        )
        return {"ruta": ruta, "costo": costo, "df_iter": df_iter,
                "razon_parada": razon, "tiempo_seg": t}

    else:
        raise ValueError(f"Método desconocido: {metodo}")


def experimento_paper_busqueda_local(
    carpeta_entrada: str,
    carpeta_salida: str = "resultados/paper_busqueda_local",
    heuristicas_a_probar: list = None,
    metodos: list = None,
    max_iter: int = 100,
    tolerancia: float = 1e-9,
    timeout: float = None,
    seed_randomkeys: int = 11,
) -> tuple:
    """
    Experimento 1 del paper: Búsqueda Local Comparativa.

    Compara 4 métodos partiendo de la MISMA solución inicial:
        - 2OPT_BEST  : 2-opt Best Improvement
        - 2OPT_FIRST : 2-opt First Improvement
        - SPLIT_M01  : Split con M0+M1
        - SPLIT_FULL : Split con todos los movimientos

    Para cada heurística:
        construye UNA solución inicial, y todos los métodos parten de ella.

    Genera: paper_busqueda_local.xlsx + graficos/
    Devuelve: df_resultados, df_resumen_metodo, df_por_tamano, df_convergencia
    """
    os.makedirs(carpeta_salida, exist_ok=True)
    archivos = listar_tsp_en_carpeta(carpeta_entrada)
    heuristicas_a_probar = heuristicas_a_probar or ["NNH", "INSERCION", "RANDOM_KEYS"]
    metodos = metodos or ["2OPT_BEST", "2OPT_FIRST", "SPLIT_M01", "SPLIT_FULL"]

    print(f"\n{'='*55}")
    print("PAPER EXP 1: BÚSQUEDA LOCAL COMPARATIVA")
    print(f"Instancias: {len(archivos)} | Heurísticas: {len(heuristicas_a_probar)} | Métodos: {len(metodos)}")
    if timeout:
        print(f"Timeout por corrida: {timeout}s")
    print(f"{'='*55}\n")

    filas_resultados = []
    filas_convergencia = []

    for ruta in archivos:
        nombre = os.path.splitext(os.path.basename(ruta))[0]
        valida, tipo = _verificar_instancia(ruta)
        if not valida:
            print(f"  OMITIDA {nombre} — tipo {tipo}")
            continue

        try:
            nodos = leer_archivo_tsp(ruta)
            lb_mst = calcular_mst(nodos)

            # Para cada heurística construimos UNA ruta inicial
            for heuristica in heuristicas_a_probar:
                ruta_ini, _ = _construir_ruta_inicial(nodos, heuristica, seed_randomkeys)
                costo_inicial = distancia_euclidea_total(ruta_ini)

                print(f"\n  {nombre:<18} ({len(nodos)} nodos) — heurística: {heuristica}")
                print(f"    Costo inicial: {costo_inicial:.2f}")

                # Cada método parte de la MISMA ruta_ini (con copia)
                for metodo in metodos:
                    res = _ejecutar_metodo_busqueda_local(
                        metodo=metodo,
                        ruta_inicial=ruta_ini.copy(),
                        max_iter=max_iter,
                        tolerancia=tolerancia,
                        timeout=timeout,
                    )

                    mejora_pct = (costo_inicial - res["costo"]) / costo_inicial * 100 if costo_inicial > 0 else 0
                    gap_pct = (res["costo"] - lb_mst) / lb_mst * 100 if lb_mst > 0 else 0

                    filas_resultados.append({
                        "instancia": nombre,
                        "n_nodos": len(nodos),
                        "rango_nodos": _rango_nodos(len(nodos)),
                        "heuristica": heuristica,
                        "metodo": metodo,
                        "costo_inicial": round(costo_inicial, 4),
                        "costo_final": round(res["costo"], 4),
                        "mejora_abs": round(costo_inicial - res["costo"], 4),
                        "mejora_pct": round(mejora_pct, 4),
                        "cota_mst": round(lb_mst, 4),
                        "gap_pct": round(gap_pct, 4),
                        "tiempo_seg": round(res["tiempo_seg"], 4),
                        "razon_parada": res["razon_parada"],
                        "iteraciones": len(res["df_iter"]) if res["df_iter"] is not None else 0,
                    })

                    # Convergencia
                    if res["df_iter"] is not None and not res["df_iter"].empty:
                        for _, row in res["df_iter"].iterrows():
                            costo_iter = row.get("costo_salida", row.get("costo_entrada", 0))
                            iter_num = row.get("iteracion", 0)
                            filas_convergencia.append({
                                "instancia": nombre,
                                "n_nodos": len(nodos),
                                "rango_nodos": _rango_nodos(len(nodos)),
                                "heuristica": heuristica,
                                "metodo": metodo,
                                "iteracion": iter_num,
                                "costo": round(costo_iter, 4),
                            })

                    print(f"    {metodo:<12} | costo={res['costo']:>10.2f} | "
                          f"gap={gap_pct:>6.2f}% | t={res['tiempo_seg']:>6.2f}s | {res['razon_parada']}")

        except Exception as e:
            print(f"  ERROR {nombre}: {e}")

    df_resultados = pd.DataFrame(filas_resultados)
    df_convergencia = pd.DataFrame(filas_convergencia)

    df_resumen_metodo = (
        df_resultados.groupby(["heuristica", "metodo"], as_index=False)
        [["mejora_pct", "gap_pct", "tiempo_seg", "iteraciones"]].mean().round(4)
        .sort_values(["heuristica", "gap_pct"])
    ) if not df_resultados.empty else pd.DataFrame()

    df_por_tamano = (
        df_resultados.groupby(["rango_nodos", "metodo"], as_index=False)
        [["gap_pct", "mejora_pct", "tiempo_seg"]].mean().round(4)
        .sort_values(["rango_nodos", "metodo"])
    ) if not df_resultados.empty else pd.DataFrame()

    _guardar_excel(os.path.join(carpeta_salida, "paper_busqueda_local.xlsx"), {
        "resultados": df_resultados,
        "resumen_metodo": df_resumen_metodo,
        "por_tamano": df_por_tamano,
        "convergencia": df_convergencia,
    })
    print(f"\n✓ Excel guardado en: {carpeta_salida}/paper_busqueda_local.xlsx")

    if not df_resultados.empty:
        dashboard_paper_busqueda_local(
            df_resultados=df_resultados,
            df_resumen_metodo=df_resumen_metodo,
            df_por_tamano=df_por_tamano,
            df_convergencia=df_convergencia,
            carpeta_salida=os.path.join(carpeta_salida, "graficos"),
        )

    return df_resultados, df_resumen_metodo, df_por_tamano, df_convergencia


# =============================================================
# EXPERIMENTO PAPER 2: ITERATED LOCAL SEARCH (ILS)
# =============================================================

def _ejecutar_ils(
    metodo: str,
    ruta_inicial: list,
    perturbacion_nombre: str,
    max_perturbaciones: int,
    max_iter_bl: int,
    tolerancia: float,
    timeout_total: float = None,
    seed: int = 42,
    n_movimientos_perturb: int = 3,
) -> dict:
    """
    Ejecuta ILS aplicando perturbaciones repetidas + búsqueda local.

    Algoritmo:
        1. BL desde ruta_inicial → mejor_ruta
        2. Repetir hasta max_perturbaciones o timeout:
            a. ruta_perturbada = perturbar(mejor_ruta)
            b. ruta_bl = BL(ruta_perturbada)
            c. Si costo(ruta_bl) < costo(mejor_ruta): mejor_ruta = ruta_bl
        3. Retornar mejor_ruta

    Devuelve dict con: ruta, costo, n_perturbaciones, n_mejoras, df_perturbaciones,
                       razon_parada, tiempo_total_seg
    """
    tiempo_inicio = time.perf_counter()

    # Paso 1: Búsqueda local inicial
    res_inicial = _ejecutar_metodo_busqueda_local(
        metodo=metodo,
        ruta_inicial=ruta_inicial.copy(),
        max_iter=max_iter_bl,
        tolerancia=tolerancia,
        timeout=timeout_total,
    )

    mejor_ruta = res_inicial["ruta"]
    mejor_costo = res_inicial["costo"]
    costo_post_bl_inicial = mejor_costo

    historial_perturb = []
    n_mejoras = 0
    n_perturbaciones_ejecutadas = 0
    razon_parada = "max_perturbaciones"

    for k in range(max_perturbaciones):
        # Verificar timeout total
        tiempo_transcurrido = time.perf_counter() - tiempo_inicio
        if timeout_total is not None and tiempo_transcurrido >= timeout_total:
            razon_parada = "timeout"
            break

        timeout_restante = None
        if timeout_total is not None:
            timeout_restante = max(0.001, timeout_total - tiempo_transcurrido)

        # Aplicar perturbación
        seed_perturb = seed + k
        if perturbacion_nombre == "DOUBLE_BRIDGE":
            ruta_perturbada = perturbacion_double_bridge(mejor_ruta, seed=seed_perturb)
        elif perturbacion_nombre == "MULTI_MOVIMIENTO":
            ruta_perturbada = perturbacion_multi_movimiento(
                mejor_ruta, n_segmentos=n_movimientos_perturb, seed=seed_perturb,
            )
        else:
            raise ValueError(f"Perturbación desconocida: {perturbacion_nombre}")

        costo_perturbado = distancia_euclidea_total(ruta_perturbada)

        # Búsqueda local sobre perturbada
        res_bl = _ejecutar_metodo_busqueda_local(
            metodo=metodo,
            ruta_inicial=ruta_perturbada,
            max_iter=max_iter_bl,
            tolerancia=tolerancia,
            timeout=timeout_restante,
        )

        n_perturbaciones_ejecutadas += 1
        mejoro = res_bl["costo"] < mejor_costo - tolerancia

        historial_perturb.append({
            "iteracion_perturb": k + 1,
            "costo_antes_perturb": mejor_costo,
            "costo_post_perturb": costo_perturbado,
            "costo_post_bl": res_bl["costo"],
            "mejoro_global": mejoro,
            "razon_bl": res_bl["razon_parada"],
            "tiempo_bl_seg": res_bl["tiempo_seg"],
        })

        if mejoro:
            mejor_ruta = res_bl["ruta"]
            mejor_costo = res_bl["costo"]
            n_mejoras += 1

    tiempo_total = time.perf_counter() - tiempo_inicio

    return {
        "ruta": mejor_ruta,
        "costo": mejor_costo,
        "costo_post_bl_inicial": costo_post_bl_inicial,
        "n_perturbaciones": n_perturbaciones_ejecutadas,
        "n_mejoras": n_mejoras,
        "df_perturbaciones": pd.DataFrame(historial_perturb),
        "razon_parada": razon_parada,
        "tiempo_total_seg": tiempo_total,
    }


def experimento_paper_ils(
    carpeta_entrada: str,
    carpeta_salida: str = "resultados/paper_ils",
    heuristicas_a_probar: list = None,
    metodos: list = None,
    perturbaciones: list = None,
    max_iter_bl: int = 100,
    max_perturbaciones: int = 20,
    n_movimientos_perturb: int = 3,
    tolerancia: float = 1e-9,
    timeout_total: float = None,
    seed_randomkeys: int = 11,
    seed_perturbacion: int = 42,
) -> tuple:
    """
    Experimento 2 del paper: Iterated Local Search comparativo.

    Compara 5 métodos como búsqueda local dentro de un esquema ILS:
        - 2OPT_BEST, 2OPT_FIRST, SPLIT_M01, SPLIT_FULL, VND

    Probados con 2 perturbaciones:
        - DOUBLE_BRIDGE
        - MULTI_MOVIMIENTO

    Para cada heurística construye UNA solución inicial y todos los
    métodos × perturbaciones parten de ella.

    Genera: paper_ils.xlsx + graficos/
    Devuelve: df_resultados, df_resumen, df_por_tamano, df_perturbaciones
    """
    os.makedirs(carpeta_salida, exist_ok=True)
    archivos = listar_tsp_en_carpeta(carpeta_entrada)
    heuristicas_a_probar = heuristicas_a_probar or ["NNH", "INSERCION", "RANDOM_KEYS"]
    metodos = metodos or ["2OPT_BEST", "2OPT_FIRST", "SPLIT_M01", "SPLIT_FULL", "VND"]
    perturbaciones = perturbaciones or ["DOUBLE_BRIDGE", "MULTI_MOVIMIENTO"]

    print(f"\n{'='*55}")
    print("PAPER EXP 2: ITERATED LOCAL SEARCH (ILS)")
    print(f"Instancias: {len(archivos)} | Heurísticas: {len(heuristicas_a_probar)}")
    print(f"Métodos: {len(metodos)} | Perturbaciones: {len(perturbaciones)}")
    print(f"Max perturbaciones: {max_perturbaciones} | Max iter BL: {max_iter_bl}")
    if timeout_total:
        print(f"Timeout total por corrida: {timeout_total}s")
    print(f"{'='*55}\n")

    filas_resultados = []
    filas_perturbaciones = []

    for ruta in archivos:
        nombre = os.path.splitext(os.path.basename(ruta))[0]
        valida, tipo = _verificar_instancia(ruta)
        if not valida:
            print(f"  OMITIDA {nombre} — tipo {tipo}")
            continue

        try:
            nodos = leer_archivo_tsp(ruta)
            lb_mst = calcular_mst(nodos)

            for heuristica in heuristicas_a_probar:
                ruta_ini, _ = _construir_ruta_inicial(nodos, heuristica, seed_randomkeys)
                costo_inicial = distancia_euclidea_total(ruta_ini)

                print(f"\n  {nombre:<18} ({len(nodos)} nodos) — heurística: {heuristica}")
                print(f"    Costo inicial: {costo_inicial:.2f}")

                for metodo in metodos:
                    for perturbacion in perturbaciones:
                        res = _ejecutar_ils(
                            metodo=metodo,
                            ruta_inicial=ruta_ini.copy(),
                            perturbacion_nombre=perturbacion,
                            max_perturbaciones=max_perturbaciones,
                            max_iter_bl=max_iter_bl,
                            tolerancia=tolerancia,
                            timeout_total=timeout_total,
                            seed=seed_perturbacion,
                            n_movimientos_perturb=n_movimientos_perturb,
                        )

                        mejora_pct = (costo_inicial - res["costo"]) / costo_inicial * 100 if costo_inicial > 0 else 0
                        gap_pct = (res["costo"] - lb_mst) / lb_mst * 100 if lb_mst > 0 else 0
                        # Mejora ganada por la perturbación vs solo BL inicial
                        mejora_perturb_pct = (
                            (res["costo_post_bl_inicial"] - res["costo"]) /
                            res["costo_post_bl_inicial"] * 100
                        ) if res["costo_post_bl_inicial"] > 0 else 0

                        filas_resultados.append({
                            "instancia": nombre,
                            "n_nodos": len(nodos),
                            "rango_nodos": _rango_nodos(len(nodos)),
                            "heuristica": heuristica,
                            "metodo": metodo,
                            "perturbacion": perturbacion,
                            "costo_inicial": round(costo_inicial, 4),
                            "costo_post_bl_inicial": round(res["costo_post_bl_inicial"], 4),
                            "costo_final": round(res["costo"], 4),
                            "mejora_pct": round(mejora_pct, 4),
                            "mejora_perturb_pct": round(mejora_perturb_pct, 4),
                            "cota_mst": round(lb_mst, 4),
                            "gap_pct": round(gap_pct, 4),
                            "n_perturbaciones": res["n_perturbaciones"],
                            "n_mejoras": res["n_mejoras"],
                            "tasa_exito_pct": round(
                                res["n_mejoras"] / res["n_perturbaciones"] * 100, 2
                            ) if res["n_perturbaciones"] > 0 else 0,
                            "tiempo_total_seg": round(res["tiempo_total_seg"], 4),
                            "razon_parada": res["razon_parada"],
                        })

                        # Detalle de cada perturbación
                        if not res["df_perturbaciones"].empty:
                            df_p = res["df_perturbaciones"].copy()
                            df_p["instancia"] = nombre
                            df_p["heuristica"] = heuristica
                            df_p["metodo"] = metodo
                            df_p["perturbacion"] = perturbacion
                            filas_perturbaciones.append(df_p)

                        print(f"    {metodo:<12} + {perturbacion:<18} | "
                              f"costo={res['costo']:>10.2f} | gap={gap_pct:>6.2f}% | "
                              f"mejoras={res['n_mejoras']}/{res['n_perturbaciones']} | "
                              f"t={res['tiempo_total_seg']:>6.2f}s")

        except Exception as e:
            print(f"  ERROR {nombre}: {e}")

    df_resultados = pd.DataFrame(filas_resultados)
    df_perturbaciones = (
        pd.concat(filas_perturbaciones, ignore_index=True)
        if filas_perturbaciones else pd.DataFrame()
    )

    df_resumen = (
        df_resultados.groupby(["metodo", "perturbacion"], as_index=False)
        [["gap_pct", "mejora_pct", "mejora_perturb_pct", "tasa_exito_pct",
          "n_mejoras", "tiempo_total_seg"]].mean().round(4)
        .sort_values(["metodo", "perturbacion"])
    ) if not df_resultados.empty else pd.DataFrame()

    df_por_tamano = (
        df_resultados.groupby(["rango_nodos", "metodo", "perturbacion"], as_index=False)
        [["gap_pct", "mejora_pct", "tiempo_total_seg"]].mean().round(4)
        .sort_values(["rango_nodos", "metodo", "perturbacion"])
    ) if not df_resultados.empty else pd.DataFrame()

    _guardar_excel(os.path.join(carpeta_salida, "paper_ils.xlsx"), {
        "resultados": df_resultados,
        "resumen": df_resumen,
        "por_tamano": df_por_tamano,
        "perturbaciones_detalle": df_perturbaciones,
    })
    print(f"\n✓ Excel guardado en: {carpeta_salida}/paper_ils.xlsx")

    if not df_resultados.empty:
        dashboard_paper_ils(
            df_resultados=df_resultados,
            df_resumen=df_resumen,
            df_por_tamano=df_por_tamano,
            df_perturbaciones=df_perturbaciones,
            carpeta_salida=os.path.join(carpeta_salida, "graficos"),
        )

    return df_resultados, df_resumen, df_por_tamano, df_perturbaciones


# =============================================================
# EXPERIMENTO PAPER 3: BRKGA vs BRKGA + SPLIT
# =============================================================

def experimento_paper_brkga(
    carpeta_entrada: str,
    carpeta_salida: str = "resultados/paper_brkga",
    decoders: list = None,
    n_semillas: int = 10,
    semilla_inicio: int = 1,
    n_poblacion: int = 100,
    n_generaciones: int = 100,
    pct_elite: float = 0.25,
    pct_mutantes: float = 0.20,
    prob_elite_crossover: float = 0.70,
    timeout: float = None,
    usar_m1: bool = True,
    usar_m2: bool = True,
    usar_m3: bool = True,
    usar_m4: bool = True,
) -> tuple:
    """
    Experimento 3 del paper: BRKGA puro vs BRKGA + Split.

    Para cada instancia, ejecuta cada decoder con n_semillas semillas distintas
    para evaluar la variabilidad estadística del algoritmo.

    Decoders:
        - DECODER_SORT  : BRKGA puro (decode = ordenar por keys)
        - DECODER_SPLIT : BRKGA híbrido (decode = ordenar + Split DP)

    Parámetros
    ----------
    n_semillas : int
        Número de corridas independientes con semillas distintas
        (típicamente 10 en papers)
    semilla_inicio : int
        Primera semilla; las demás son consecutivas

    Genera: paper_brkga.xlsx + graficos/
    Devuelve: df_resultados, df_resumen, df_por_tamano, df_evolucion
    """
    os.makedirs(carpeta_salida, exist_ok=True)
    archivos = listar_tsp_en_carpeta(carpeta_entrada)
    decoders = decoders or ["DECODER_SORT", "DECODER_SPLIT"]
    semillas = list(range(semilla_inicio, semilla_inicio + n_semillas))

    print(f"\n{'='*55}")
    print("PAPER EXP 3: BRKGA vs BRKGA + SPLIT")
    print(f"Instancias: {len(archivos)} | Decoders: {len(decoders)} | Semillas: {len(semillas)}")
    print(f"Población: {n_poblacion} | Generaciones: {n_generaciones}")
    print(f"Élite: {pct_elite:.0%} | Mutantes: {pct_mutantes:.0%} | ρ_elite: {prob_elite_crossover}")
    if timeout:
        print(f"Timeout por corrida: {timeout}s")
    print(f"{'='*55}\n")

    filas_resultados = []
    filas_evolucion = []

    for ruta in archivos:
        nombre = os.path.splitext(os.path.basename(ruta))[0]
        valida, tipo = _verificar_instancia(ruta)
        if not valida:
            print(f"  OMITIDA {nombre} — tipo {tipo}")
            continue

        try:
            nodos = leer_archivo_tsp(ruta)
            lb_mst = calcular_mst(nodos)

            print(f"\n  {nombre:<18} ({len(nodos)} nodos)  MST={lb_mst:.2f}")

            for decoder_nombre in decoders:
                for semilla in semillas:
                    mejor_ruta, mejor_costo, df_evol, razon, t = brkga(
                        nodos=nodos,
                        decoder_nombre=decoder_nombre,
                        n_poblacion=n_poblacion,
                        n_generaciones=n_generaciones,
                        pct_elite=pct_elite,
                        pct_mutantes=pct_mutantes,
                        prob_elite_crossover=prob_elite_crossover,
                        seed=semilla,
                        timeout=timeout,
                        usar_m1=usar_m1, usar_m2=usar_m2,
                        usar_m3=usar_m3, usar_m4=usar_m4,
                    )

                    gap_pct = (mejor_costo - lb_mst) / lb_mst * 100 if lb_mst > 0 else 0

                    filas_resultados.append({
                        "instancia": nombre,
                        "n_nodos": len(nodos),
                        "rango_nodos": _rango_nodos(len(nodos)),
                        "decoder": decoder_nombre,
                        "semilla": semilla,
                        "n_poblacion": n_poblacion,
                        "n_generaciones_max": n_generaciones,
                        "n_generaciones_corridas": len(df_evol) - 1,
                        "costo_final": round(mejor_costo, 4),
                        "cota_mst": round(lb_mst, 4),
                        "gap_pct": round(gap_pct, 4),
                        "tiempo_seg": round(t, 4),
                        "razon_parada": razon,
                    })

                    # Guardar evolución (puede ser pesado, guardamos cada generación)
                    df_e = df_evol.copy()
                    df_e["instancia"] = nombre
                    df_e["n_nodos"] = len(nodos)
                    df_e["decoder"] = decoder_nombre
                    df_e["semilla"] = semilla
                    filas_evolucion.append(df_e)

                    print(f"    {decoder_nombre:<14} seed={semilla:<3} | "
                          f"costo={mejor_costo:>10.2f} | gap={gap_pct:>6.2f}% | "
                          f"t={t:>6.2f}s | {razon}")

        except Exception as e:
            print(f"  ERROR {nombre}: {e}")

    df_resultados = pd.DataFrame(filas_resultados)
    df_evolucion = (
        pd.concat(filas_evolucion, ignore_index=True)
        if filas_evolucion else pd.DataFrame()
    )

    # Estadísticas por decoder e instancia (media + desv estándar sobre semillas)
    df_resumen_instancia = (
        df_resultados.groupby(["instancia", "n_nodos", "rango_nodos", "decoder"], as_index=False)
        .agg(
            gap_mean=("gap_pct", "mean"),
            gap_std=("gap_pct", "std"),
            gap_min=("gap_pct", "min"),
            gap_max=("gap_pct", "max"),
            costo_mean=("costo_final", "mean"),
            costo_std=("costo_final", "std"),
            costo_min=("costo_final", "min"),
            tiempo_mean=("tiempo_seg", "mean"),
        ).round(4)
    ) if not df_resultados.empty else pd.DataFrame()

    # Resumen global por decoder
    df_resumen = (
        df_resultados.groupby("decoder", as_index=False)
        .agg(
            gap_mean=("gap_pct", "mean"),
            gap_std=("gap_pct", "std"),
            gap_min=("gap_pct", "min"),
            gap_max=("gap_pct", "max"),
            tiempo_mean=("tiempo_seg", "mean"),
        ).round(4)
    ) if not df_resultados.empty else pd.DataFrame()

    # Resumen por tamaño
    df_por_tamano = (
        df_resultados.groupby(["rango_nodos", "decoder"], as_index=False)
        .agg(
            gap_mean=("gap_pct", "mean"),
            gap_std=("gap_pct", "std"),
            tiempo_mean=("tiempo_seg", "mean"),
        ).round(4)
        .sort_values(["rango_nodos", "decoder"])
    ) if not df_resultados.empty else pd.DataFrame()

    _guardar_excel(os.path.join(carpeta_salida, "paper_brkga.xlsx"), {
        "resultados": df_resultados,
        "resumen": df_resumen,
        "resumen_instancia": df_resumen_instancia,
        "por_tamano": df_por_tamano,
    })
    # Evolución por separado (puede ser muy grande)
    if not df_evolucion.empty:
        df_evol_resumen = (
            df_evolucion.groupby(["decoder", "generacion"], as_index=False)
            [["mejor_costo", "promedio_costo"]].mean().round(4)
        )
        _guardar_excel(os.path.join(carpeta_salida, "paper_brkga_evolucion.xlsx"), {
            "evolucion_promedio": df_evol_resumen,
        })

    print(f"\n✓ Excel guardado en: {carpeta_salida}/paper_brkga.xlsx")

    if not df_resultados.empty:
        dashboard_paper_brkga(
            df_resultados=df_resultados,
            df_resumen_instancia=df_resumen_instancia,
            df_por_tamano=df_por_tamano,
            df_evolucion=df_evolucion,
            carpeta_salida=os.path.join(carpeta_salida, "graficos"),
        )

    return df_resultados, df_resumen, df_por_tamano, df_evolucion