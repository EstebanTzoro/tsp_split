# tsp/split.py
import pandas as pd
from tsp.distancias import distancia_euclidea


def split_tsp_dp(
    secuencia: list,
    usar_m1: bool = True,
    usar_m2: bool = True,
    usar_m3: bool = True,
    usar_m4: bool = True,
) -> tuple:
    """
    Algoritmo Split adaptado al TSP con diagnóstico completo de movimientos.

    Movimientos disponibles sobre un bloque [i .. j]:
        M0: camino base sin modificación
        M1: invierte el interior del bloque
        M2: intercambia los extremos internos (requiere >= 5 nodos)
        M3: saca el primer nodo interno y lo mueve al final
        M4: saca el último nodo interno y lo mueve al inicio

    Parámetros
    ----------
    secuencia : list of (id, x, y)
        Ruta de entrada sobre la que aplicar el DP.
    usar_m1..usar_m4 : bool
        Activa o desactiva cada movimiento alternativo.

    Devuelve
    --------
    V : list[float]       Costo óptimo hasta cada posición
    P : list[int]         Predecesor óptimo de cada posición
    M : list[int]         Movimiento usado en cada posición
    stats_mov : dict      Estadísticas por movimiento
    df_detalle : DataFrame  Detalle de cada par (i, j) evaluado
    """
    n = len(secuencia)
    d = distancia_euclidea

    V = [float("inf")] * n
    P = [0] * n
    M = [0] * n
    V[0] = 0

    # Inicialización base: cadena directa
    for i in range(1, n):
        V[i] = V[i - 1] + d(secuencia[i - 1], secuencia[i])
        P[i] = 0
        M[i] = 0

    stats_mov = {
        k: {"nombre": f"M{k}", "evaluado": 0, "mejora_vs_M0": 0,
            "ganador_local": 0, "usado_final": 0}
        for k in range(5)
    }

    detalle_bloques = []

    for i in range(0, n - 2):
        C = 0.0

        for j in range(i + 1, n):
            costos_locales = {k: None for k in range(5)}
            mejor_mov_local = 0
            mejor_costo_local = float("inf")

            # ---- caso j == i+1 ----
            if j == i + 1:
                m0 = V[i] + d(secuencia[i], secuencia[j])
                costos_locales[0] = m0
                stats_mov[0]["evaluado"] += 1
                mejor_mov_local = 0
                mejor_costo_local = m0
                stats_mov[0]["ganador_local"] += 1
                if m0 < V[j]:
                    V[j], P[j], M[j] = m0, i, 0

            # ---- caso j == i+2 ----
            elif j == i + 2:
                m0 = (
                    V[i]
                    + d(secuencia[i], secuencia[i + 1])
                    + d(secuencia[i + 1], secuencia[j])
                )
                costos_locales[0] = m0
                stats_mov[0]["evaluado"] += 1
                mejor_mov_local = 0
                mejor_costo_local = m0
                stats_mov[0]["ganador_local"] += 1
                if m0 < V[j]:
                    V[j], P[j], M[j] = m0, i, 0

            # ---- caso general j > i+2 ----
            else:
                C += d(secuencia[j - 1], secuencia[j - 2])

                # M0
                m0 = (
                    V[i]
                    + d(secuencia[i], secuencia[i + 1])
                    + C
                    + d(secuencia[j - 1], secuencia[j])
                )
                costos_locales[0] = m0
                stats_mov[0]["evaluado"] += 1
                mejor_mov_local = 0
                mejor_costo_local = m0
                if m0 < V[j]:
                    V[j], P[j], M[j] = m0, i, 0

                # M1: invierte el interior [a, b..d, e] -> [a, d..b, e]
                if usar_m1:
                    m1 = (
                        V[i]
                        + C
                        + d(secuencia[i], secuencia[j - 1])
                        + d(secuencia[i + 1], secuencia[j])
                    )
                    costos_locales[1] = m1
                    stats_mov[1]["evaluado"] += 1
                    if m1 < m0:
                        stats_mov[1]["mejora_vs_M0"] += 1
                    if m1 < mejor_costo_local:
                        mejor_costo_local = m1
                        mejor_mov_local = 1
                    if m1 < V[j]:
                        V[j], P[j], M[j] = m1, i, 1

                # M2: intercambia extremos internos (mínimo 5 nodos en bloque)
                if usar_m2 and j >= i + 4:
                    m2 = (
                        m0
                        - d(secuencia[i],     secuencia[i + 1])
                        - d(secuencia[i + 1], secuencia[i + 2])
                        - d(secuencia[j - 2], secuencia[j - 1])
                        - d(secuencia[j - 1], secuencia[j])
                        + d(secuencia[i],     secuencia[j - 1])
                        + d(secuencia[j - 1], secuencia[i + 2])
                        + d(secuencia[j - 2], secuencia[i + 1])
                        + d(secuencia[i + 1], secuencia[j])
                    )
                    costos_locales[2] = m2
                    stats_mov[2]["evaluado"] += 1
                    if m2 < m0:
                        stats_mov[2]["mejora_vs_M0"] += 1
                    if m2 < mejor_costo_local:
                        mejor_costo_local = m2
                        mejor_mov_local = 2
                    if m2 < V[j]:
                        V[j], P[j], M[j] = m2, i, 2

                # M3: saca primer nodo interno al final [a,b,c..y,z] -> [a,c..y,b,z]
                if usar_m3:
                    m3 = (
                        m0
                        - d(secuencia[i],     secuencia[i + 1])
                        - d(secuencia[i + 1], secuencia[i + 2])
                        - d(secuencia[j - 1], secuencia[j])
                        + d(secuencia[i],     secuencia[i + 2])
                        + d(secuencia[j - 1], secuencia[i + 1])
                        + d(secuencia[i + 1], secuencia[j])
                    )
                    costos_locales[3] = m3
                    stats_mov[3]["evaluado"] += 1
                    if m3 < m0:
                        stats_mov[3]["mejora_vs_M0"] += 1
                    if m3 < mejor_costo_local:
                        mejor_costo_local = m3
                        mejor_mov_local = 3
                    if m3 < V[j]:
                        V[j], P[j], M[j] = m3, i, 3

                # M4: saca último nodo interno al inicio [a,b..x,y,z] -> [a,y,b..x,z]
                if usar_m4:
                    m4 = (
                        m0
                        - d(secuencia[i],     secuencia[i + 1])
                        - d(secuencia[j - 2], secuencia[j - 1])
                        - d(secuencia[j - 1], secuencia[j])
                        + d(secuencia[i],     secuencia[j - 1])
                        + d(secuencia[j - 1], secuencia[i + 1])
                        + d(secuencia[j - 2], secuencia[j])
                    )
                    costos_locales[4] = m4
                    stats_mov[4]["evaluado"] += 1
                    if m4 < m0:
                        stats_mov[4]["mejora_vs_M0"] += 1
                    if m4 < mejor_costo_local:
                        mejor_costo_local = m4
                        mejor_mov_local = 4
                    if m4 < V[j]:
                        V[j], P[j], M[j] = m4, i, 4

                stats_mov[mejor_mov_local]["ganador_local"] += 1

            detalle_bloques.append({
                "i": i,
                "j": j,
                "tam_bloque": j - i + 1,
                "M0_costo": costos_locales[0],
                "M1_costo": costos_locales[1],
                "M2_costo": costos_locales[2],
                "M3_costo": costos_locales[3],
                "M4_costo": costos_locales[4],
                "ganador_local": mejor_mov_local,
                "ganador_local_nombre": f"M{mejor_mov_local}",
            })

    return V, P, M, stats_mov, pd.DataFrame(detalle_bloques)


def reconstruir_camino(
    secuencia: list,
    P: list,
    M: list,
    stats_mov: dict,
) -> tuple[list, dict]:
    """
    Reconstruye el camino óptimo a partir de los vectores P y M del DP.
    Aplica la transformación de bloque correspondiente a cada movimiento.

    Devuelve (camino reconstruido, stats_mov actualizado con usado_final).
    """
    bloques = []
    j = len(secuencia) - 1

    while j > 0:
        i = P[j]
        tipo = M[j]
        stats_mov[tipo]["usado_final"] += 1
        bloque = secuencia[i: j + 1]

        if tipo == 0:
            bloque_rec = bloque

        elif tipo == 1:
            interno = bloque[1:-1]
            bloque_rec = [bloque[0]] + interno[::-1] + [bloque[-1]]

        elif tipo == 2:
            a, b, z, y = bloque[0], bloque[1], bloque[-1], bloque[-2]
            medio = bloque[2:-2]
            bloque_rec = [a, y] + medio + [b, z]

        elif tipo == 3:
            a, b, z = bloque[0], bloque[1], bloque[-1]
            medio = bloque[2:-1]
            bloque_rec = [a] + medio + [b, z]

        elif tipo == 4:
            a, z, y = bloque[0], bloque[-1], bloque[-2]
            medio = bloque[1:-2]
            bloque_rec = [a, y] + medio + [z]

        else:
            raise ValueError(f"Movimiento desconocido: {tipo}")

        bloques.append(bloque_rec)
        j = i

    bloques.reverse()

    camino = []
    for bloque in bloques:
        if not camino:
            camino.extend(bloque)
        else:
            camino.extend(bloque[1:])

    return camino, stats_mov


def stats_mov_a_dataframe(stats_mov: dict) -> pd.DataFrame:
    """Convierte el dict de estadísticas de movimientos a DataFrame."""
    return pd.DataFrame([
        {
            "movimiento": stats_mov[k]["nombre"],
            "evaluado": stats_mov[k]["evaluado"],
            "mejora_vs_M0": stats_mov[k]["mejora_vs_M0"],
            "ganador_local": stats_mov[k]["ganador_local"],
            "usado_final": stats_mov[k]["usado_final"],
        }
        for k in sorted(stats_mov.keys())
    ])
