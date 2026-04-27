# tsp/brkga.py
"""
Biased Random-Key Genetic Algorithm (BRKGA) para TSP.

Implementación basada en:
    Gonçalves, J.F. & Resende, M.G.C. (2011).
    "Biased random-key genetic algorithms for combinatorial optimization."
    Journal of Heuristics, 17, 487-525.

Decoders disponibles:
    - DECODER_SORT  : ordena nodos por random keys (BRKGA puro)
    - DECODER_SPLIT : ordena por keys + aplica Split como local search
"""
import time
import random
import pandas as pd
from tsp.distancias import distancia_euclidea_total
from tsp.split import split_tsp_dp, reconstruir_camino


# =============================================================
# DECODERS
# =============================================================

def decoder_sort(keys: list, nodos: list) -> tuple:
    """
    Decoder simple: ordena los nodos por su random key ascendente.

    Parámetros
    ----------
    keys : list
        Vector de n random keys en [0, 1]
    nodos : list
        Lista de nodos (id, x, y)

    Devuelve
    --------
    ruta : list de nodos en el orden decodificado
    costo : float
    """
    n = len(nodos)
    indices_ordenados = sorted(range(n), key=lambda i: keys[i])
    ruta = [nodos[i] for i in indices_ordenados]
    costo = distancia_euclidea_total(ruta)
    return ruta, costo


def decoder_split(
    keys: list,
    nodos: list,
    usar_m1: bool = True,
    usar_m2: bool = True,
    usar_m3: bool = True,
    usar_m4: bool = True,
) -> tuple:
    """
    Decoder híbrido: ordena por keys y aplica Split DP como local search.

    Esto es lo que en literatura se llama "BRKGA híbrido" o "BRKGA con local search".
    El decoder ya devuelve una ruta optimizada localmente.

    Devuelve
    --------
    ruta : list de nodos optimizados
    costo : float
    """
    n = len(nodos)
    indices_ordenados = sorted(range(n), key=lambda i: keys[i])
    ruta_inicial = [nodos[i] for i in indices_ordenados]

    # Aplicar Split DP
    V, P, M, _, _ = split_tsp_dp(
        ruta_inicial,
        usar_m1=usar_m1, usar_m2=usar_m2,
        usar_m3=usar_m3, usar_m4=usar_m4,
    )
    stats_dummy = {k: {"nombre": f"M{k}", "evaluado": 0, "mejora_vs_M0": 0,
                       "ganador_local": 0, "usado_final": 0} for k in range(5)}
    ruta_optimizada, _ = reconstruir_camino(ruta_inicial, P, M, stats_dummy)
    costo = distancia_euclidea_total(ruta_optimizada)

    return ruta_optimizada, costo


# =============================================================
# OPERADORES BRKGA
# =============================================================

def _generar_individuo(n: int, rng: random.Random) -> list:
    """Genera un vector aleatorio de n random keys en [0, 1]."""
    return [rng.random() for _ in range(n)]


def _crossover_sesgado(padre_elite: list, padre_no_elite: list,
                       prob_elite: float, rng: random.Random) -> list:
    """
    Cruce sesgado parametrizado.

    Para cada gen, con probabilidad prob_elite hereda del padre élite,
    sino del no-élite. Típicamente prob_elite = 0.7.
    """
    n = len(padre_elite)
    hijo = []
    for i in range(n):
        if rng.random() < prob_elite:
            hijo.append(padre_elite[i])
        else:
            hijo.append(padre_no_elite[i])
    return hijo


# =============================================================
# BRKGA PRINCIPAL
# =============================================================

def brkga(
    nodos: list,
    decoder_nombre: str = "DECODER_SORT",
    n_poblacion: int = 100,
    n_generaciones: int = 100,
    pct_elite: float = 0.25,
    pct_mutantes: float = 0.20,
    prob_elite_crossover: float = 0.70,
    seed: int = 42,
    timeout: float = None,
    target_costo: float = None,
    usar_m1: bool = True,
    usar_m2: bool = True,
    usar_m3: bool = True,
    usar_m4: bool = True,
) -> tuple:
    """
    Ejecuta BRKGA sobre una instancia TSP.

    Parámetros
    ----------
    nodos : list
        Lista de nodos (id, x, y)
    decoder_nombre : str
        "DECODER_SORT" para BRKGA puro
        "DECODER_SPLIT" para BRKGA híbrido con Split
    n_poblacion : int
        Tamaño de la población (típicamente 100-200)
    n_generaciones : int
        Número de generaciones (típicamente 100-500)
    pct_elite : float
        Porcentaje élite de la población (típicamente 0.20-0.30)
    pct_mutantes : float
        Porcentaje de mutantes nuevos por generación (típicamente 0.15-0.25)
    prob_elite_crossover : float
        Probabilidad de heredar del padre élite (típicamente 0.65-0.75)
    seed : int
        Semilla para reproducibilidad
    timeout : float, opcional
        Tiempo máximo en segundos
    target_costo : float, opcional
        Costo objetivo (TTT)
    usar_m1..usar_m4 : bool
        Movimientos del Split (solo aplica si decoder_nombre == "DECODER_SPLIT")

    Devuelve
    --------
    mejor_ruta, mejor_costo, df_evolucion, razon_parada, tiempo_total
    """
    rng = random.Random(seed)
    n = len(nodos)

    # Tamaños de cada grupo
    n_elite = max(1, int(n_poblacion * pct_elite))
    n_mutantes = max(1, int(n_poblacion * pct_mutantes))
    n_crossover = n_poblacion - n_elite - n_mutantes

    # Función de decodificación
    if decoder_nombre == "DECODER_SORT":
        def decode(keys):
            return decoder_sort(keys, nodos)
    elif decoder_nombre == "DECODER_SPLIT":
        def decode(keys):
            return decoder_split(keys, nodos,
                                 usar_m1=usar_m1, usar_m2=usar_m2,
                                 usar_m3=usar_m3, usar_m4=usar_m4)
    else:
        raise ValueError(f"Decoder desconocido: {decoder_nombre}")

    tiempo_inicio = time.perf_counter()
    razon_parada = "max_generaciones"

    # Población inicial: n_poblacion individuos aleatorios
    poblacion = []  # lista de (keys, ruta, costo)
    for _ in range(n_poblacion):
        keys = _generar_individuo(n, rng)
        ruta, costo = decode(keys)
        poblacion.append((keys, ruta, costo))

    # Ordenar por costo (ascendente)
    poblacion.sort(key=lambda ind: ind[2])

    mejor_keys, mejor_ruta, mejor_costo = poblacion[0]
    historial = [{
        "generacion": 0,
        "mejor_costo": mejor_costo,
        "promedio_costo": sum(ind[2] for ind in poblacion) / len(poblacion),
        "peor_costo": poblacion[-1][2],
        "tiempo_seg": time.perf_counter() - tiempo_inicio,
    }]

    # Iteración generacional
    for gen in range(1, n_generaciones + 1):
        # Verificar timeout
        if timeout is not None:
            if (time.perf_counter() - tiempo_inicio) >= timeout:
                razon_parada = "timeout"
                break

        # Verificar target
        if target_costo is not None and mejor_costo <= target_costo:
            razon_parada = "target_alcanzado"
            break

        # Construir nueva población
        # 1) Élite (top n_elite, pasa directo)
        elite = poblacion[:n_elite]

        # 2) Mutantes (n_mutantes individuos aleatorios nuevos)
        mutantes = []
        for _ in range(n_mutantes):
            keys = _generar_individuo(n, rng)
            ruta, costo = decode(keys)
            mutantes.append((keys, ruta, costo))

        # 3) Crossover (n_crossover hijos de elite × no-elite)
        no_elite = poblacion[n_elite:]
        hijos = []
        for _ in range(n_crossover):
            padre_e = elite[rng.randrange(len(elite))][0]
            padre_ne = no_elite[rng.randrange(len(no_elite))][0]
            hijo_keys = _crossover_sesgado(padre_e, padre_ne,
                                           prob_elite_crossover, rng)
            hijo_ruta, hijo_costo = decode(hijo_keys)
            hijos.append((hijo_keys, hijo_ruta, hijo_costo))

        # Nueva generación = elite + hijos + mutantes
        poblacion = elite + hijos + mutantes
        poblacion.sort(key=lambda ind: ind[2])

        # Actualizar mejor
        if poblacion[0][2] < mejor_costo:
            mejor_keys, mejor_ruta, mejor_costo = poblacion[0]

        historial.append({
            "generacion": gen,
            "mejor_costo": mejor_costo,
            "promedio_costo": sum(ind[2] for ind in poblacion) / len(poblacion),
            "peor_costo": poblacion[-1][2],
            "tiempo_seg": time.perf_counter() - tiempo_inicio,
        })

    tiempo_total = time.perf_counter() - tiempo_inicio
    df_evolucion = pd.DataFrame(historial)

    return mejor_ruta, mejor_costo, df_evolucion, razon_parada, tiempo_total