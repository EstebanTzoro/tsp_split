# tsp/perturbaciones.py
"""
Perturbaciones para Iterated Local Search (ILS).

Incluye:
- Double-Bridge: estándar de la literatura para ILS sobre TSP
  (Martin, Otto, Felten 1991; Lourenço, Martin, Stützle 2003)
- Multi-Movimiento Disjunto: perturbación basada en los movimientos M1-M4
  del Split, garantizando irreversibilidad bajo 2-opt y control de intensidad
"""
import random


# =============================================================
# DOUBLE-BRIDGE
# =============================================================
def perturbacion_double_bridge(ruta: list, seed: int = None) -> list:
    """
    Perturbación Double-Bridge clásica.

    Divide la ruta en 4 segmentos usando 3 puntos de corte aleatorios
    y los reordena como S1 + S3 + S2 + S4.

    Propiedades:
    - Irreversible para 2-opt (matemáticamente probado)
    - Cambia exactamente 4 aristas
    - No depende del tamaño de instancia

    Parámetros
    ----------
    ruta : list
        Secuencia de nodos a perturbar
    seed : int, opcional
        Semilla para reproducibilidad

    Devuelve
    --------
    list : ruta perturbada
    """
    if seed is not None:
        random.seed(seed)

    n = len(ruta)

    # Necesitamos al menos 8 nodos para que double-bridge tenga sentido
    if n < 8:
        return ruta[:]

    # Tres puntos de corte que dividen la ruta en 4 segmentos no triviales
    # Cada segmento debe tener al menos 1 nodo
    # pos1 ∈ [1, n/4], pos2 ∈ [n/4+1, n/2], pos3 ∈ [n/2+1, 3n/4]
    pos1 = 1 + random.randint(0, n // 4)
    pos2 = pos1 + 1 + random.randint(0, n // 4)
    pos3 = pos2 + 1 + random.randint(0, n // 4)

    # Asegurar que pos3 < n
    pos3 = min(pos3, n - 1)
    pos2 = min(pos2, pos3 - 1)
    pos1 = min(pos1, pos2 - 1)

    s1 = ruta[:pos1]
    s2 = ruta[pos1:pos2]
    s3 = ruta[pos2:pos3]
    s4 = ruta[pos3:]

    # Reordenar: S1 + S3 + S2 + S4
    return s1 + s3 + s2 + s4


# =============================================================
# MULTI-MOVIMIENTO DISJUNTO
# =============================================================
def _aplicar_m1_segmento(segmento: list) -> list:
    """M1: invertir el interior del segmento (mantiene extremos)."""
    if len(segmento) < 3:
        return segmento[:]
    return [segmento[0]] + segmento[1:-1][::-1] + [segmento[-1]]


def _aplicar_m2_segmento(segmento: list) -> list:
    """M2: intercambiar extremos internos (requiere >= 5 nodos)."""
    if len(segmento) < 5:
        return segmento[:]
    a = segmento[0]
    b = segmento[1]
    z = segmento[-1]
    y = segmento[-2]
    medio = segmento[2:-2]
    return [a, y] + medio + [b, z]


def _aplicar_m3_segmento(segmento: list) -> list:
    """M3: mover primer nodo interno al final."""
    if len(segmento) < 4:
        return segmento[:]
    a = segmento[0]
    b = segmento[1]
    z = segmento[-1]
    medio = segmento[2:-1]
    return [a] + medio + [b, z]


def _aplicar_m4_segmento(segmento: list) -> list:
    """M4: mover último nodo interno al inicio."""
    if len(segmento) < 4:
        return segmento[:]
    a = segmento[0]
    z = segmento[-1]
    y = segmento[-2]
    medio = segmento[1:-2]
    return [a, y] + medio + [z]


def perturbacion_multi_movimiento(
    ruta: list,
    n_segmentos: int = 3,
    tam_min: int = 5,
    tam_max: int = None,
    movimientos_habilitados: list = None,
    seed: int = None,
) -> list:
    """
    Perturbación Multi-Movimiento Disjunto.

    Selecciona K segmentos disjuntos (no se solapan) y aplica un movimiento
    aleatorio M1-M4 a cada uno. Diseñada para aprovechar los movimientos
    del operador Split como perturbación.

    Propiedades:
    - Irreversible para 2-opt (M2, M3, M4 no son reducibles a 2-opt)
    - Intensidad controlada (K segmentos exactos)
    - Aprovecha los movimientos del Split

    Parámetros
    ----------
    ruta : list
        Secuencia de nodos a perturbar
    n_segmentos : int
        Número de segmentos disjuntos a perturbar (típicamente 3-4)
    tam_min : int
        Tamaño mínimo de cada segmento (debe ser >= 5 para que M2 sea aplicable)
    tam_max : int, opcional
        Tamaño máximo de cada segmento. Si None, se calcula como n / (n_segmentos * 2)
    movimientos_habilitados : list, opcional
        Lista de movimientos a usar. Por defecto: ["M1", "M2", "M3", "M4"]
    seed : int, opcional
        Semilla para reproducibilidad

    Devuelve
    --------
    list : ruta perturbada
    """
    if seed is not None:
        random.seed(seed)

    if movimientos_habilitados is None:
        movimientos_habilitados = ["M1", "M2", "M3", "M4"]

    n = len(ruta)

    # Validar que la ruta sea suficientemente grande
    espacio_minimo_requerido = n_segmentos * tam_min
    if n < espacio_minimo_requerido + n_segmentos:
        # Reducir n_segmentos si la instancia es muy pequeña
        n_segmentos = max(1, (n - n_segmentos) // tam_min)
        if n_segmentos == 0:
            return ruta[:]

    if tam_max is None:
        tam_max = max(tam_min, n // (n_segmentos * 2))

    # Generar n_segmentos disjuntos aleatorios
    # Estrategia: dividir la ruta en n_segmentos zonas iguales y dentro
    # de cada zona elegir un segmento aleatorio
    zona_size = n // n_segmentos
    segmentos_indices = []  # lista de (inicio, fin) inclusivos

    for z in range(n_segmentos):
        zona_ini = z * zona_size
        zona_fin = (z + 1) * zona_size - 1 if z < n_segmentos - 1 else n - 1

        # Tamaño del segmento dentro de los límites
        tam_max_zona = min(tam_max, zona_fin - zona_ini + 1)
        if tam_max_zona < tam_min:
            continue  # zona muy pequeña, saltarla

        tam = random.randint(tam_min, tam_max_zona)

        # Inicio aleatorio dentro de la zona
        max_inicio = zona_fin - tam + 1
        if max_inicio < zona_ini:
            continue
        inicio = random.randint(zona_ini, max_inicio)
        fin = inicio + tam - 1

        segmentos_indices.append((inicio, fin))

    if not segmentos_indices:
        return ruta[:]

    # Aplicar movimientos a cada segmento (de derecha a izquierda
    # para no afectar índices de los segmentos siguientes)
    ruta_perturbada = ruta[:]

    funciones_mov = {
        "M1": _aplicar_m1_segmento,
        "M2": _aplicar_m2_segmento,
        "M3": _aplicar_m3_segmento,
        "M4": _aplicar_m4_segmento,
    }

    for inicio, fin in sorted(segmentos_indices, reverse=True):
        segmento = ruta_perturbada[inicio:fin + 1]
        mov = random.choice(movimientos_habilitados)
        segmento_modificado = funciones_mov[mov](segmento)
        ruta_perturbada[inicio:fin + 1] = segmento_modificado

    return ruta_perturbada


# =============================================================
# REGISTRO DE PERTURBACIONES DISPONIBLES
# =============================================================
PERTURBACIONES = {
    "DOUBLE_BRIDGE": perturbacion_double_bridge,
    "MULTI_MOVIMIENTO": perturbacion_multi_movimiento,
}


def obtener_perturbacion(nombre: str):
    """Devuelve la función de perturbación por nombre."""
    if nombre not in PERTURBACIONES:
        raise ValueError(
            f"Perturbación '{nombre}' no válida. "
            f"Disponibles: {list(PERTURBACIONES.keys())}"
        )
    return PERTURBACIONES[nombre]