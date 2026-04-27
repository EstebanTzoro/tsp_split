# probar_entrega1.py
from tsp.io import leer_archivo_tsp
from tsp.heuristicas import obtener_heuristica
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

RUTA = "datos/eil51.tsp"
nodos = leer_archivo_tsp(RUTA)

# Construir solución inicial con NNH
nnh = obtener_heuristica("NNH")
ruta_ini, costo_ini = nnh(nodos)
print(f"Costo inicial NNH: {costo_ini:.2f}")

# Probar 2-opt Best
_, c, _, razon, t = busqueda_local_2opt(ruta_ini, max_iter=50)
print(f"2-opt Best:   {c:.2f} | {razon} | {t:.2f}s")

# Probar 2-opt First
_, c, _, razon, t = busqueda_local_2opt_first(ruta_ini, max_iter=50)
print(f"2-opt First:  {c:.2f} | {razon} | {t:.2f}s")

# Probar Split M0-M1
_, c, _, _, _, razon, t = busqueda_local_split(
    ruta_ini, max_iter=50, usar_m1=True, usar_m2=False, usar_m3=False, usar_m4=False
)
print(f"Split M0-M1:  {c:.2f} | {razon} | {t:.2f}s")

# Probar VND
_, c, df_hist, razon, t, _ = vnd_split(ruta_ini, max_iter_por_vecindario=50)
print(f"VND:          {c:.2f} | {razon} | {t:.2f}s")
print("\nHistorial VND:")
print(df_hist[["vecindario", "costo_salida", "hubo_mejora"]])

# Probar perturbaciones
ruta_db = perturbacion_double_bridge(ruta_ini, seed=42)
ruta_mm = perturbacion_multi_movimiento(ruta_ini, seed=42)
print(f"\nLongitud original: {len(ruta_ini)}")
print(f"Longitud Double-Bridge: {len(ruta_db)} (debe ser igual)")
print(f"Longitud Multi-Movimiento: {len(ruta_mm)} (debe ser igual)")