# probar.py
from tsp.io import leer_archivo_tsp
from tsp.distancias import distancia_euclidea_total, calcular_mst
from tsp.split import split_tsp_dp, reconstruir_camino

RUTA = "/Users/juanesteban/Documents/Un/PI/Codigo/tsp_project/datos/ali535.tsp"

# 1. Leer nodos
nodos = leer_archivo_tsp(RUTA)
print(f"Nodos leídos: {len(nodos)}")

# 2. Costo de la secuencia original (sin optimizar)
costo_inicial = distancia_euclidea_total(nodos)
print(f"Costo inicial: {costo_inicial:.4f}")

# 3. Correr el Split DP una sola vez con todos los movimientos activos
V, P, M, stats_mov, _ = split_tsp_dp(
    nodos,
    usar_m1=True,
    usar_m2=True,
    usar_m3=True,
    usar_m4=True,
)

# 4. Reconstruir el camino
camino, stats_mov = reconstruir_camino(nodos, P, M, stats_mov)
costo_final = distancia_euclidea_total(camino)
print(f"Costo final:   {costo_final:.4f}")
print(f"Mejora:        {costo_inicial - costo_final:.4f}")
print(f"Mejora %:      {(costo_inicial - costo_final) / costo_inicial * 100:.4f}%")

# 5. MST y gap
lb_mst = calcular_mst(nodos)
gap_pct = (costo_final - lb_mst) / lb_mst * 100
print(f"Cota MST:      {lb_mst:.4f}")
print(f"Gap %:         {gap_pct:.4f}%")

# 6. Movimientos usados
print("\n=== MOVIMIENTOS ===")
for k in sorted(stats_mov):
    m = stats_mov[k]
    print(f"  {m['nombre']}  evaluado={m['evaluado']:>6}  ganador_local={m['ganador_local']:>6}  usado_final={m['usado_final']:>4}  mejora_vs_M0={m['mejora_vs_M0']:>6}")