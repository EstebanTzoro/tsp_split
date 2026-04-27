# tsp_project/tsp/defaults.py
# Valores por defecto globales del proyecto.

CARPETA_DATOS = "/Users/juanesteban/Documents/Un/PI/Codigo/tsp_project/datos"

DEFAULTS = {
    "max_iter": 100,
    "tolerancia": 1e-9,
    "seed_randomkeys": 11,
    "heuristicas": ["NNH", "INSERCION", "RANDOM_KEYS"],
    "movimientos": {
        "usar_m1": True,
        "usar_m2": True,
        "usar_m3": True,
        "usar_m4": True,
    },
    "carpeta_salida": "resultados",
}
