import yaml
import os
from tsp.defaults import CARPETA_DATOS, DEFAULTS


def cargar_config(ruta_yaml: str) -> dict:
    """
    Lee un archivo YAML de experimento y completa los valores faltantes
    con los defaults definidos en tsp/defaults.py.
    """
    with open(ruta_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    modos_validos = (
        "diagnostico",
        "busqueda_local",
        "comparativo",
        "randomkeys_estabilidad",
        "perturbacion",
        "paper_busqueda_local",
        "paper_ils",
        "paper_brkga",
        "paper_completo",
    )
    modo = cfg.get("modo")
    if modo not in modos_validos:
        raise ValueError(
            f"Campo 'modo' inválido: '{modo}'. "
            f"Valores permitidos: {' | '.join(modos_validos)}"
        )

    # Modo maestro: solo necesita la lista de YAMLs a ejecutar
    if modo == "paper_completo":
        experimentos_a_correr = cfg.get("experimentos", [])
        if not experimentos_a_correr:
            raise ValueError(
                "Modo 'paper_completo' requiere lista 'experimentos' con rutas a YAMLs"
            )
        return {
            "modo": modo,
            "experimentos": experimentos_a_correr,
        }

    carpeta_entrada = cfg.get("carpeta_entrada", CARPETA_DATOS)
    if not os.path.isdir(carpeta_entrada):
        raise FileNotFoundError(f"carpeta_entrada no existe: {carpeta_entrada}")

    nombre_experimento = os.path.splitext(os.path.basename(ruta_yaml))[0]
    carpeta_salida = cfg.get(
        "carpeta_salida",
        os.path.join(DEFAULTS["carpeta_salida"], nombre_experimento),
    )

    # ---- comunes ----
    max_iter = cfg.get("max_iter", DEFAULTS["max_iter"])
    tolerancia = cfg.get("tolerancia", DEFAULTS["tolerancia"])
    timeout = cfg.get("timeout", None)
    seed = cfg.get("seed_randomkeys", DEFAULTS["seed_randomkeys"])
    heuristicas = cfg.get("heuristicas", DEFAULTS["heuristicas"])
    semilla_inicio = cfg.get("semilla_inicio", DEFAULTS["seed_randomkeys"])
    n_semillas = cfg.get("n_semillas", 20)

    # ---- perturbación (modo viejo) ----
    max_iter_perturb = cfg.get("max_iter_perturb", 10)
    n_movimientos_perturb = cfg.get("n_movimientos_perturb", 5)

    # ---- paper común ----
    metodos = cfg.get("metodos", None)
    perturbaciones = cfg.get("perturbaciones", None)
    max_perturbaciones = cfg.get("max_perturbaciones", 20)
    timeout_total = cfg.get("timeout_total", None)
    seed_perturbacion = cfg.get("seed_perturbacion", 42)

    # ---- BRKGA específicos ----
    decoders = cfg.get("decoders", None)
    n_poblacion = cfg.get("n_poblacion", 100)
    n_generaciones = cfg.get("n_generaciones", 100)
    pct_elite = cfg.get("pct_elite", 0.25)
    pct_mutantes = cfg.get("pct_mutantes", 0.20)
    prob_elite_crossover = cfg.get("prob_elite_crossover", 0.70)

    # ---- configs movimientos ----
    configs_mov = cfg.get("configuraciones_movimientos", "todas")
    if configs_mov == "todas":
        correr_todas = True
        solo_config = None
    else:
        correr_todas = False
        m = cfg.get("configuraciones_movimientos", DEFAULTS["movimientos"])
        solo_config = {
            "usar_m1": m.get("usar_m1", True),
            "usar_m2": m.get("usar_m2", True),
            "usar_m3": m.get("usar_m3", True),
            "usar_m4": m.get("usar_m4", True),
            "config": (
                f"M1{int(m.get('usar_m1', True))}"
                f"_M2{int(m.get('usar_m2', True))}"
                f"_M3{int(m.get('usar_m3', True))}"
                f"_M4{int(m.get('usar_m4', True))}"
            ),
        }

    return {
        "modo": modo,
        "carpeta_entrada": carpeta_entrada,
        "carpeta_salida": carpeta_salida,
        "max_iter": max_iter,
        "tolerancia": tolerancia,
        "timeout": timeout,
        "seed_randomkeys": seed,
        "heuristicas_a_probar": heuristicas,
        "correr_todas_configuraciones": correr_todas,
        "solo_config": solo_config,
        "semilla_inicio": semilla_inicio,
        "n_semillas": n_semillas,
        "max_iter_perturb": max_iter_perturb,
        "n_movimientos_perturb": n_movimientos_perturb,
        "metodos": metodos,
        "perturbaciones": perturbaciones,
        "max_perturbaciones": max_perturbaciones,
        "timeout_total": timeout_total,
        "seed_perturbacion": seed_perturbacion,
        "decoders": decoders,
        "n_poblacion": n_poblacion,
        "n_generaciones": n_generaciones,
        "pct_elite": pct_elite,
        "pct_mutantes": pct_mutantes,
        "prob_elite_crossover": prob_elite_crossover,
    }