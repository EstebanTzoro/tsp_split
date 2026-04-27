"""
run.py — Entry point del proyecto TSP Split

Uso:
    python run.py configs/mi_experimento.yaml

El archivo YAML define qué experimento correr y con qué parámetros.

Modos disponibles:
    diagnostico              — Diagnóstico de movimientos del Split
    busqueda_local           — BL iterativa con Split
    comparativo              — BL Split vs 2-opt
    randomkeys_estabilidad   — Estabilidad multisemilla de RANDOM_KEYS
    perturbacion             — BL con perturbación aleatoria

    paper_busqueda_local     — [PAPER EXP 1] BL comparativa (2-opt vs Split)
    paper_ils                — [PAPER EXP 2] ILS comparativo
    paper_brkga              — [PAPER EXP 3] BRKGA puro vs BRKGA + Split
    paper_completo           — Ejecuta los 3 experimentos del paper en cadena
"""
import sys
import os

from tsp.config_parser import cargar_config
from tsp.experimentos import (
    experimento_diagnostico,
    experimento_busqueda_local,
    experimento_comparativo,
    experimento_randomkeys_estabilidad,
    experimento_perturbacion,
    experimento_paper_busqueda_local,
    experimento_paper_ils,
    experimento_paper_brkga,
)
from tsp.visualizacion import (
    dashboard_busqueda_local,
    dashboard_comparativo,
    dashboard_randomkeys_estabilidad,
    dashboard_perturbacion,
    dashboard_paper_busqueda_local,
    dashboard_paper_ils,
    dashboard_paper_brkga,
)


def ejecutar_experimento(ruta_yaml: str):
    """Ejecuta UN experimento desde su archivo YAML."""
    if not os.path.isfile(ruta_yaml):
        print(f"ERROR: No se encontró el archivo: {ruta_yaml}")
        return False

    print(f"\nCargando configuración: {ruta_yaml}")
    cfg = cargar_config(ruta_yaml)

    modo = cfg.pop("modo")

    # ----- MODO MAESTRO (paper completo) -----
    if modo == "paper_completo":
        print(f"\n{'#'*60}")
        print(f"# MODO PAPER COMPLETO")
        print(f"# Ejecutando {len(cfg['experimentos'])} experimentos en cadena")
        print(f"{'#'*60}\n")
        for sub_yaml in cfg["experimentos"]:
            print(f"\n{'>'*60}")
            print(f"> Sub-experimento: {sub_yaml}")
            print(f"{'>'*60}")
            ejecutar_experimento(sub_yaml)
        return True

    # ----- MODOS NORMALES -----
    carpeta_graficos = os.path.join(cfg["carpeta_salida"], "graficos")

    print(f"Modo: {modo}")
    print(f"Carpeta entrada: {cfg['carpeta_entrada']}")
    print(f"Carpeta salida:  {cfg['carpeta_salida']}\n")

    # ------------------------------------------------------------------
    if modo == "diagnostico":
        experimento_diagnostico(
            carpeta_entrada=cfg["carpeta_entrada"],
            carpeta_salida=cfg["carpeta_salida"],
            correr_todas_configuraciones=cfg["correr_todas_configuraciones"],
            solo_config=cfg["solo_config"],
        )

    # ------------------------------------------------------------------
    elif modo == "busqueda_local":
        df_resultados, df_movimientos, df_convergencia, df_resumen_config = experimento_busqueda_local(
            carpeta_entrada=cfg["carpeta_entrada"],
            carpeta_salida=cfg["carpeta_salida"],
            heuristicas_a_probar=cfg["heuristicas_a_probar"],
            correr_todas_configuraciones=cfg["correr_todas_configuraciones"],
            solo_config=cfg["solo_config"],
            max_iter=cfg["max_iter"],
            tolerancia=cfg["tolerancia"],
            seed_randomkeys=cfg["seed_randomkeys"],
        )
        if not df_resultados.empty:
            instancia_ejemplo = df_resultados.sort_values("mejora_pct", ascending=False).iloc[0]["instancia"]
            dashboard_busqueda_local(
                df_resumen=df_resultados,
                df_iteraciones=df_convergencia,
                df_movimientos_agregado=df_movimientos,
                df_resumen_config=df_resumen_config,
                carpeta_salida=carpeta_graficos,
                instancia_ejemplo=instancia_ejemplo,
            )

    # ------------------------------------------------------------------
    elif modo == "comparativo":
        df_resultados, df_resumen, df_por_tamano, df_convergencia = experimento_comparativo(
            carpeta_entrada=cfg["carpeta_entrada"],
            carpeta_salida=cfg["carpeta_salida"],
            heuristicas_a_probar=cfg["heuristicas_a_probar"],
            correr_todas_configuraciones=cfg["correr_todas_configuraciones"],
            solo_config=cfg["solo_config"],
            max_iter=cfg["max_iter"],
            tolerancia=cfg["tolerancia"],
            seed_randomkeys=cfg["seed_randomkeys"],
        )
        if not df_resultados.empty:
            dashboard_comparativo(
                df_comparativo=df_resultados,
                df_iter_agregadas=df_convergencia,
                carpeta_salida=carpeta_graficos,
            )

    # ------------------------------------------------------------------
    elif modo == "randomkeys_estabilidad":
        df_resultados, df_resumen_config, df_convergencia = experimento_randomkeys_estabilidad(
            carpeta_entrada=cfg["carpeta_entrada"],
            carpeta_salida=cfg["carpeta_salida"],
            correr_todas_configuraciones=cfg["correr_todas_configuraciones"],
            solo_config=cfg["solo_config"],
            max_iter=cfg["max_iter"],
            tolerancia=cfg["tolerancia"],
            timeout=cfg["timeout"],
            semilla_inicio=cfg["semilla_inicio"],
            n_semillas=cfg["n_semillas"],
        )
        if not df_resultados.empty:
            dashboard_randomkeys_estabilidad(
                df_resultados=df_resultados,
                df_convergencia=df_convergencia,
                carpeta_salida=carpeta_graficos,
            )

    # ------------------------------------------------------------------
    elif modo == "perturbacion":
        df_resultados, df_convergencia, df_perturbaciones, df_resumen_config = experimento_perturbacion(
            carpeta_entrada=cfg["carpeta_entrada"],
            carpeta_salida=cfg["carpeta_salida"],
            heuristicas_a_probar=cfg["heuristicas_a_probar"],
            solo_config=cfg["solo_config"],
            max_iter_bl=cfg["max_iter"],
            max_iter_perturb=cfg.get("max_iter_perturb", 10),
            n_movimientos_perturb=cfg.get("n_movimientos_perturb", 5),
            tolerancia=cfg["tolerancia"],
            timeout=cfg.get("timeout"),
            seed_randomkeys=cfg["seed_randomkeys"],
        )
        if not df_resultados.empty:
            dashboard_perturbacion(
                df_resultados=df_resultados,
                df_convergencia=df_convergencia,
                df_perturbaciones=df_perturbaciones,
                carpeta_salida=carpeta_graficos,
            )

    # ------------------------------------------------------------------
    # PAPER EXP 1
    # ------------------------------------------------------------------
    elif modo == "paper_busqueda_local":
        experimento_paper_busqueda_local(
            carpeta_entrada=cfg["carpeta_entrada"],
            carpeta_salida=cfg["carpeta_salida"],
            heuristicas_a_probar=cfg["heuristicas_a_probar"],
            metodos=cfg["metodos"],
            max_iter=cfg["max_iter"],
            tolerancia=cfg["tolerancia"],
            timeout=cfg["timeout"],
            seed_randomkeys=cfg["seed_randomkeys"],
        )

    # ------------------------------------------------------------------
    # PAPER EXP 2
    # ------------------------------------------------------------------
    elif modo == "paper_ils":
        experimento_paper_ils(
            carpeta_entrada=cfg["carpeta_entrada"],
            carpeta_salida=cfg["carpeta_salida"],
            heuristicas_a_probar=cfg["heuristicas_a_probar"],
            metodos=cfg["metodos"],
            perturbaciones=cfg["perturbaciones"],
            max_iter_bl=cfg["max_iter"],
            max_perturbaciones=cfg["max_perturbaciones"],
            n_movimientos_perturb=cfg["n_movimientos_perturb"],
            tolerancia=cfg["tolerancia"],
            timeout_total=cfg["timeout_total"],
            seed_randomkeys=cfg["seed_randomkeys"],
            seed_perturbacion=cfg["seed_perturbacion"],
        )

    # ------------------------------------------------------------------
    # PAPER EXP 3
    # ------------------------------------------------------------------
    elif modo == "paper_brkga":
        experimento_paper_brkga(
            carpeta_entrada=cfg["carpeta_entrada"],
            carpeta_salida=cfg["carpeta_salida"],
            decoders=cfg["decoders"],
            n_semillas=cfg["n_semillas"],
            semilla_inicio=cfg["semilla_inicio"],
            n_poblacion=cfg["n_poblacion"],
            n_generaciones=cfg["n_generaciones"],
            pct_elite=cfg["pct_elite"],
            pct_mutantes=cfg["pct_mutantes"],
            prob_elite_crossover=cfg["prob_elite_crossover"],
            timeout=cfg["timeout"],
        )

    print("\n✓ Experimento finalizado.")
    return True


def main():
    if len(sys.argv) < 2:
        print("Uso: python run.py <ruta_al_yaml>")
        print("Ejemplo: python run.py configs/exp1_busqueda_local_paper.yaml")
        print("         python run.py configs/paper_completo.yaml")
        sys.exit(1)

    ruta_yaml = sys.argv[1]
    ejecutar_experimento(ruta_yaml)


if __name__ == "__main__":
    main()