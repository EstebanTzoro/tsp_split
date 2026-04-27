# tsp/visualizacion.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.figsize": (12, 6),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

COLORES_HEURISTICAS = {
    "NNH":         "#2E86C1",
    "INSERCION":   "#27AE60",
    "RANDOM_KEYS": "#E67E22",
}

COLORES_METODOS = {
    "SPLIT": "#2E86C1",
    "2OPT":  "#E74C3C",
}


# =============================================================
# HELPERS INTERNOS
# =============================================================

def _filtrar(df, heuristica=None, config=None, instancia=None):
    df = df.copy()
    if heuristica is not None and "heuristica" in df.columns:
        vals = [heuristica] if isinstance(heuristica, str) else list(heuristica)
        df = df[df["heuristica"].isin(vals)]
    if config is not None and "config" in df.columns:
        vals = [config] if isinstance(config, str) else list(config)
        df = df[df["config"].isin(vals)]
    if instancia is not None and "instancia" in df.columns:
        vals = [instancia] if isinstance(instancia, str) else list(instancia)
        df = df[df["instancia"].isin(vals)]
    return df


def _rotar_xticks(ax, angle=45):
    for label in ax.get_xticklabels():
        label.set_rotation(angle)
        label.set_horizontalalignment("right")


def _annotate_bars(ax, decimals=2):
    for p in ax.patches:
        h = p.get_height()
        if pd.notnull(h):
            ax.annotate(
                f"{h:.{decimals}f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center", va="bottom", fontsize=8,
                xytext=(0, 3), textcoords="offset points",
            )


def _guardar(fig, carpeta_salida, nombre_archivo):
    if carpeta_salida:
        os.makedirs(carpeta_salida, exist_ok=True)
        ruta = os.path.join(carpeta_salida, nombre_archivo)
        fig.savefig(ruta, dpi=150, bbox_inches="tight")
        print(f"  Guardado: {nombre_archivo}")


def _color_heuristica(nombre):
    return COLORES_HEURISTICAS.get(nombre, "#7F8C8D")


def _color_metodo(nombre):
    return COLORES_METODOS.get(nombre, "#7F8C8D")


# =============================================================
# TABLAS RESUMEN
# =============================================================

def tabla_resumen_por_heuristica(df_resumen):
    cols = [c for c in ["mejora_pct", "gap_pct", "iteraciones_ejecutadas"] if c in df_resumen.columns]
    return df_resumen.groupby("heuristica", as_index=False)[cols].mean().sort_values("gap_pct")


def tabla_ranking_configuraciones(df_resumen_config, top_n=15):
    cols = [c for c in ["heuristica", "config", "mejora_pct", "gap_pct", "iteraciones_ejecutadas"]
            if c in df_resumen_config.columns]
    return df_resumen_config[cols].sort_values(["gap_pct", "mejora_pct"], ascending=[True, False]).head(top_n)


def tabla_movimientos_global(df_movimientos):
    return (
        df_movimientos.groupby("movimiento", as_index=False)
        [["evaluado", "mejora_vs_M0", "ganador_local", "usado_final"]]
        .mean().sort_values("usado_final", ascending=False)
    )


# =============================================================
# GRÁFICOS — DIAGNÓSTICO (7 gráficos)
# =============================================================

def plot_gap_por_instancia(df_resultados, carpeta_salida=None):
    """Gap % por instancia, ordenado de mayor a menor con línea de promedio."""
    df = df_resultados.sort_values("gap_pct", ascending=False)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(max(12, len(df) * 0.4), 6))
    ax.bar(df["instancia"], df["gap_pct"], color="#2E86C1", edgecolor="white")
    promedio = df["gap_pct"].mean()
    ax.axhline(promedio, linestyle="--", color="tomato", linewidth=1.2,
               label=f"Promedio: {promedio:.2f}%")
    ax.set_title("Gap % por instancia (Split DP — una pasada)")
    ax.set_ylabel("Gap % vs MST")
    ax.set_xlabel("Instancia")
    ax.legend()
    _rotar_xticks(ax, 75)
    plt.tight_layout()
    _guardar(fig, carpeta_salida, "01_gap_por_instancia.png")
    plt.close(fig)


def plot_mejora_por_instancia(df_resultados, carpeta_salida=None):
    """Mejora % por instancia, ordenado de mayor a menor con línea de promedio."""
    df = df_resultados.sort_values("mejora_pct", ascending=False)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(max(12, len(df) * 0.4), 6))
    ax.bar(df["instancia"], df["mejora_pct"], color="#27AE60", edgecolor="white")
    promedio = df["mejora_pct"].mean()
    ax.axhline(promedio, linestyle="--", color="tomato", linewidth=1.2,
               label=f"Promedio: {promedio:.2f}%")
    ax.set_title("Mejora % por instancia (respecto al orden original)")
    ax.set_ylabel("Mejora %")
    ax.set_xlabel("Instancia")
    ax.legend()
    _rotar_xticks(ax, 75)
    plt.tight_layout()
    _guardar(fig, carpeta_salida, "02_mejora_por_instancia.png")
    plt.close(fig)


def plot_gap_por_rango_nodos(df_resultados, carpeta_salida=None, prefijo="03"):
    """Boxplot de gap% agrupado por rango de nodos."""
    if df_resultados.empty or "rango_nodos" not in df_resultados.columns:
        return
    rangos = sorted(df_resultados["rango_nodos"].dropna().unique())
    pares = [(r, df_resultados.loc[df_resultados["rango_nodos"] == r, "gap_pct"].dropna().values)
             for r in rangos]
    pares = [(r, g) for r, g in pares if len(g) > 0]
    if not pares:
        return
    labels, datos = zip(*pares)
    colors = ["#AED6F1", "#85C1E9", "#5DADE2", "#2E86C1", "#1A5276"]
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(datos, labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors[:len(bp["boxes"])]):
        patch.set_facecolor(color)
    ax.set_title("Gap % según tamaño de instancia")
    ax.set_ylabel("Gap % vs MST")
    ax.set_xlabel("Rango de nodos")
    _rotar_xticks(ax, 20)
    plt.tight_layout()
    _guardar(fig, carpeta_salida, f"{prefijo}_gap_por_rango_nodos.png")
    plt.close(fig)


def plot_usado_final_por_movimiento(df_movimientos, carpeta_salida=None, prefijo="04"):
    """Barras de cuántas veces cada movimiento terminó en la solución final."""
    if df_movimientos.empty:
        return
    df_agg = (
        df_movimientos.groupby("movimiento", as_index=False)["usado_final"]
        .sum().sort_values("usado_final", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(df_agg["movimiento"], df_agg["usado_final"], color="#8E44AD", edgecolor="white")
    ax.set_title("Bloques en solución final por movimiento")
    ax.set_ylabel("Veces usado en solución final (total)")
    ax.set_xlabel("Movimiento")
    _annotate_bars(ax, decimals=0)
    plt.tight_layout()
    _guardar(fig, carpeta_salida, f"{prefijo}_usado_final_por_movimiento.png")
    plt.close(fig)


def plot_ganador_vs_usado_final(df_movimientos, carpeta_salida=None, prefijo="05"):
    """
    Barras agrupadas: ganador_local vs usado_final por movimiento.
    Revela si un movimiento gana localmente pero el DP global lo descarta.
    """
    if df_movimientos.empty:
        return
    df_agg = (
        df_movimientos.groupby("movimiento", as_index=False)[["ganador_local", "usado_final"]]
        .sum().sort_values("ganador_local", ascending=False)
    )
    movimientos = df_agg["movimiento"].tolist()
    x = np.arange(len(movimientos))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w / 2, df_agg["ganador_local"], width=w, label="Ganador local",
           color="#2E86C1", edgecolor="white")
    ax.bar(x + w / 2, df_agg["usado_final"], width=w, label="Usado en solución final",
           color="#8E44AD", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(movimientos)
    ax.set_title("Ganador local vs Usado en solución final por movimiento")
    ax.set_ylabel("Cantidad (total sobre todas las instancias)")
    ax.set_xlabel("Movimiento")
    ax.legend()
    plt.tight_layout()
    _guardar(fig, carpeta_salida, f"{prefijo}_ganador_vs_usado_final.png")
    plt.close(fig)


def plot_scatter_gap_vs_nodos(df_resultados, carpeta_salida=None, prefijo="06"):
    """Scatter gap% vs n_nodos con línea de tendencia."""
    if df_resultados.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_resultados["n_nodos"], df_resultados["gap_pct"],
               alpha=0.75, color="#2E86C1", edgecolors="white", linewidths=0.5)
    if len(df_resultados) > 2:
        z = np.polyfit(df_resultados["n_nodos"], df_resultados["gap_pct"], 1)
        x_line = np.linspace(df_resultados["n_nodos"].min(), df_resultados["n_nodos"].max(), 100)
        ax.plot(x_line, np.poly1d(z)(x_line), linestyle="--", color="tomato",
                linewidth=1.2, label="Tendencia")
        ax.legend()
    ax.set_title("Gap % vs Número de nodos")
    ax.set_xlabel("Número de nodos")
    ax.set_ylabel("Gap % vs MST")
    plt.tight_layout()
    _guardar(fig, carpeta_salida, f"{prefijo}_scatter_gap_vs_nodos.png")
    plt.close(fig)


def plot_scatter_mejora_vs_nodos(df_resultados, carpeta_salida=None, prefijo="07"):
    """Scatter mejora% vs n_nodos con línea de tendencia."""
    if df_resultados.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_resultados["n_nodos"], df_resultados["mejora_pct"],
               alpha=0.75, color="#27AE60", edgecolors="white", linewidths=0.5)
    if len(df_resultados) > 2:
        z = np.polyfit(df_resultados["n_nodos"], df_resultados["mejora_pct"], 1)
        x_line = np.linspace(df_resultados["n_nodos"].min(), df_resultados["n_nodos"].max(), 100)
        ax.plot(x_line, np.poly1d(z)(x_line), linestyle="--", color="tomato",
                linewidth=1.2, label="Tendencia")
        ax.legend()
    ax.set_title("Mejora % vs Número de nodos")
    ax.set_xlabel("Número de nodos")
    ax.set_ylabel("Mejora %")
    plt.tight_layout()
    _guardar(fig, carpeta_salida, f"{prefijo}_scatter_mejora_vs_nodos.png")
    plt.close(fig)


def dashboard_diagnostico(df_resultados, df_movimientos, carpeta_salida=None):
    """Genera y guarda los 7 gráficos del experimento de diagnóstico."""
    print(f"\nGenerando gráficos diagnóstico en: {carpeta_salida}")
    plot_gap_por_instancia(df_resultados, carpeta_salida)
    plot_mejora_por_instancia(df_resultados, carpeta_salida)
    plot_gap_por_rango_nodos(df_resultados, carpeta_salida, prefijo="03")
    plot_usado_final_por_movimiento(df_movimientos, carpeta_salida, prefijo="04")
    plot_ganador_vs_usado_final(df_movimientos, carpeta_salida, prefijo="05")
    plot_scatter_gap_vs_nodos(df_resultados, carpeta_salida, prefijo="06")
    plot_scatter_mejora_vs_nodos(df_resultados, carpeta_salida, prefijo="07")
    print("✓ 7 gráficos generados.\n")


# =============================================================
# GRÁFICOS — BÚSQUEDA LOCAL (11 gráficos)
# =============================================================

def plot_bl_gap_por_heuristica(df_resultados, carpeta_salida=None):
    """Gap % promedio por heurística."""
    if df_resultados.empty:
        return
    df = df_resultados.groupby("heuristica", as_index=False)["gap_pct"].mean().sort_values("gap_pct")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(df["heuristica"], df["gap_pct"],
           color=[_color_heuristica(h) for h in df["heuristica"]], edgecolor="white")
    _annotate_bars(ax)
    ax.set_title("Gap % promedio por heurística (búsqueda local)")
    ax.set_ylabel("Gap % vs MST")
    ax.set_xlabel("Heurística")
    plt.tight_layout()
    _guardar(fig, carpeta_salida, "01_gap_por_heuristica.png")
    plt.close(fig)


def plot_bl_mejora_por_heuristica(df_resultados, carpeta_salida=None):
    """Mejora % promedio por heurística."""
    if df_resultados.empty:
        return
    df = df_resultados.groupby("heuristica", as_index=False)["mejora_pct"].mean().sort_values("mejora_pct", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(df["heuristica"], df["mejora_pct"],
           color=[_color_heuristica(h) for h in df["heuristica"]], edgecolor="white")
    _annotate_bars(ax)
    ax.set_title("Mejora % promedio por heurística (búsqueda local)")
    ax.set_ylabel("Mejora %")
    ax.set_xlabel("Heurística")
    plt.tight_layout()
    _guardar(fig, carpeta_salida, "02_mejora_por_heuristica.png")
    plt.close(fig)


def plot_bl_convergencia_por_heuristica(df_convergencia, carpeta_salida=None):
    """
    Cómo cae el costo iteración a iteración para NNH, INSERCION y RANDOM_KEYS.
    Gráfico clave para el informe.
    """
    if df_convergencia.empty or "heuristica" not in df_convergencia.columns:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    for h in sorted(df_convergencia["heuristica"].dropna().unique()):
        d = (
            df_convergencia[df_convergencia["heuristica"] == h]
            .groupby("iteracion", as_index=False)["costo_salida"].mean()
            .sort_values("iteracion")
        )
        ax.plot(d["iteracion"], d["costo_salida"], marker="o", markersize=4,
                label=h, color=_color_heuristica(h), linewidth=2)
    ax.set_title("Convergencia del costo por heurística (promedio sobre instancias)")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Costo promedio de salida")
    ax.legend()
    plt.tight_layout()
    _guardar(fig, carpeta_salida, "03_convergencia_por_heuristica.png")
    plt.close(fig)


def plot_bl_iteraciones_por_heuristica(df_resultados, carpeta_salida=None):
    """Boxplot de iteraciones hasta convergencia por heurística."""
    if df_resultados.empty or "iteraciones_ejecutadas" not in df_resultados.columns:
        return
    heuristicas = sorted(df_resultados["heuristica"].dropna().unique())
    pares = [(h, df_resultados.loc[df_resultados["heuristica"] == h, "iteraciones_ejecutadas"].dropna().values)
             for h in heuristicas]
    pares = [(h, g) for h, g in pares if len(g) > 0]
    if not pares:
        return
    labels, datos = zip(*pares)
    fig, ax = plt.subplots(figsize=(9, 6))
    bp = ax.boxplot(datos, labels=labels, patch_artist=True)
    for patch, h in zip(bp["boxes"], labels):
        patch.set_facecolor(_color_heuristica(h))
    ax.set_title("Iteraciones hasta convergencia por heurística")
    ax.set_ylabel("Número de iteraciones")
    ax.set_xlabel("Heurística")
    plt.tight_layout()
    _guardar(fig, carpeta_salida, "04_iteraciones_por_heuristica.png")
    plt.close(fig)


def plot_bl_gap_por_rango_nodos(df_resultados, carpeta_salida=None):
    """Gap % por rango de nodos con búsqueda local."""
    plot_gap_por_rango_nodos(df_resultados, carpeta_salida, prefijo="05")


def plot_bl_mejor_config_por_heuristica(df_resumen_config, carpeta_salida=None, top_n=8):
    """Top configuraciones de movimientos por heurística según gap%."""
    if df_resumen_config.empty or "heuristica" not in df_resumen_config.columns:
        return
    heuristicas = sorted(df_resumen_config["heuristica"].dropna().unique())
    fig, axes = plt.subplots(1, len(heuristicas), figsize=(7 * len(heuristicas), 6), sharey=False)
    if len(heuristicas) == 1:
        axes = [axes]
    for ax, h in zip(axes, heuristicas):
        df_h = df_resumen_config[df_resumen_config["heuristica"] == h].sort_values("gap_pct").head(top_n)
        ax.barh(df_h["config"], df_h["gap_pct"], color=_color_heuristica(h), edgecolor="white")
        ax.set_title(h)
        ax.set_xlabel("Gap % vs MST")
        ax.invert_yaxis()
    fig.suptitle(f"Top {top_n} configuraciones por heurística (menor gap)", fontsize=13)
    plt.tight_layout()
    _guardar(fig, carpeta_salida, "06_mejor_config_por_heuristica.png")
    plt.close(fig)


def plot_bl_heatmap_config_gap(df_resumen_config, carpeta_salida=None):
    """Heatmap: configuración M1-M4 vs gap% por heurística."""
    if df_resumen_config.empty or "heuristica" not in df_resumen_config.columns:
        return
    pivot = df_resumen_config.pivot_table(
        index="config", columns="heuristica", values="gap_pct", aggfunc="mean"
    ).fillna(0)
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.35)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Gap % por configuración de movimientos y heurística")
    plt.colorbar(im, ax=ax, label="Gap %")
    plt.tight_layout()
    _guardar(fig, carpeta_salida, "07_heatmap_config_gap.png")
    plt.close(fig)


def plot_bl_movimientos_por_heuristica(df_movimientos, metrica="usado_final", carpeta_salida=None):
    """Uso de cada movimiento desglosado por heurística."""
    if df_movimientos.empty or "heuristica" not in df_movimientos.columns:
        return
    df_agg = df_movimientos.groupby(["heuristica", "movimiento"], as_index=False)[metrica].sum()
    heuristicas = sorted(df_agg["heuristica"].dropna().unique())
    movimientos = sorted(df_agg["movimiento"].dropna().unique())
    x = np.arange(len(movimientos))
    w = 0.8 / len(heuristicas)
    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, h in enumerate(heuristicas):
        df_h = df_agg[df_agg["heuristica"] == h].set_index("movimiento").reindex(movimientos).fillna(0)
        offset = (idx - len(heuristicas) / 2 + 0.5) * w
        ax.bar(x + offset, df_h[metrica], width=w, label=h,
               color=_color_heuristica(h), edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(movimientos)
    ax.set_title(f"{metrica} por movimiento y heurística")
    ax.set_ylabel(metrica)
    ax.set_xlabel("Movimiento")
    ax.legend()
    plt.tight_layout()
    nombre = "08_usado_final_por_heuristica.png" if metrica == "usado_final" else f"08b_{metrica}_por_heuristica.png"
    _guardar(fig, carpeta_salida, nombre)
    plt.close(fig)


def plot_bl_scatter_gap_vs_nodos(df_resultados, carpeta_salida=None):
    """Scatter gap% vs n_nodos coloreado por heurística."""
    if df_resultados.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for h in sorted(df_resultados["heuristica"].dropna().unique()):
        d = df_resultados[df_resultados["heuristica"] == h]
        ax.scatter(d["n_nodos"], d["gap_pct"], alpha=0.7, label=h,
                   color=_color_heuristica(h), edgecolors="white", linewidths=0.5)
    ax.set_title("Gap % vs Número de nodos por heurística")
    ax.set_xlabel("Número de nodos")
    ax.set_ylabel("Gap % vs MST")
    ax.legend()
    plt.tight_layout()
    _guardar(fig, carpeta_salida, "09_scatter_gap_vs_nodos.png")
    plt.close(fig)


def plot_bl_scatter_gap_vs_tiempo(df_resultados, carpeta_salida=None):
    """Scatter gap% vs tiempo — ¿los métodos más lentos dan mejor solución?"""
    col_tiempo = next((c for c in ["tiempo_seg", "tiempo_split_seg"] if c in df_resultados.columns), None)
    if df_resultados.empty or col_tiempo is None:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for h in sorted(df_resultados["heuristica"].dropna().unique()):
        d = df_resultados[df_resultados["heuristica"] == h]
        ax.scatter(d[col_tiempo], d["gap_pct"], alpha=0.7, label=h,
                   color=_color_heuristica(h), edgecolors="white", linewidths=0.5)
    ax.set_title("Gap % vs Tiempo de ejecución")
    ax.set_xlabel("Tiempo (segundos)")
    ax.set_ylabel("Gap % vs MST")
    ax.legend()
    plt.tight_layout()
    _guardar(fig, carpeta_salida, "10_scatter_gap_vs_tiempo.png")
    plt.close(fig)


def plot_bl_ganador_vs_usado_por_heuristica(df_movimientos, carpeta_salida=None):
    """Ganador local vs usado_final desglosado por heurística."""
    plot_bl_movimientos_por_heuristica(df_movimientos, metrica="ganador_local", carpeta_salida=carpeta_salida)


def dashboard_busqueda_local(
    df_resumen, df_iteraciones, df_movimientos_agregado, df_resumen_config,
    carpeta_salida=None, instancia_ejemplo=None, heuristica_foco=None, config_foco=None,
):
    """11 gráficos del experimento de búsqueda local."""
    print(f"\nGenerando gráficos búsqueda local en: {carpeta_salida}")
    plot_bl_gap_por_heuristica(df_resumen, carpeta_salida)
    plot_bl_mejora_por_heuristica(df_resumen, carpeta_salida)
    plot_bl_convergencia_por_heuristica(df_iteraciones, carpeta_salida)
    plot_bl_iteraciones_por_heuristica(df_resumen, carpeta_salida)
    plot_bl_gap_por_rango_nodos(df_resumen, carpeta_salida)
    plot_bl_mejor_config_por_heuristica(df_resumen_config, carpeta_salida)
    plot_bl_heatmap_config_gap(df_resumen_config, carpeta_salida)
    plot_bl_movimientos_por_heuristica(df_movimientos_agregado, metrica="usado_final", carpeta_salida=carpeta_salida)
    plot_bl_scatter_gap_vs_nodos(df_resumen, carpeta_salida)
    plot_bl_scatter_gap_vs_tiempo(df_resumen, carpeta_salida)
    plot_bl_ganador_vs_usado_por_heuristica(df_movimientos_agregado, carpeta_salida)
    print("✓ 11 gráficos generados.\n")


# =============================================================
# GRÁFICOS — COMPARATIVO SPLIT VS 2-OPT (10 gráficos)
# =============================================================

def plot_comp_convergencia_split_vs_2opt(df_convergencia, heuristica=None, carpeta_salida=None):
    """
    Costo por iteración de Split y 2-Opt en la misma gráfica.
    El gráfico más importante del informe.
    """
    if df_convergencia.empty or "metodo" not in df_convergencia.columns:
        return
    df = df_convergencia.copy()
    if heuristica and "heuristica" in df.columns:
        df = df[df["heuristica"] == heuristica]
    fig, ax = plt.subplots(figsize=(12, 6))
    for metodo in sorted(df["metodo"].dropna().unique()):
        d = (
            df[df["metodo"] == metodo]
            .groupby("iteracion", as_index=False)["costo_salida"].mean()
            .sort_values("iteracion")
        )
        ax.plot(d["iteracion"], d["costo_salida"], marker="o", markersize=4,
                label=metodo, color=_color_metodo(metodo), linewidth=2)
    titulo = "Convergencia: Split vs 2-Opt"
    if heuristica:
        titulo += f" ({heuristica})"
    ax.set_title(titulo + " — promedio sobre instancias")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Costo promedio de salida")
    ax.legend()
    plt.tight_layout()
    sufijo = f"_{heuristica}" if heuristica else ""
    _guardar(fig, carpeta_salida, f"01_convergencia_split_vs_2opt{sufijo}.png")
    plt.close(fig)


def plot_comp_gap_por_heuristica(df_resultados, carpeta_salida=None):
    """Gap % promedio Split vs 2-Opt por heurística."""
    if df_resultados.empty:
        return
    df = df_resultados.groupby("heuristica", as_index=False)[["gap_pct_split", "gap_pct_2opt"]].mean()
    x, w = np.arange(len(df)), 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w / 2, df["gap_pct_split"], width=w, label="Split",
           color=_color_metodo("SPLIT"), edgecolor="white")
    ax.bar(x + w / 2, df["gap_pct_2opt"], width=w, label="2-Opt",
           color=_color_metodo("2OPT"), edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(df["heuristica"])
    ax.set_title("Gap % promedio: Split vs 2-Opt por heurística")
    ax.set_ylabel("Gap % vs MST")
    ax.legend()
    _annotate_bars(ax)
    plt.tight_layout()
    _guardar(fig, carpeta_salida, "02_gap_por_heuristica.png")
    plt.close(fig)


def plot_comp_mejora_por_heuristica(df_resultados, carpeta_salida=None):
    """Mejora % promedio Split vs 2-Opt por heurística."""
    if df_resultados.empty:
        return
    df = df_resultados.groupby("heuristica", as_index=False)[["mejora_pct_split", "mejora_pct_2opt"]].mean()
    x, w = np.arange(len(df)), 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w / 2, df["mejora_pct_split"], width=w, label="Split",
           color=_color_metodo("SPLIT"), edgecolor="white")
    ax.bar(x + w / 2, df["mejora_pct_2opt"], width=w, label="2-Opt",
           color=_color_metodo("2OPT"), edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(df["heuristica"])
    ax.set_title("Mejora % promedio: Split vs 2-Opt por heurística")
    ax.set_ylabel("Mejora %")
    ax.legend()
    _annotate_bars(ax)
    plt.tight_layout()
    _guardar(fig, carpeta_salida, "03_mejora_por_heuristica.png")
    plt.close(fig)


def plot_comp_tiempo_por_heuristica(df_resultados, carpeta_salida=None):
    """Tiempo promedio Split vs 2-Opt por heurística."""
    if df_resultados.empty:
        return
    df = df_resultados.groupby("heuristica", as_index=False)[["tiempo_split_seg", "tiempo_2opt_seg"]].mean()
    x, w = np.arange(len(df)), 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w / 2, df["tiempo_split_seg"], width=w, label="Split",
           color=_color_metodo("SPLIT"), edgecolor="white")
    ax.bar(x + w / 2, df["tiempo_2opt_seg"], width=w, label="2-Opt",
           color=_color_metodo("2OPT"), edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(df["heuristica"])
    ax.set_title("Tiempo promedio por heurística: Split vs 2-Opt")
    ax.set_ylabel("Segundos")
    ax.legend()
    _annotate_bars(ax)
    plt.tight_layout()
    _guardar(fig, carpeta_salida, "04_tiempo_por_heuristica.png")
    plt.close(fig)


def plot_comp_victorias_por_heuristica(df_resultados, carpeta_salida=None):
    """% victorias Split vs 2-Opt desglosado por heurística."""
    if df_resultados.empty:
        return
    df = df_resultados.groupby("heuristica", as_index=False)[["gana_split", "gana_2opt", "empate"]].mean()
    df[["gana_split", "gana_2opt", "empate"]] *= 100
    x, w = np.arange(len(df)), 0.25
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w,     df["gana_split"], width=w, label="Gana Split",
           color=_color_metodo("SPLIT"), edgecolor="white")
    ax.bar(x,         df["gana_2opt"],  width=w, label="Gana 2-Opt",
           color=_color_metodo("2OPT"), edgecolor="white")
    ax.bar(x + w,     df["empate"],     width=w, label="Empate",
           color="#95A5A6", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(df["heuristica"])
    ax.set_title("% Victorias Split vs 2-Opt por heurística")
    ax.set_ylabel("Porcentaje (%)")
    ax.legend()
    plt.tight_layout()
    _guardar(fig, carpeta_salida, "05_victorias_por_heuristica.png")
    plt.close(fig)


def plot_comp_ventaja_por_rango_nodos(df_resultados, carpeta_salida=None):
    """
    Ventaja de Split por rango de nodos.
    Verde = Split gana, rojo = 2-Opt gana.
    """
    if df_resultados.empty or "rango_nodos" not in df_resultados.columns:
        return
    df = (
        df_resultados.groupby("rango_nodos", as_index=False)["ventaja_split"]
        .mean().sort_values("rango_nodos")
    )
    colores = ["#27AE60" if v >= 0 else "#E74C3C" for v in df["ventaja_split"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df["rango_nodos"], df["ventaja_split"], color=colores, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Ventaja de Split vs 2-Opt por tamaño de instancia\n(positivo = Split mejor)")
    ax.set_ylabel("Diferencia de costo (2-Opt − Split)")
    ax.set_xlabel("Rango de nodos")
    _rotar_xticks(ax, 20)
    plt.tight_layout()
    _guardar(fig, carpeta_salida, "06_ventaja_por_rango_nodos.png")
    plt.close(fig)


def plot_comp_gap_por_rango_nodos(df_resultados, carpeta_salida=None):
    """Gap % Split vs 2-Opt por rango de nodos — líneas."""
    if df_resultados.empty or "rango_nodos" not in df_resultados.columns:
        return
    df = (
        df_resultados.groupby("rango_nodos", as_index=False)[["gap_pct_split", "gap_pct_2opt"]]
        .mean().sort_values("rango_nodos")
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["rango_nodos"], df["gap_pct_split"], marker="o", label="Split",
            color=_color_metodo("SPLIT"), linewidth=2)
    ax.plot(df["rango_nodos"], df["gap_pct_2opt"], marker="s", label="2-Opt",
            color=_color_metodo("2OPT"), linewidth=2)
    ax.set_title("Gap % Split vs 2-Opt según tamaño de instancia")
    ax.set_ylabel("Gap % vs MST")
    ax.set_xlabel("Rango de nodos")
    ax.legend()
    _rotar_xticks(ax, 20)
    plt.tight_layout()
    _guardar(fig, carpeta_salida, "07_gap_por_rango_nodos.png")
    plt.close(fig)


def plot_comp_tiempo_por_rango_nodos(df_resultados, carpeta_salida=None):
    """Tiempo Split vs 2-Opt por rango de nodos."""
    if df_resultados.empty or "rango_nodos" not in df_resultados.columns:
        return
    df = (
        df_resultados.groupby("rango_nodos", as_index=False)[["tiempo_split_seg", "tiempo_2opt_seg"]]
        .mean().sort_values("rango_nodos")
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["rango_nodos"], df["tiempo_split_seg"], marker="o", label="Split",
            color=_color_metodo("SPLIT"), linewidth=2)
    ax.plot(df["rango_nodos"], df["tiempo_2opt_seg"], marker="s", label="2-Opt",
            color=_color_metodo("2OPT"), linewidth=2)
    ax.set_title("Tiempo de ejecución según tamaño de instancia")
    ax.set_ylabel("Tiempo promedio (segundos)")
    ax.set_xlabel("Rango de nodos")
    ax.legend()
    _rotar_xticks(ax, 20)
    plt.tight_layout()
    _guardar(fig, carpeta_salida, "08_tiempo_por_rango_nodos.png")
    plt.close(fig)


def plot_comp_scatter_gap_vs_tiempo(df_resultados, carpeta_salida=None):
    """
    Scatter gap% vs tiempo para Split y 2-Opt.
    Muestra el tradeoff calidad/velocidad.
    """
    if df_resultados.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_resultados["tiempo_split_seg"], df_resultados["gap_pct_split"],
               alpha=0.6, label="Split", color=_color_metodo("SPLIT"),
               edgecolors="white", linewidths=0.5)
    ax.scatter(df_resultados["tiempo_2opt_seg"], df_resultados["gap_pct_2opt"],
               alpha=0.6, label="2-Opt", color=_color_metodo("2OPT"),
               edgecolors="white", linewidths=0.5)
    ax.set_title("Gap % vs Tiempo de ejecución: Split vs 2-Opt")
    ax.set_xlabel("Tiempo (segundos)")
    ax.set_ylabel("Gap % vs MST")
    ax.legend()
    plt.tight_layout()
    _guardar(fig, carpeta_salida, "09_scatter_gap_vs_tiempo.png")
    plt.close(fig)


def plot_comp_heatmap_config_gap(df_resumen, carpeta_salida=None):
    """
    Heatmap: las 16 configuraciones M1-M4 vs gap% por heurística.
    Ideal para la sección de resultados del informe.
    """
    if df_resumen.empty or "config_split" not in df_resumen.columns:
        return
    pivot = df_resumen.pivot_table(
        index="config_split", columns="heuristica", values="gap_pct_split", aggfunc="mean"
    ).fillna(0)
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.35)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Gap % Split por configuración de movimientos y heurística")
    plt.colorbar(im, ax=ax, label="Gap %")
    plt.tight_layout()
    _guardar(fig, carpeta_salida, "10_heatmap_config_gap.png")
    plt.close(fig)


def dashboard_comparativo(df_comparativo, df_iter_agregadas, carpeta_salida=None):
    """10 gráficos del experimento comparativo Split vs 2-Opt."""
    print(f"\nGenerando gráficos comparativo en: {carpeta_salida}")

    # Convergencia general + una por heurística
    plot_comp_convergencia_split_vs_2opt(df_iter_agregadas, heuristica=None, carpeta_salida=carpeta_salida)
    if "heuristica" in df_iter_agregadas.columns:
        for h in sorted(df_iter_agregadas["heuristica"].dropna().unique()):
            plot_comp_convergencia_split_vs_2opt(df_iter_agregadas, heuristica=h, carpeta_salida=carpeta_salida)

    plot_comp_gap_por_heuristica(df_comparativo, carpeta_salida)
    plot_comp_mejora_por_heuristica(df_comparativo, carpeta_salida)
    plot_comp_tiempo_por_heuristica(df_comparativo, carpeta_salida)
    plot_comp_victorias_por_heuristica(df_comparativo, carpeta_salida)
    plot_comp_ventaja_por_rango_nodos(df_comparativo, carpeta_salida)
    plot_comp_gap_por_rango_nodos(df_comparativo, carpeta_salida)
    plot_comp_tiempo_por_rango_nodos(df_comparativo, carpeta_salida)
    plot_comp_scatter_gap_vs_tiempo(df_comparativo, carpeta_salida)
    plot_comp_heatmap_config_gap(df_comparativo, carpeta_salida)

    print("✓ Gráficos comparativo generados.\n")

    # =============================================================
# GRÁFICOS — RANDOM_KEYS MULTISEMILLA
# =============================================================

def _boxplot_por_config(df, columna, titulo, ylabel, nombre_archivo, carpeta_salida=None, color="#E67E22"):
    if df.empty or "config" not in df.columns or columna not in df.columns:
        return

    configs = sorted(df["config"].dropna().unique())
    pares = [
        (cfg, df.loc[df["config"] == cfg, columna].dropna().values)
        for cfg in configs
    ]
    pares = [(cfg, vals) for cfg, vals in pares if len(vals) > 0]
    if not pares:
        return

    labels, datos = zip(*pares)

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.7), 6))
    bp = ax.boxplot(datos, labels=labels, patch_artist=True)

    for patch in bp["boxes"]:
        patch.set_facecolor(color)

    ax.set_title(titulo)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Configuración")
    _rotar_xticks(ax, 45)
    plt.tight_layout()
    _guardar(fig, carpeta_salida, nombre_archivo)
    plt.close(fig)


def plot_rk_boxplot_gap_por_config(df_resultados, carpeta_salida=None):
    _boxplot_por_config(
        df_resultados,
        columna="gap_pct",
        titulo="Random Keys multisemilla — Gap % por configuración",
        ylabel="Gap % vs MST",
        nombre_archivo="01_boxplot_gap_por_config.png",
        carpeta_salida=carpeta_salida,
        color="#E67E22",
    )


def plot_rk_boxplot_mejora_por_config(df_resultados, carpeta_salida=None):
    _boxplot_por_config(
        df_resultados,
        columna="mejora_pct",
        titulo="Random Keys multisemilla — Mejora % por configuración",
        ylabel="Mejora %",
        nombre_archivo="02_boxplot_mejora_por_config.png",
        carpeta_salida=carpeta_salida,
        color="#F5B041",
    )


def plot_rk_boxplot_tiempo_por_config(df_resultados, carpeta_salida=None):
    _boxplot_por_config(
        df_resultados,
        columna="tiempo_seg",
        titulo="Random Keys multisemilla — Tiempo por configuración",
        ylabel="Tiempo (segundos)",
        nombre_archivo="03_boxplot_tiempo_por_config.png",
        carpeta_salida=carpeta_salida,
        color="#AF7AC5",
    )


def plot_rk_boxplot_iteraciones_por_config(df_resultados, carpeta_salida=None):
    _boxplot_por_config(
        df_resultados,
        columna="iteraciones_ejecutadas",
        titulo="Random Keys multisemilla — Iteraciones por configuración",
        ylabel="Iteraciones ejecutadas",
        nombre_archivo="04_boxplot_iteraciones_por_config.png",
        carpeta_salida=carpeta_salida,
        color="#5DADE2",
    )


def plot_rk_convergencia_promedio_por_config(df_convergencia, carpeta_salida=None, top_n=6):
    if df_convergencia.empty or "config" not in df_convergencia.columns:
        return

    resumen_cfg = (
        df_convergencia.groupby("config", as_index=False)["costo_salida"]
        .last()
        .sort_values("costo_salida")
        .head(top_n)
    )

    configs_top = resumen_cfg["config"].tolist()
    df_plot = df_convergencia[df_convergencia["config"].isin(configs_top)].copy()
    if df_plot.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for cfg in configs_top:
        d = (
            df_plot[df_plot["config"] == cfg]
            .groupby("iteracion", as_index=False)["costo_salida"]
            .mean()
            .sort_values("iteracion")
        )
        ax.plot(d["iteracion"], d["costo_salida"], marker="o", markersize=3, linewidth=2, label=cfg)

    ax.set_title(f"Random Keys multisemilla — Convergencia promedio por configuración (top {top_n})")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Costo promedio de salida")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    _guardar(fig, carpeta_salida, "05_convergencia_promedio_por_config.png")
    plt.close(fig)

def dashboard_randomkeys_estabilidad(df_resultados, df_convergencia, carpeta_salida=None):
    print(f"\nGenerando gráficos Random Keys multisemilla en: {carpeta_salida}")
    plot_rk_boxplot_gap_por_config(df_resultados, carpeta_salida)
    plot_rk_boxplot_mejora_por_config(df_resultados, carpeta_salida)
    plot_rk_boxplot_tiempo_por_config(df_resultados, carpeta_salida)
    plot_rk_boxplot_iteraciones_por_config(df_resultados, carpeta_salida)
    plot_rk_convergencia_promedio_por_config(df_convergencia, carpeta_salida)

# =============================================================
# GRÁFICOS — PERTURBACIÓN ALEATORIA
# =============================================================

def plot_perturb_gap_por_heuristica(df_resultados, carpeta_salida=None):
    """Gap % por heurística para el experimento de perturbación."""
    if df_resultados.empty or "heuristica" not in df_resultados.columns:
        return
    
    df_agg = (
        df_resultados.groupby("heuristica", as_index=False)["gap_pct"]
        .mean().sort_values("gap_pct")
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [_color_heuristica(h) for h in df_agg["heuristica"]]
    ax.bar(df_agg["heuristica"], df_agg["gap_pct"], color=colors, edgecolor="white")
    ax.set_title("Gap % promedio por heurística (con perturbación)")
    ax.set_ylabel("Gap % vs MST")
    ax.set_xlabel("Heurística")
    _annotate_bars(ax, decimals=2)
    plt.tight_layout()
    _guardar(fig, carpeta_salida, "01_gap_por_heuristica.png")
    plt.close(fig)


def plot_perturb_mejora_por_heuristica(df_resultados, carpeta_salida=None):
    """Mejora % por heurística para el experimento de perturbación."""
    if df_resultados.empty or "heuristica" not in df_resultados.columns:
        return
    
    df_agg = (
        df_resultados.groupby("heuristica", as_index=False)["mejora_pct"]
        .mean().sort_values("mejora_pct", ascending=False)
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [_color_heuristica(h) for h in df_agg["heuristica"]]
    ax.bar(df_agg["heuristica"], df_agg["mejora_pct"], color=colors, edgecolor="white")
    ax.set_title("Mejora % promedio por heurística (con perturbación)")
    ax.set_ylabel("Mejora %")
    ax.set_xlabel("Heurística")
    _annotate_bars(ax, decimals=2)
    plt.tight_layout()
    _guardar(fig, carpeta_salida, "02_mejora_por_heuristica.png")
    plt.close(fig)


def plot_perturb_tasa_exito(df_resultados, carpeta_salida=None):
    """Tasa de éxito de perturbaciones (mejoras / total perturbaciones)."""
    if df_resultados.empty:
        return
    
    df_agg = df_resultados.groupby("heuristica", as_index=False).agg({
        "n_perturbaciones": "mean",
        "n_mejoras": "mean"
    })
    df_agg["tasa_exito"] = (df_agg["n_mejoras"] / df_agg["n_perturbaciones"] * 100).fillna(0)
    df_agg = df_agg.sort_values("tasa_exito", ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [_color_heuristica(h) for h in df_agg["heuristica"]]
    ax.bar(df_agg["heuristica"], df_agg["tasa_exito"], color=colors, edgecolor="white")
    ax.set_title("Tasa de éxito de perturbaciones por heurística")
    ax.set_ylabel("% de perturbaciones que mejoraron")
    ax.set_xlabel("Heurística")
    ax.set_ylim(0, 100)
    _annotate_bars(ax, decimals=1)
    plt.tight_layout()
    _guardar(fig, carpeta_salida, "03_tasa_exito_perturbaciones.png")
    plt.close(fig)


def plot_perturb_convergencia_promedio(df_convergencia, carpeta_salida=None):
    """
    Convergencia promedio mostrando todas las fases de perturbación.
    Cada fase de perturbación se muestra como una curva separada.
    """
    if df_convergencia.empty:
        return
    
    # Agrupar por fase de perturbación
    fases = sorted(df_convergencia["fase_perturbacion"].unique())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for fase in fases:
        df_fase = df_convergencia[df_convergencia["fase_perturbacion"] == fase]
        df_plot = (
            df_fase.groupby("iteracion", as_index=False)["costo_salida"]
            .mean().sort_values("iteracion")
        )
        
        label = "Inicial" if fase == 0 else f"Perturbación {fase}"
        alpha = 1.0 if fase == 0 else 0.6
        linewidth = 2 if fase == 0 else 1.5
        
        ax.plot(df_plot["iteracion"], df_plot["costo_salida"], 
                marker="o" if fase == 0 else None,
                markersize=3, linewidth=linewidth, alpha=alpha, label=label)
    
    ax.set_title("Convergencia promedio con perturbaciones")
    ax.set_xlabel("Iteración (dentro de cada fase)")
    ax.set_ylabel("Costo promedio")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    _guardar(fig, carpeta_salida, "04_convergencia_con_perturbaciones.png")
    plt.close(fig)


def plot_perturb_evolucion_mejor_costo(df_perturbaciones, carpeta_salida=None):
    """
    Evolución del mejor costo global a través de las perturbaciones.
    Muestra cómo el mejor costo mejora (o no) con cada perturbación.
    """
    if df_perturbaciones.empty:
        return
    
    # Promediar por fase de perturbación
    df_plot = (
        df_perturbaciones.groupby("fase_perturbacion", as_index=False)
        ["costo_mejor_global"].mean().sort_values("fase_perturbacion")
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_plot["fase_perturbacion"], df_plot["costo_mejor_global"], 
            marker="o", linewidth=2, color="#E67E22", markersize=6)
    ax.set_title("Evolución del mejor costo global a través de perturbaciones")
    ax.set_xlabel("Fase de perturbación")
    ax.set_ylabel("Mejor costo global (promedio)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _guardar(fig, carpeta_salida, "05_evolucion_mejor_costo.png")
    plt.close(fig)


def plot_perturb_tiempo_por_heuristica(df_resultados, carpeta_salida=None):
    """Tiempo total de ejecución por heurística."""
    if df_resultados.empty:
        return
    
    df_agg = (
        df_resultados.groupby("heuristica", as_index=False)["tiempo_total_seg"]
        .mean().sort_values("tiempo_total_seg")
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [_color_heuristica(h) for h in df_agg["heuristica"]]
    ax.bar(df_agg["heuristica"], df_agg["tiempo_total_seg"], color=colors, edgecolor="white")
    ax.set_title("Tiempo promedio de ejecución por heurística")
    ax.set_ylabel("Tiempo (segundos)")
    ax.set_xlabel("Heurística")
    _annotate_bars(ax, decimals=2)
    plt.tight_layout()
    _guardar(fig, carpeta_salida, "06_tiempo_por_heuristica.png")
    plt.close(fig)


def plot_perturb_gap_por_rango_nodos(df_resultados, carpeta_salida=None):
    """Boxplot de gap% agrupado por rango de nodos para perturbación."""
    if df_resultados.empty or "rango_nodos" not in df_resultados.columns:
        return
    rangos = sorted(df_resultados["rango_nodos"].dropna().unique())
    pares = [(r, df_resultados.loc[df_resultados["rango_nodos"] == r, "gap_pct"].dropna().values)
             for r in rangos]
    pares = [(r, g) for r, g in pares if len(g) > 0]
    if not pares:
        return
    labels, datos = zip(*pares)
    colors = ["#AED6F1", "#85C1E9", "#5DADE2", "#2E86C1", "#1A5276"]
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(datos, labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors[:len(bp["boxes"])]):
        patch.set_facecolor(color)
    ax.set_title("Gap % según tamaño de instancia (con perturbación)")
    ax.set_ylabel("Gap % vs MST")
    ax.set_xlabel("Rango de nodos")
    _rotar_xticks(ax, 20)
    plt.tight_layout()
    _guardar(fig, carpeta_salida, "07_gap_por_rango_nodos.png")
    plt.close(fig)


def plot_perturb_mejora_por_rango_nodos(df_resultados, carpeta_salida=None):
    """Boxplot de mejora% agrupado por rango de nodos para perturbación."""
    if df_resultados.empty or "rango_nodos" not in df_resultados.columns:
        return
    rangos = sorted(df_resultados["rango_nodos"].dropna().unique())
    pares = [(r, df_resultados.loc[df_resultados["rango_nodos"] == r, "mejora_pct"].dropna().values)
             for r in rangos]
    pares = [(r, m) for r, m in pares if len(m) > 0]
    if not pares:
        return
    labels, datos = zip(*pares)
    colors = ["#ABEBC6", "#82E0AA", "#58D68D", "#2ECC71", "#239B56"]
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(datos, labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors[:len(bp["boxes"])]):
        patch.set_facecolor(color)
    ax.set_title("Mejora % según tamaño de instancia (con perturbación)")
    ax.set_ylabel("Mejora %")
    ax.set_xlabel("Rango de nodos")
    _rotar_xticks(ax, 20)
    plt.tight_layout()
    _guardar(fig, carpeta_salida, "08_mejora_por_rango_nodos.png")
    plt.close(fig)


def plot_perturb_tasa_exito_por_rango_nodos(df_resultados, carpeta_salida=None):
    """Tasa de éxito de perturbaciones por rango de nodos."""
    if df_resultados.empty or "rango_nodos" not in df_resultados.columns:
        return
    
    df_agg = df_resultados.groupby("rango_nodos", as_index=False).agg({
        "n_perturbaciones": "mean",
        "n_mejoras": "mean"
    })
    df_agg["tasa_exito"] = (df_agg["n_mejoras"] / df_agg["n_perturbaciones"] * 100).fillna(0)
    df_agg = df_agg.sort_values("rango_nodos")
    
    if df_agg.empty:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#AED6F1", "#85C1E9", "#5DADE2", "#2E86C1", "#1A5276"]
    ax.bar(df_agg["rango_nodos"], df_agg["tasa_exito"], 
           color=colors[:len(df_agg)], edgecolor="white")
    ax.set_title("Tasa de éxito de perturbaciones según tamaño de instancia")
    ax.set_ylabel("% de perturbaciones que mejoraron")
    ax.set_xlabel("Rango de nodos")
    ax.set_ylim(0, 100)
    _rotar_xticks(ax, 20)
    _annotate_bars(ax, decimals=1)
    plt.tight_layout()
    _guardar(fig, carpeta_salida, "09_tasa_exito_por_rango_nodos.png")
    plt.close(fig)


def dashboard_perturbacion(df_resultados, df_convergencia, df_perturbaciones, carpeta_salida=None):
    """Dashboard completo para el experimento de perturbación."""
    print(f"\nGenerando gráficos de perturbación en: {carpeta_salida}")
    
    # Gráficos por heurística
    plot_perturb_gap_por_heuristica(df_resultados, carpeta_salida)
    plot_perturb_mejora_por_heuristica(df_resultados, carpeta_salida)
    plot_perturb_tasa_exito(df_resultados, carpeta_salida)
    
    # Gráficos de convergencia
    plot_perturb_convergencia_promedio(df_convergencia, carpeta_salida)
    plot_perturb_evolucion_mejor_costo(df_perturbaciones, carpeta_salida)
    
    # Gráficos de tiempo
    plot_perturb_tiempo_por_heuristica(df_resultados, carpeta_salida)
    
    # Gráficos por rango de nodos
    plot_perturb_gap_por_rango_nodos(df_resultados, carpeta_salida)
    plot_perturb_mejora_por_rango_nodos(df_resultados, carpeta_salida)
    plot_perturb_tasa_exito_por_rango_nodos(df_resultados, carpeta_salida)
    
    print("✓ Gráficos de perturbación generados.\n")

# =============================================================
# COLORES PARA MÉTODOS DEL PAPER
# =============================================================

_COLORES_METODO = {
    "2OPT_BEST":  "#E74C3C",   # rojo
    "2OPT_FIRST": "#F39C12",   # naranja
    "SPLIT_M01":  "#3498DB",   # azul claro
    "SPLIT_FULL": "#1A5276",   # azul oscuro
    "VND":        "#27AE60",   # verde
}

_COLORES_PERTURBACION = {
    "DOUBLE_BRIDGE":   "#8E44AD",   # morado
    "MULTI_MOVIMIENTO": "#16A085",  # verde agua
}


def _color_metodo(metodo: str) -> str:
    return _COLORES_METODO.get(metodo, "#7F8C8D")


def _color_perturbacion(p: str) -> str:
    return _COLORES_PERTURBACION.get(p, "#7F8C8D")


# =============================================================
# DASHBOARD PAPER EXP 1: BÚSQUEDA LOCAL
# =============================================================

def plot_paper_bl_gap_por_metodo(df, carpeta=None):
    """Boxplot de gap% por método (todos los datos)."""
    if df.empty:
        return
    metodos = sorted(df["metodo"].unique())
    datos = [df.loc[df["metodo"] == m, "gap_pct"].dropna().values for m in metodos]
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(datos, labels=metodos, patch_artist=True)
    for patch, m in zip(bp["boxes"], metodos):
        patch.set_facecolor(_color_metodo(m))
    ax.set_title("Gap % por método (búsqueda local pura)")
    ax.set_ylabel("Gap % vs MST")
    ax.set_xlabel("Método")
    plt.tight_layout()
    _guardar(fig, carpeta, "01_gap_por_metodo.png")
    plt.close(fig)


def plot_paper_bl_gap_por_metodo_y_heuristica(df, carpeta=None):
    """Barras agrupadas: gap promedio por método y heurística."""
    if df.empty:
        return
    df_agg = (
        df.groupby(["metodo", "heuristica"], as_index=False)["gap_pct"]
        .mean()
    )
    pivot = df_agg.pivot(index="metodo", columns="heuristica", values="gap_pct")
    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", ax=ax, edgecolor="white")
    ax.set_title("Gap % promedio por método y heurística inicial")
    ax.set_ylabel("Gap % vs MST")
    ax.set_xlabel("Método")
    ax.legend(title="Heurística", fontsize=9)
    _rotar_xticks(ax, 0)
    plt.tight_layout()
    _guardar(fig, carpeta, "02_gap_por_metodo_y_heuristica.png")
    plt.close(fig)


def plot_paper_bl_tiempo_por_metodo_y_tamano(df, carpeta=None):
    """Tiempo promedio por método para cada rango de tamaño."""
    if df.empty:
        return
    df_agg = (
        df.groupby(["rango_nodos", "metodo"], as_index=False)["tiempo_seg"]
        .mean()
    )
    pivot = df_agg.pivot(index="rango_nodos", columns="metodo", values="tiempo_seg")
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = [_color_metodo(m) for m in pivot.columns]
    pivot.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Tiempo promedio por método según tamaño de instancia")
    ax.set_ylabel("Tiempo (segundos)")
    ax.set_xlabel("Rango de nodos")
    ax.legend(title="Método", fontsize=9)
    _rotar_xticks(ax, 20)
    plt.tight_layout()
    _guardar(fig, carpeta, "03_tiempo_por_metodo_y_tamano.png")
    plt.close(fig)


def plot_paper_bl_gap_por_tamano(df, carpeta=None):
    """Gap promedio por método y rango de nodos."""
    if df.empty:
        return
    df_agg = (
        df.groupby(["rango_nodos", "metodo"], as_index=False)["gap_pct"]
        .mean()
    )
    pivot = df_agg.pivot(index="rango_nodos", columns="metodo", values="gap_pct")
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = [_color_metodo(m) for m in pivot.columns]
    pivot.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Gap % promedio por método y tamaño de instancia")
    ax.set_ylabel("Gap % vs MST")
    ax.set_xlabel("Rango de nodos")
    ax.legend(title="Método", fontsize=9)
    _rotar_xticks(ax, 20)
    plt.tight_layout()
    _guardar(fig, carpeta, "04_gap_por_metodo_y_tamano.png")
    plt.close(fig)


def plot_paper_bl_calidad_vs_tiempo(df, carpeta=None):
    """Scatter: calidad (gap) vs tiempo, coloreado por método."""
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for metodo in df["metodo"].unique():
        sub = df[df["metodo"] == metodo]
        ax.scatter(sub["tiempo_seg"], sub["gap_pct"],
                   color=_color_metodo(metodo), label=metodo,
                   alpha=0.6, edgecolor="white", s=60)
    ax.set_xscale("log")
    ax.set_title("Calidad vs Tiempo (cada punto = 1 instancia × heurística)")
    ax.set_xlabel("Tiempo (segundos, escala log)")
    ax.set_ylabel("Gap % vs MST")
    ax.legend(title="Método", fontsize=9)
    plt.tight_layout()
    _guardar(fig, carpeta, "05_calidad_vs_tiempo.png")
    plt.close(fig)


def plot_paper_bl_razon_parada(df, carpeta=None):
    """Stacked bar: distribución de razones de parada por método."""
    if df.empty or "razon_parada" not in df.columns:
        return
    df_agg = (
        df.groupby(["metodo", "razon_parada"]).size()
        .reset_index(name="conteo")
    )
    pivot = df_agg.pivot(index="metodo", columns="razon_parada", values="conteo").fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", stacked=True, ax=ax, edgecolor="white")
    ax.set_title("Distribución de razones de parada por método")
    ax.set_ylabel("Número de corridas")
    ax.set_xlabel("Método")
    ax.legend(title="Razón", fontsize=9)
    _rotar_xticks(ax, 0)
    plt.tight_layout()
    _guardar(fig, carpeta, "06_razon_parada.png")
    plt.close(fig)


def dashboard_paper_busqueda_local(df_resultados, df_resumen_metodo, df_por_tamano,
                                   df_convergencia, carpeta_salida=None):
    """Dashboard completo del Experimento 1 del paper."""
    print(f"\nGenerando gráficos paper Exp 1 en: {carpeta_salida}")
    plot_paper_bl_gap_por_metodo(df_resultados, carpeta_salida)
    plot_paper_bl_gap_por_metodo_y_heuristica(df_resultados, carpeta_salida)
    plot_paper_bl_tiempo_por_metodo_y_tamano(df_resultados, carpeta_salida)
    plot_paper_bl_gap_por_tamano(df_resultados, carpeta_salida)
    plot_paper_bl_calidad_vs_tiempo(df_resultados, carpeta_salida)
    plot_paper_bl_razon_parada(df_resultados, carpeta_salida)
    print("✓ Gráficos paper Exp 1 generados.\n")


# =============================================================
# DASHBOARD PAPER EXP 2: ITERATED LOCAL SEARCH (ILS)
# =============================================================

def plot_paper_ils_gap_por_metodo_y_perturbacion(df, carpeta=None):
    """Barras agrupadas: gap promedio por método y perturbación."""
    if df.empty:
        return
    df_agg = (
        df.groupby(["metodo", "perturbacion"], as_index=False)["gap_pct"]
        .mean()
    )
    pivot = df_agg.pivot(index="metodo", columns="perturbacion", values="gap_pct")
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = [_color_perturbacion(p) for p in pivot.columns]
    pivot.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Gap % promedio por método y perturbación (ILS)")
    ax.set_ylabel("Gap % vs MST")
    ax.set_xlabel("Método")
    ax.legend(title="Perturbación", fontsize=9)
    _rotar_xticks(ax, 0)
    plt.tight_layout()
    _guardar(fig, carpeta, "01_gap_por_metodo_y_perturbacion.png")
    plt.close(fig)


def plot_paper_ils_mejora_perturb(df, carpeta=None):
    """Mejora % aportada por la perturbación (vs solo BL inicial)."""
    if df.empty:
        return
    df_agg = (
        df.groupby(["metodo", "perturbacion"], as_index=False)["mejora_perturb_pct"]
        .mean()
    )
    pivot = df_agg.pivot(index="metodo", columns="perturbacion", values="mejora_perturb_pct")
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = [_color_perturbacion(p) for p in pivot.columns]
    pivot.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Mejora aportada por la perturbación (vs BL inicial)")
    ax.set_ylabel("Mejora % adicional gracias al ILS")
    ax.set_xlabel("Método")
    ax.legend(title="Perturbación", fontsize=9)
    _rotar_xticks(ax, 0)
    plt.tight_layout()
    _guardar(fig, carpeta, "02_mejora_aportada_por_perturbacion.png")
    plt.close(fig)


def plot_paper_ils_tasa_exito(df, carpeta=None):
    """Tasa de éxito de las perturbaciones (cuando lograron mejorar)."""
    if df.empty:
        return
    df_agg = (
        df.groupby(["metodo", "perturbacion"], as_index=False)["tasa_exito_pct"]
        .mean()
    )
    pivot = df_agg.pivot(index="metodo", columns="perturbacion", values="tasa_exito_pct")
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = [_color_perturbacion(p) for p in pivot.columns]
    pivot.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Tasa de éxito de perturbaciones (% que mejoraron)")
    ax.set_ylabel("% perturbaciones que mejoraron")
    ax.set_xlabel("Método")
    ax.set_ylim(0, 100)
    ax.legend(title="Perturbación", fontsize=9)
    _rotar_xticks(ax, 0)
    plt.tight_layout()
    _guardar(fig, carpeta, "03_tasa_exito_perturbaciones.png")
    plt.close(fig)


def plot_paper_ils_gap_por_tamano(df, carpeta=None):
    """Gap por método y rango de nodos (separado por perturbación)."""
    if df.empty:
        return
    perturbaciones = sorted(df["perturbacion"].unique())
    fig, axes = plt.subplots(1, len(perturbaciones), figsize=(7 * len(perturbaciones), 6),
                             sharey=True)
    if len(perturbaciones) == 1:
        axes = [axes]
    for ax, perturbacion in zip(axes, perturbaciones):
        sub = df[df["perturbacion"] == perturbacion]
        df_agg = (
            sub.groupby(["rango_nodos", "metodo"], as_index=False)["gap_pct"]
            .mean()
        )
        pivot = df_agg.pivot(index="rango_nodos", columns="metodo", values="gap_pct")
        colors = [_color_metodo(m) for m in pivot.columns]
        pivot.plot(kind="bar", ax=ax, color=colors, edgecolor="white", legend=(ax == axes[0]))
        ax.set_title(f"Perturbación: {perturbacion}")
        ax.set_xlabel("Rango de nodos")
        if ax == axes[0]:
            ax.set_ylabel("Gap % vs MST")
            ax.legend(title="Método", fontsize=8)
        _rotar_xticks(ax, 20)
    fig.suptitle("Gap % por método y tamaño (ILS)")
    plt.tight_layout()
    _guardar(fig, carpeta, "04_gap_por_tamano_y_perturbacion.png")
    plt.close(fig)


def plot_paper_ils_calidad_vs_tiempo(df, carpeta=None):
    """Scatter: calidad vs tiempo, separado por perturbación."""
    if df.empty:
        return
    perturbaciones = sorted(df["perturbacion"].unique())
    fig, axes = plt.subplots(1, len(perturbaciones), figsize=(7 * len(perturbaciones), 6),
                             sharey=True)
    if len(perturbaciones) == 1:
        axes = [axes]
    for ax, perturbacion in zip(axes, perturbaciones):
        sub = df[df["perturbacion"] == perturbacion]
        for metodo in sub["metodo"].unique():
            d = sub[sub["metodo"] == metodo]
            ax.scatter(d["tiempo_total_seg"], d["gap_pct"],
                       color=_color_metodo(metodo), label=metodo,
                       alpha=0.6, edgecolor="white", s=60)
        ax.set_xscale("log")
        ax.set_title(f"Perturbación: {perturbacion}")
        ax.set_xlabel("Tiempo total (s, log)")
        if ax == axes[0]:
            ax.set_ylabel("Gap % vs MST")
            ax.legend(title="Método", fontsize=8)
    fig.suptitle("Calidad vs Tiempo (ILS)")
    plt.tight_layout()
    _guardar(fig, carpeta, "05_calidad_vs_tiempo.png")
    plt.close(fig)


def plot_paper_ils_evolucion_perturbaciones(df_perturbaciones, carpeta=None):
    """Evolución del costo a lo largo de las perturbaciones (promediado)."""
    if df_perturbaciones is None or df_perturbaciones.empty:
        return
    df_agg = (
        df_perturbaciones.groupby(["metodo", "perturbacion", "iteracion_perturb"],
                                   as_index=False)["costo_post_bl"].mean()
    )
    perturbaciones = sorted(df_agg["perturbacion"].unique())
    fig, axes = plt.subplots(1, len(perturbaciones), figsize=(7 * len(perturbaciones), 6))
    if len(perturbaciones) == 1:
        axes = [axes]
    for ax, perturbacion in zip(axes, perturbaciones):
        sub = df_agg[df_agg["perturbacion"] == perturbacion]
        for metodo in sub["metodo"].unique():
            d = sub[sub["metodo"] == metodo].sort_values("iteracion_perturb")
            ax.plot(d["iteracion_perturb"], d["costo_post_bl"],
                    marker="o", color=_color_metodo(metodo),
                    label=metodo, linewidth=2, markersize=4)
        ax.set_title(f"Perturbación: {perturbacion}")
        ax.set_xlabel("Iteración de perturbación")
        ax.set_ylabel("Costo promedio post-BL")
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.legend(title="Método", fontsize=8)
    fig.suptitle("Evolución del costo a lo largo de las perturbaciones")
    plt.tight_layout()
    _guardar(fig, carpeta, "06_evolucion_perturbaciones.png")
    plt.close(fig)


def dashboard_paper_ils(df_resultados, df_resumen, df_por_tamano,
                        df_perturbaciones, carpeta_salida=None):
    """Dashboard completo del Experimento 2 del paper."""
    print(f"\nGenerando gráficos paper Exp 2 (ILS) en: {carpeta_salida}")
    plot_paper_ils_gap_por_metodo_y_perturbacion(df_resultados, carpeta_salida)
    plot_paper_ils_mejora_perturb(df_resultados, carpeta_salida)
    plot_paper_ils_tasa_exito(df_resultados, carpeta_salida)
    plot_paper_ils_gap_por_tamano(df_resultados, carpeta_salida)
    plot_paper_ils_calidad_vs_tiempo(df_resultados, carpeta_salida)
    plot_paper_ils_evolucion_perturbaciones(df_perturbaciones, carpeta_salida)
    print("✓ Gráficos paper Exp 2 generados.\n")


# =============================================================
# COLORES BRKGA
# =============================================================

_COLORES_DECODER = {
    "DECODER_SORT":  "#E74C3C",   # rojo (BRKGA puro)
    "DECODER_SPLIT": "#27AE60",   # verde (BRKGA híbrido con Split)
}


def _color_decoder(d: str) -> str:
    return _COLORES_DECODER.get(d, "#7F8C8D")


# =============================================================
# DASHBOARD PAPER EXP 3: BRKGA
# =============================================================

def plot_paper_brkga_gap_global(df, carpeta=None):
    """Boxplot de gap por decoder (todas las corridas, todas las instancias)."""
    if df.empty:
        return
    decoders = sorted(df["decoder"].unique())
    datos = [df.loc[df["decoder"] == d, "gap_pct"].dropna().values for d in decoders]
    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(datos, labels=decoders, patch_artist=True)
    for patch, d in zip(bp["boxes"], decoders):
        patch.set_facecolor(_color_decoder(d))
    ax.set_title("Gap % por decoder (todas las corridas)")
    ax.set_ylabel("Gap % vs MST")
    ax.set_xlabel("Decoder")
    plt.tight_layout()
    _guardar(fig, carpeta, "01_gap_global_por_decoder.png")
    plt.close(fig)


def plot_paper_brkga_gap_por_tamano(df, carpeta=None):
    """Gap promedio por decoder y rango de nodos (con error bars = std)."""
    if df.empty:
        return
    df_agg = (
        df.groupby(["rango_nodos", "decoder"], as_index=False)
        .agg(gap_mean=("gap_pct", "mean"),
             gap_std=("gap_pct", "std"))
    )
    pivot_mean = df_agg.pivot(index="rango_nodos", columns="decoder", values="gap_mean")
    pivot_std = df_agg.pivot(index="rango_nodos", columns="decoder", values="gap_std").fillna(0)
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = [_color_decoder(d) for d in pivot_mean.columns]
    pivot_mean.plot(kind="bar", ax=ax, color=colors, edgecolor="white",
                    yerr=pivot_std, capsize=4)
    ax.set_title("Gap % promedio por decoder y tamaño (± desv. estándar)")
    ax.set_ylabel("Gap % vs MST")
    ax.set_xlabel("Rango de nodos")
    ax.legend(title="Decoder", fontsize=9)
    _rotar_xticks(ax, 20)
    plt.tight_layout()
    _guardar(fig, carpeta, "02_gap_por_decoder_y_tamano.png")
    plt.close(fig)


def plot_paper_brkga_variabilidad_por_instancia(df_resumen_instancia, carpeta=None):
    """
    Variabilidad del gap entre semillas para cada instancia.
    Muestra std del gap en función del decoder.
    """
    if df_resumen_instancia is None or df_resumen_instancia.empty:
        return
    decoders = sorted(df_resumen_instancia["decoder"].unique())
    datos = [df_resumen_instancia.loc[df_resumen_instancia["decoder"] == d, "gap_std"]
             .dropna().values for d in decoders]
    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(datos, labels=decoders, patch_artist=True)
    for patch, d in zip(bp["boxes"], decoders):
        patch.set_facecolor(_color_decoder(d))
    ax.set_title("Variabilidad del gap entre semillas (std por instancia)")
    ax.set_ylabel("Desv. estándar del gap %")
    ax.set_xlabel("Decoder")
    plt.tight_layout()
    _guardar(fig, carpeta, "03_variabilidad_por_decoder.png")
    plt.close(fig)


def plot_paper_brkga_tiempo_por_tamano(df, carpeta=None):
    """Tiempo promedio por decoder y rango de nodos."""
    if df.empty:
        return
    df_agg = (
        df.groupby(["rango_nodos", "decoder"], as_index=False)["tiempo_seg"]
        .mean()
    )
    pivot = df_agg.pivot(index="rango_nodos", columns="decoder", values="tiempo_seg")
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = [_color_decoder(d) for d in pivot.columns]
    pivot.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Tiempo promedio por decoder según tamaño")
    ax.set_ylabel("Tiempo (segundos)")
    ax.set_xlabel("Rango de nodos")
    ax.legend(title="Decoder", fontsize=9)
    _rotar_xticks(ax, 20)
    plt.tight_layout()
    _guardar(fig, carpeta, "04_tiempo_por_decoder_y_tamano.png")
    plt.close(fig)


def plot_paper_brkga_evolucion(df_evolucion, carpeta=None):
    """Convergencia promedio del mejor costo a lo largo de las generaciones."""
    if df_evolucion is None or df_evolucion.empty:
        return
    df_agg = (
        df_evolucion.groupby(["decoder", "generacion"], as_index=False)
        ["mejor_costo"].mean()
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    for decoder in df_agg["decoder"].unique():
        sub = df_agg[df_agg["decoder"] == decoder].sort_values("generacion")
        ax.plot(sub["generacion"], sub["mejor_costo"],
                color=_color_decoder(decoder), linewidth=2, label=decoder)
    ax.set_title("Convergencia promedio del mejor costo (BRKGA)")
    ax.set_xlabel("Generación")
    ax.set_ylabel("Mejor costo promedio (sobre semillas e instancias)")
    ax.legend(title="Decoder", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _guardar(fig, carpeta, "05_evolucion_promedio.png")
    plt.close(fig)


def plot_paper_brkga_calidad_vs_tiempo(df, carpeta=None):
    """Scatter: calidad vs tiempo, separando decoders."""
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for decoder in df["decoder"].unique():
        sub = df[df["decoder"] == decoder]
        ax.scatter(sub["tiempo_seg"], sub["gap_pct"],
                   color=_color_decoder(decoder), label=decoder,
                   alpha=0.5, edgecolor="white", s=50)
    ax.set_xscale("log")
    ax.set_title("Calidad vs Tiempo (cada punto = 1 corrida)")
    ax.set_xlabel("Tiempo (segundos, escala log)")
    ax.set_ylabel("Gap % vs MST")
    ax.legend(title="Decoder", fontsize=9)
    plt.tight_layout()
    _guardar(fig, carpeta, "06_calidad_vs_tiempo.png")
    plt.close(fig)


def dashboard_paper_brkga(df_resultados, df_resumen_instancia, df_por_tamano,
                          df_evolucion, carpeta_salida=None):
    """Dashboard completo del Experimento 3 del paper."""
    print(f"\nGenerando gráficos paper Exp 3 (BRKGA) en: {carpeta_salida}")
    plot_paper_brkga_gap_global(df_resultados, carpeta_salida)
    plot_paper_brkga_gap_por_tamano(df_resultados, carpeta_salida)
    plot_paper_brkga_variabilidad_por_instancia(df_resumen_instancia, carpeta_salida)
    plot_paper_brkga_tiempo_por_tamano(df_resultados, carpeta_salida)
    plot_paper_brkga_evolucion(df_evolucion, carpeta_salida)
    plot_paper_brkga_calidad_vs_tiempo(df_resultados, carpeta_salida)
    print("✓ Gráficos paper Exp 3 generados.\n")