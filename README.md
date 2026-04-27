# TSP Split: Operador Split como Búsqueda Local para el Problema del Vendedor Viajero

Este repositorio contiene la implementación y los experimentos asociados a un estudio sobre el operador **Split** como método de búsqueda local para el problema del vendedor viajero (TSP). El operador Split, basado en programación dinámica, considera 5 movimientos (M0–M4) sobre los bloques de la ruta, y se compara contra alternativas clásicas como 2-opt (Best y First Improvement), VND (Variable Neighborhood Descent) y BRKGA (Biased Random-Key Genetic Algorithm).

## 📖 Tabla de contenidos

- [Estructura del proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Uso rápido](#uso-rápido)
- [Experimentos del paper](#experimentos-del-paper)
- [Configuración](#configuración)
- [Resultados](#resultados)

---

## Estructura del proyecto

```
tsp_project/
├── tsp/                          # Paquete principal
│   ├── __init__.py
│   ├── io.py                     # Lectura de archivos .tsp
│   ├── distancias.py             # Métricas y MST como cota inferior
│   ├── heuristicas.py            # NNH, INSERCION, RANDOM_KEYS
│   ├── split.py                  # Algoritmo Split DP (M0-M4)
│   ├── dos_opt.py                # 2-opt Best y First Improvement
│   ├── busqueda_local.py         # Búsqueda local iterativa
│   ├── vnd.py                    # Variable Neighborhood Descent
│   ├── perturbaciones.py         # Double-Bridge y Multi-Movimiento
│   ├── brkga.py                  # BRKGA puro y BRKGA + Split
│   ├── experimentos.py           # Experimentos del paper
│   ├── visualizacion.py          # Generación de gráficos
│   ├── config_parser.py          # Lector de YAMLs
│   └── defaults.py               # Valores por defecto
├── configs/                      # Archivos de configuración YAML
│   ├── exp1_busqueda_local_paper.yaml
│   ├── exp2_ils_paper.yaml
│   ├── exp3_brkga_paper.yaml
│   └── paper_completo.yaml       # Maestro: ejecuta los 3 experimentos
├── datos/                        # Instancias TSPLIB (.tsp)
├── resultados/                   # Salidas de experimentos (no versionado)
├── run.py                        # Entry point
├── requirements.txt              # Dependencias Python
└── README.md
```

---

## Instalación

### Requisitos previos

- Python 3.10 o superior
- pip

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/<tu-usuario>/<tu-repo>.git
cd <tu-repo>

# 2. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate          # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

### Datos

Las instancias TSP usadas son del repositorio público **TSPLIB**. Coloca los archivos `.tsp` (formato `EUC_2D`) en la carpeta `datos/`.

---

## Uso rápido

Cada experimento se configura mediante un archivo YAML y se ejecuta con un único comando:

```bash
python run.py configs/<nombre_del_yaml>.yaml
```

### Ejemplos

```bash
# Ejecutar solo el Experimento 1 del paper (búsqueda local comparativa)
python run.py configs/exp1_busqueda_local_paper.yaml

# Ejecutar solo el Experimento 3 (BRKGA)
python run.py configs/exp3_brkga_paper.yaml

# Ejecutar los 3 experimentos del paper en cadena
python run.py configs/paper_completo.yaml
```

Cada experimento genera:
- Un archivo Excel con varios sheets (resultados, resumen, etc.)
- Una carpeta `graficos/` con visualizaciones en PNG

Los resultados se guardan en `resultados/<nombre_experimento>/`.

---

## Experimentos del paper

### Experimento 1: Búsqueda Local Comparativa

Compara cuatro métodos de búsqueda local pura partiendo de la misma solución inicial:

- **2-opt Best Improvement** — explora todos los vecinos y escoge el mejor
- **2-opt First Improvement** — acepta el primer vecino que mejore
- **Split M0+M1** — Split DP usando solo M0 y M1 (equivalente conceptual a 2-opt)
- **Split M0–M4** — Split DP con los 5 movimientos completos

Los cuatro métodos parten de la misma solución inicial (NNH, Insertion o Random Keys).

### Experimento 2: Iterated Local Search (ILS)

Compara cinco métodos como búsqueda local dentro de un esquema ILS:

- 2-opt Best, 2-opt First, Split M0+M1, Split M0–M4 y **VND** (M1→M2→M3→M4)

Cada método se prueba con dos perturbaciones:

- **Double-Bridge** — perturbación clásica de la literatura, irreversible para 2-opt
- **Multi-Movimiento Disjunto** — perturbación propia que aplica M1–M4 sobre segmentos disjuntos

### Experimento 3: BRKGA puro vs BRKGA + Split

Compara dos decoders dentro del esquema BRKGA:

- **DECODER_SORT** — decoder clásico que ordena nodos por sus random keys
- **DECODER_SPLIT** — decoder híbrido que tras ordenar aplica Split DP como local search

Cada combinación se ejecuta con 10 semillas independientes para evaluar la variabilidad estadística.

---

## Configuración

Cada YAML define el experimento. Ejemplo de `exp1_busqueda_local_paper.yaml`:

```yaml
modo: paper_busqueda_local

heuristicas:
  - NNH
  - INSERCION
  - RANDOM_KEYS

metodos:
  - 2OPT_BEST
  - 2OPT_FIRST
  - SPLIT_M01
  - SPLIT_FULL

max_iter: 100
tolerancia: 1.0e-9
timeout: 60.0
```

Todos los parámetros tienen defaults razonables; ver `tsp/defaults.py`.

---

## Resultados

Cada experimento genera un Excel con varias hojas:

- `resultados` — fila por (instancia × método × heurística)
- `resumen` — agregados estadísticos
- `por_tamano` — agregados por rango de número de nodos
- `convergencia` / `evolucion` — datos iteración a iteración

Los gráficos PNG cubren:

- Boxplots de gap %
- Comparaciones por método y heurística
- Convergencia
- Calidad vs tiempo (escala log)
- Variabilidad entre semillas (Exp 3)

---

## Cita


```
@article{<tu_apellido>2025tsp,
  title  = {<Título de tu paper>},
  author = {<Tus autores>},
  year   = {2025},
  ...
}
```

---

## Licencia

MIT License (ver `LICENSE`).
