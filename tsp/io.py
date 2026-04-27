# tsp/io.py
import os


def leer_archivo_tsp(ruta_archivo: str) -> list[tuple]:
    """
    Lee un archivo .tsp en formato NODE_COORD_SECTION.
    Devuelve lista de tuplas (nodo_id, x, y).
    """
    with open(ruta_archivo, "r") as f:
        lines = f.readlines()

    nodo_coord_section = False
    nodos = []

    for line in lines:
        if "NODE_COORD_SECTION" in line:
            nodo_coord_section = True
            continue
        if "EOF" in line:
            nodo_coord_section = False

        if nodo_coord_section:
            parts = line.split()
            if len(parts) >= 3:
                nodo_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                nodos.append((nodo_id, x, y))

    return nodos


def listar_tsp_en_carpeta(carpeta: str) -> list[str]:
    """
    Devuelve lista ordenada de rutas absolutas de archivos .tsp en la carpeta.
    """
    archivos = []
    for archivo in os.listdir(carpeta):
        if archivo.lower().endswith(".tsp"):
            archivos.append(os.path.join(carpeta, archivo))
    archivos.sort()
    return archivos
