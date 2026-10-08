"""
comun.py - Utilidades compartidas por los tres ejemplos.
Autor: Kevin del Jesús González Maas | Sistemas Inteligentes | FI-UAC
"""
import os
import urllib.request
import matplotlib
matplotlib.use("Agg")                      # sin ventana: guarda directo a PNG
import matplotlib.pyplot as plt

RAIZ = os.path.dirname(os.path.abspath(__file__))
DIR_DATOS = os.path.join(RAIZ, "datos")
DIR_FIGS = os.path.join(RAIZ, "figuras")
SEMILLA = 42

# Paleta de ingeniería (sobria, apta para impresión)
AZUL, ROJO, VERDE, NARANJA, GRIS, NEGRO = ("#1F4E79", "#C0392B", "#2E7D32",
                                           "#E67E22", "#7F8C8D", "#212121")


def estilo_figuras():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "font.family": "DejaVu Sans",
        "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold",
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
        "legend.frameon": False, "figure.facecolor": "white",
    })


def descargar(url, nombre):
    """Descarga el dataset a ./datos si aún no existe (caché local)."""
    os.makedirs(DIR_DATOS, exist_ok=True)
    ruta = os.path.join(DIR_DATOS, nombre)
    if not os.path.exists(ruta):
        print(f"[datos] Descargando {nombre} ...")
        urllib.request.urlretrieve(url, ruta)
    return ruta


def guardar(fig, nombre):
    os.makedirs(DIR_FIGS, exist_ok=True)
    ruta = os.path.join(DIR_FIGS, nombre)
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    print(f"[figura] {os.path.relpath(ruta, RAIZ)}")
    return ruta


def encabezado(titulo):
    linea = "=" * 72
    print(f"\n{linea}\n{titulo}\n{linea}")
