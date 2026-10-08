# Redes neuronales en Python: clasificación, regresión y series de tiempo

**Alumno:** Kevin del Jesús González Maas · **Materia:** Sistemas Inteligentes ·
**Docente:** Justino Ramírez Ortegón · Facultad de Ingeniería, Universidad Autónoma de Campeche

Tres ejemplos con redes neuronales (perceptrón multicapa, scikit-learn) sobre datasets reales
que cada script descarga automáticamente desde GitHub.

| Script | Tipo | Problema | Dataset |
|---|---|---|---|
| `ej1_clasificacion_churn.py` | Pattern recognition | Fuga de clientes (comercial) | IBM Telco Customer Churn (7,043) |
| `ej2_regresion_concreto.py` | Fitting | Resistencia del concreto, MPa (industrial) | UCI Concrete Compressive Strength (1,030) |
| `ej3_series_tiempo_pm25.py` | Series de tiempo (NARX) | PM2.5 a 6 h + alerta (industrial/ambiental) | UCI Beijing PM2.5 (43,824 horas) |

## Instrucciones de uso

Requisitos: Python 3.9+ y conexión a internet (solo la primera vez, para descargar los datos).

```bash
pip install -r requirements.txt
python ej1_clasificacion_churn.py   # ~1-2 min
python ej2_regresion_concreto.py   # ~4-5 min (validación cruzada)
python ej3_series_tiempo_pm25.py   # ~6-8 min
```

Cada script imprime en consola la carga de datos, el ajuste de hiperparámetros, el entrenamiento y
las métricas en el conjunto de prueba, y guarda sus gráficas en `figuras/`.
Los datos descargados quedan en `datos/`. Todo es reproducible (semilla 42).

Para regenerar las capturas del reporte: `python make_capturas.py` (usa los archivos de `logs/`).

## Ejecución en línea (Google Colab, sin instalar nada)

1. Entrar a https://colab.research.google.com → *Archivo → Subir cuaderno*.
2. Elegir `RNA_Practica_Colab.ipynb` y ejecutar *Entorno de ejecución → Ejecutar todo*.
   El cuaderno trae el código completo, instala lo necesario y muestra las figuras.

## Resultados obtenidos (conjunto de prueba)

- Ejemplo 1: AUC 0.843 · exactitud 79.3 % · F1 0.634 (la regresión logística de referencia da AUC 0.845).
- Ejemplo 2: R² 0.926 · RMSE 4.40 MPa (regresión lineal: 9.66 MPa).
- Ejemplo 3 (6 h): RMSE 44.8 µg/m³ (persistencia 53.6; ARX lineal 47.2) · alerta: AUC 0.941, F1 0.684.

## Origen del código

Generado con IA generativa (Claude, de Anthropic) a partir de los prompts descritos en el reporte,
revisado, ejecutado y corregido durante la sesión. Datasets: ver bibliografía del reporte.
