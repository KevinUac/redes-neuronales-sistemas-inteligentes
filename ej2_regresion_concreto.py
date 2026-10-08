"""
EJEMPLO 2 - RED NEURONAL TIPO FITTING (REGRESION)
Problema industrial: estimar la resistencia a compresion del concreto (MPa) a partir de su
dosificacion y edad de curado, evitando ensayos destructivos prolongados.
Dataset : Concrete Compressive Strength (UCI, I-Cheng Yeh, 1998) - 1,030 mezclas, 8 variables.
Modelo  : Perceptron multicapa (MLPRegressor, scikit-learn) con ajuste de hiperparametros.
Autor   : Kevin del Jesus Gonzalez Maas - Sistemas Inteligentes - FI-UAC
"""
import itertools, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import comun as c

warnings.filterwarnings('ignore')   # avisos de convergencia de L-BFGS (esperados)

URL = ("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/"
       "master/concrete.csv")
c.estilo_figuras()
ETIQ = {"cement": "Cemento", "slag": "Escoria", "ash": "Ceniza volante", "water": "Agua",
        "superplastic": "Superplastificante", "coarseagg": "Agregado grueso",
        "fineagg": "Agregado fino", "age": "Edad (días)"}

# ---------------------------------------------------------------- 1. DATOS
c.encabezado("1. CARGA Y ANALISIS EXPLORATORIO")
df = pd.read_csv(c.descargar(URL, "concreto.csv"))
print(f"Mezclas: {len(df)} | Variables de entrada: 8 | Variable objetivo: strength (MPa)")
print(f"Datos faltantes: {int(df.isna().sum().sum())}")
print(df.describe().loc[["min", "mean", "max"]].T.round(2).to_string())
X, y = df.drop(columns="strength").values, df["strength"].values
nombres = list(df.columns[:-1])

# 15 % de prueba reservado y NO se toca hasta el final; el 85 % restante es "desarrollo".
X_dev, X_te, y_dev, y_te = train_test_split(X, y, test_size=0.15, random_state=c.SEMILLA)
print(f"\nDesarrollo (entrenamiento + validacion cruzada): {len(y_dev)} | Prueba (reservado): {len(y_te)}")

def fabricar(Xa, ya):
    """Normaliza con estadisticas del conjunto dado -> (escalador_x, escalador_y)"""
    return StandardScaler().fit(Xa), StandardScaler().fit(ya.reshape(-1, 1))

# ------------------------------------------------- 2. AJUSTE DE HIPERPARAMETROS
c.encabezado("2. AJUSTE DE HIPERPARAMETROS (validacion cruzada de 5 pliegues, criterio: RMSE)")
rejilla = {"hidden_layer_sizes": [(16,), (32,), (64,), (32, 16), (64, 32)],
           "activation": ["tanh", "relu"],
           "alpha": [1e-2, 1e-1, 1.0]}
kf = KFold(n_splits=5, shuffle=True, random_state=c.SEMILLA)
res, t0 = [], time.time()
for h, act, a in itertools.product(*rejilla.values()):
    rm = []
    for i_tr, i_va in kf.split(X_dev):
        ex_, ey_ = fabricar(X_dev[i_tr], y_dev[i_tr])
        m = MLPRegressor(hidden_layer_sizes=h, activation=act, alpha=a, solver="lbfgs",
                         max_iter=1500, random_state=c.SEMILLA)
        m.fit(ex_.transform(X_dev[i_tr]), ey_.transform(y_dev[i_tr].reshape(-1, 1)).ravel())
        p = ey_.inverse_transform(m.predict(ex_.transform(X_dev[i_va])).reshape(-1, 1)).ravel()
        rm.append(np.sqrt(mean_squared_error(y_dev[i_va], p)))
    res.append((np.mean(rm), np.std(rm), h, act, a))
res.sort(key=lambda r: r[0])
print(f"Se evaluaron {len(res)} configuraciones x 5 pliegues = {len(res)*5} entrenamientos ({time.time()-t0:.0f} s)")
print("Top 6 configuraciones:")
for rm, sd, h, act, a in res[:6]:
    print(f"  capas={str(h):<9} activ={act:<5} alpha={a:<5} RMSE_CV = {rm:.3f} +/- {sd:.3f} MPa")
print(f"Peor configuracion evaluada: RMSE_CV = {res[-1][0]:.3f} MPa")
cv_rmse, cv_sd, h_opt, act_opt, a_opt = res[0]
print(f"\n>> Mejor configuracion: capas={h_opt}, activacion={act_opt}, alpha={a_opt}")

# ------------------------------------------------------ 3. ENTRENAMIENTO FINAL
c.encabezado("3. ENTRENAMIENTO FINAL")
ex, ey = fabricar(X_dev, y_dev)
s = lambda a: ex.transform(a)
t = lambda v: ey.transform(v.reshape(-1, 1)).ravel()
inv = lambda v: ey.inverse_transform(np.asarray(v).reshape(-1, 1)).ravel()

# 3a. Curva de aprendizaje (Adam, particion 70/15 del total) para visualizar convergencia
X_a, X_v, y_a, y_v = train_test_split(X_dev, y_dev, test_size=0.1765, random_state=c.SEMILLA)
exa, eya = fabricar(X_a, y_a)
curva = MLPRegressor(hidden_layer_sizes=h_opt, activation=act_opt, alpha=a_opt, solver="adam",
                     learning_rate_init=3e-3, max_iter=1, warm_start=True, random_state=c.SEMILLA)
EPOCAS = 1500
perd_tr, perd_va = [], []
tt = lambda v: eya.transform(v.reshape(-1, 1)).ravel()
for ep in range(EPOCAS):
    curva.fit(exa.transform(X_a), tt(y_a))
    perd_tr.append(mean_squared_error(tt(y_a), curva.predict(exa.transform(X_a))))
    perd_va.append(mean_squared_error(tt(y_v), curva.predict(exa.transform(X_v))))
mejor_ep = int(np.argmin(perd_va)) + 1
print(f"Curva de aprendizaje (Adam, {len(y_a)} entrenamiento / {len(y_v)} validacion): "
      f"minimo de validacion en la epoca {mejor_ep} de {EPOCAS}")

# 3b. Comite (ensamble) de 5 redes con distintas inicializaciones, entrenado con todo "desarrollo"
SEMILLAS = [c.SEMILLA + k for k in range(5)]
comite = [MLPRegressor(hidden_layer_sizes=h_opt, activation=act_opt, alpha=a_opt, solver="lbfgs",
                       max_iter=3000, random_state=sd).fit(s(X_dev), t(y_dev)) for sd in SEMILLAS]
predecir = lambda Xn: inv(np.mean([m.predict(s(Xn)) for m in comite], axis=0))
red = comite[0]
print(f"Arquitectura de cada red: 8 -> {' -> '.join(map(str, h_opt))} -> 1 (lineal) | oculta: {act_opt}")
print(f"Parametros entrenables por red: {sum(w.size for w in red.coefs_) + sum(b.size for b in red.intercepts_)} "
      f"| redes en el comite: {len(comite)}")

# ------------------------------------------------------------ 4. EVALUACION
c.encabezado("4. EVALUACION")
lin = LinearRegression().fit(X_dev, y_dev)
def metricas(nombre, yr, yp):
    r = np.corrcoef(yr, yp)[0, 1]
    return (nombre, r, r2_score(yr, yp), np.sqrt(mean_squared_error(yr, yp)),
            mean_absolute_error(yr, yp), 100 * np.mean(np.abs((yr - yp) / yr)))
p_dev, p_te = predecir(X_dev), predecir(X_te)
p_una = inv(red.predict(s(X_te)))
tab = pd.DataFrame([metricas("Comite MLP - desarrollo", y_dev, p_dev),
                    metricas("Una sola red - prueba", y_te, p_una),
                    metricas("Comite MLP - PRUEBA", y_te, p_te),
                    metricas("Regresion lineal - prueba", y_te, lin.predict(X_te))],
                   columns=["Conjunto", "R", "R2", "RMSE (MPa)", "MAE (MPa)", "MAPE (%)"])
print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\nRMSE en validacion cruzada (5 pliegues, una red): {cv_rmse:.3f} +/- {cv_sd:.3f} MPa")
err = y_te - p_te
print(f"Error en prueba: media = {err.mean():+.3f} MPa | desv. estandar = {err.std():.3f} MPa")
print(f"El 90 % de las predicciones cae a menos de {np.percentile(np.abs(err), 90):.2f} MPa del valor real")
print(f"Mejora del RMSE respecto a la regresion lineal: {100*(1-tab.iloc[2,3]/tab.iloc[3,3]):.1f} %")

# Importancia por permutacion expresada en MPa (aumento del RMSE al desordenar cada variable)
rng = np.random.RandomState(c.SEMILLA)
rmse0 = np.sqrt(mean_squared_error(y_te, p_te))
imp_m, imp_s = np.zeros(8), np.zeros(8)
for j in range(8):
    d = []
    for _ in range(30):
        Xp = X_te.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
        d.append(np.sqrt(mean_squared_error(y_te, predecir(Xp))) - rmse0)
    imp_m[j], imp_s[j] = np.mean(d), np.std(d)
orden = np.argsort(imp_m)[::-1]
print("\nImportancia de cada variable (aumento del RMSE al permutarla, en MPa):")
for i in orden:
    print(f"  {ETIQ[nombres[i]]:<20} +{imp_m[i]:.2f} MPa")

ej = pd.DataFrame([[350, 0, 0, 180, 0, 1000, 800, 28]], columns=nombres)
print(f"\nPrediccion de ejemplo -> cemento 350, agua 180, edad 28 d: {predecir(ej.values)[0]:.2f} MPa")

# ------------------------------------------------------------------ 5. FIGURAS
fig, ax = plt.subplots(2, 3, figsize=(16, 9.2))
fig.suptitle("Ejemplo 2 · Fitting: resistencia a la compresión del concreto (MPa)",
             fontsize=14, fontweight="bold", y=1.0)
a = ax[0, 0]
suave = lambda v, w=25: np.convolve(v, np.ones(w) / w, mode="valid")   # media movil (quita ruido de mini-lotes)
for v, col, nom in [(perd_tr, c.AZUL, "Entrenamiento"), (perd_va, c.NARANJA, "Validación")]:
    a.plot(range(1, EPOCAS + 1), v, color=col, lw=0.6, alpha=.25)
    a.plot(range(25, EPOCAS + 1), suave(v), color=col, lw=2.2, label=nom)
a.axvline(mejor_ep, color=c.ROJO, ls="--", label=f"Mínimo validación (época {mejor_ep})")
a.set_yscale("log"); a.set_xlabel("Época"); a.set_ylabel("MSE (datos normalizados)")
a.set_title("Rendimiento durante el entrenamiento"); a.legend()

a = ax[0, 1]
for yr, yp, col, nom in [(y_dev, p_dev, c.GRIS, "Desarrollo"), (y_te, p_te, c.AZUL, "Prueba")]:
    a.scatter(yr, yp, s=16, alpha=.55 if col == c.GRIS else .9, color=col,
              label=f"{nom} (R = {np.corrcoef(yr, yp)[0,1]:.3f})")
lim = [0, 85]; a.plot(lim, lim, "k--", lw=1, label="Ajuste ideal  y = x")
a.set_xlim(lim); a.set_ylim(lim); a.set_xlabel("Resistencia real (MPa)")
a.set_ylabel("Resistencia estimada (MPa)"); a.set_title("Regresión: salida vs. objetivo"); a.legend(loc="upper left")

a = ax[0, 2]
a.hist(err, bins=24, color=c.AZUL, alpha=.9, edgecolor="white")
a.axvline(0, color=c.ROJO, lw=2); a.set_xlabel("Error = real − estimado (MPa)")
a.set_ylabel("Mezclas"); a.set_title(f"Histograma de errores (σ = {err.std():.2f} MPa)")

a = ax[1, 0]
a.scatter(p_te, err, s=18, color=c.VERDE, alpha=.85)
a.axhline(0, color="k", lw=1); a.axhline(2 * err.std(), color=c.ROJO, ls="--", lw=1)
a.axhline(-2 * err.std(), color=c.ROJO, ls="--", lw=1, label="±2σ")
a.set_xlabel("Resistencia estimada (MPa)"); a.set_ylabel("Residuo (MPa)")
a.set_title("Residuos vs. valor estimado"); a.legend()

a = ax[1, 1]
a.barh([ETIQ[nombres[i]] for i in orden][::-1], imp_m[orden][::-1],
       xerr=imp_s[orden][::-1], color=c.AZUL)
a.set_xlabel("Aumento del RMSE al permutar (MPa)"); a.set_title("Importancia de variables")

a = ax[1, 2]                      # dependencia parcial: edad x cemento
base = np.tile(np.median(X_dev, axis=0), (60, 1))   # resto de ingredientes en su mediana
for cem, col in [(200, c.GRIS), (300, c.AZUL), (400, c.NARANJA), (500, c.ROJO)]:
    B = base.copy(); B[:, 0] = cem; B[:, 7] = np.geomspace(1, 90, 60)
    a.plot(B[:, 7], predecir(B), lw=2.2, color=col, label=f"Cemento = {cem} kg/m³")
a.set_xscale("log"); a.set_xlabel("Edad de curado (días, escala log)"); a.set_ylabel("Resistencia estimada (MPa)")
a.set_title("Curvas de curado estimadas (1–90 días)"); a.legend()
fig.tight_layout()
c.guardar(fig, "ej2_resultados.png")
print("\n[OK] Ejemplo 2 finalizado.")
