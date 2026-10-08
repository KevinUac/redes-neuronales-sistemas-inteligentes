"""
EJEMPLO 3 - RED NEURONAL TIPO SERIES DE TIEMPO (NARX: autorregresiva con entradas exogenas)
Problema industrial/ambiental: pronosticar la concentracion de PM2.5 en Beijing 6 horas adelante
y emitir una alerta (clasificacion) cuando se superen 150 ug/m3.
Dataset : Beijing PM2.5 (UCI, Liang et al. 2015) - 43,824 registros horarios, 2010-2014.
Modelo  : MLP con ventana de 24 h (MLPRegressor, scikit-learn) + early stopping cronologico.
Autor   : Kevin del Jesus Gonzalez Maas - Sistemas Inteligentes - FI-UAC
"""
import copy, itertools, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error, roc_auc_score,
                             precision_score, recall_score, f1_score, confusion_matrix)
import comun as c

warnings.filterwarnings("ignore")
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv"
VENTANA, H_PRINCIPAL, UMBRAL_ALERTA = 24, 6, 150.0     # horas de historia, horizonte, ug/m3
HORIZONTES = [1, 3, 6, 12, 24]
c.estilo_figuras()

# ---------------------------------------------------------------- 1. DATOS
c.encabezado("1. CARGA Y PREPROCESAMIENTO DE LA SERIE")
df = pd.read_csv(c.descargar(URL, "pm25_beijing.csv"), na_values="NA")
df["fecha"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
df = df.set_index("fecha")
faltante = df["pm2.5"].isna()
print(f"Registros horarios: {len(df)} ({df.index[0]:%Y-%m-%d} a {df.index[-1]:%Y-%m-%d})")
print(f"PM2.5 faltante: {faltante.sum()} horas ({faltante.mean():.1%}); se interpola SOLO como entrada,")
print("                 nunca se evalua ni se entrena contra valores interpolados.")
pm = df["pm2.5"].interpolate(limit=24, limit_direction="both")
viento = pd.get_dummies(df["cbwd"]).astype(float)                # NW, NE, SE, cv
F = pd.DataFrame({"pm": np.log1p(pm), "dewp": df["DEWP"], "temp": df["TEMP"],
                  "pres": df["PRES"], "iws": np.log1p(df["Iws"])}, index=df.index)
retardos = [F.shift(k).add_suffix(f"_t-{k}") for k in range(VENTANA)]          # t, t-1, ..., t-23
fija = pd.concat([viento.add_prefix("viento_"), df[["Is", "Ir"]],
                  pd.DataFrame({"h_sin": np.sin(2*np.pi*df.index.hour/24),
                                "h_cos": np.cos(2*np.pi*df.index.hour/24),
                                "d_sin": np.sin(2*np.pi*df.index.dayofyear/365.25),
                                "d_cos": np.cos(2*np.pi*df.index.dayofyear/365.25)}, index=df.index)], axis=1)
X_all = pd.concat(retardos + [fija], axis=1)
# Entradas para el baseline lineal (ARX): mismas variables pero PM2.5 y viento en escala real
F_lin = F.copy(); F_lin["pm"], F_lin["iws"] = pm, df["Iws"]
X_lin = pd.concat([F_lin.shift(k).add_suffix(f"_t-{k}") for k in range(VENTANA)] + [fija], axis=1)
print(f"Vector de entrada: {VENTANA} h x 5 variables + {fija.shape[1]} fijas = {X_all.shape[1]} entradas")

def armar(h):
    """Construye (X, y, pm_actual, fechas, X_lineal) para pronosticar PM2.5 en t+h."""
    y = df["pm2.5"].shift(-h)                                    # valor real observado en t+h
    ok = X_all.notna().all(axis=1) & y.notna() & (~faltante) & (~faltante.shift(-h, fill_value=True))
    return X_all[ok].values, y[ok].values, df["pm2.5"][ok].values, X_all.index[ok], X_lin[ok].values

def particion(n):
    return int(0.70 * n), int(0.85 * n)                          # cronologica 70/15/15

def escalar(Xtr, *otros):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    return [(a - mu) / sd for a in (Xtr, *otros)]

# --------------------------------------- 2. ENTRENAMIENTO CON EARLY STOPPING CRONOLOGICO
def entrenar(Xtr, ytr, Xva, yva, h, alpha, lr, semilla=c.SEMILLA, max_ep=150, paciencia=12):
    """Adam por epocas; se conserva la red con menor error en validacion (que es POSTERIOR al entrenamiento)."""
    m = MLPRegressor(hidden_layer_sizes=h, alpha=alpha, learning_rate_init=lr, batch_size=256,
                     solver="adam", random_state=semilla)
    mejor, w, hist_tr, hist_va, sin_mejora = np.inf, None, [], [], 0
    for ep in range(max_ep):
        m.partial_fit(Xtr, ytr)
        e_tr, e_va = mean_squared_error(ytr, m.predict(Xtr)), mean_squared_error(yva, m.predict(Xva))
        hist_tr.append(e_tr); hist_va.append(e_va)
        if e_va < mejor - 1e-5:
            mejor, sin_mejora = e_va, 0
            w = (copy.deepcopy(m.coefs_), copy.deepcopy(m.intercepts_))
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia: break
    m.coefs_, m.intercepts_ = w
    return m, np.array(hist_tr), np.array(hist_va), int(np.argmin(hist_va)) + 1

def preparar(h):
    X, y, pm_act, fechas, XL = armar(h)
    i1, i2 = particion(len(y))
    Xtr, Xva, Xte = escalar(X[:i1], X[i1:i2], X[i2:])
    Ltr, Lva, Lte = escalar(XL[:i1], XL[i1:i2], XL[i2:])
    return dict(Xtr=Xtr, Xva=Xva, Xte=Xte, Ltr=Ltr, Lva=Lva, Lte=Lte,
                ytr=y[:i1], yva=y[i1:i2], yte=y[i2:],
                pm_tr=pm_act[:i1], pm_va=pm_act[i1:i2], pm_te=pm_act[i2:],
                f_te=fechas[i2:], fechas=fechas, i1=i1, i2=i2, y=y)

def ajustar(D, h, alpha, lr, forma):
    """Entrena la red. forma='directa': y(t+h) = red(x);  forma='residual': y(t+h) = PM2.5(t) + red(x).
    Devuelve el predictor en ug/m3: pred(X, pm_actual)."""
    base = (lambda pm: pm) if forma == "residual" else (lambda pm: 0.0 * pm)
    m, htr, hva, ep = entrenar(D["Xtr"], (D["ytr"] - base(D["pm_tr"])) / 100.0,
                               D["Xva"], (D["yva"] - base(D["pm_va"])) / 100.0, h, alpha, lr)
    return m, (lambda Xs, pm: np.maximum(m.predict(Xs) * 100.0 + base(pm), 0.0)), htr, hva, ep

def baseline_arx(Dh):
    """Autorregresivo lineal con entradas exogenas (ARX): Ridge sobre las mismas 24 h en escala real."""
    r = Ridge(alpha=10.0).fit(Dh["Ltr"], Dh["ytr"])
    return lambda Lx: np.maximum(r.predict(Lx), 0.0)

rmse = lambda y, p: float(np.sqrt(mean_squared_error(y, p)))
D = preparar(H_PRINCIPAL)
print(f"\nParticion cronologica (horizonte {H_PRINCIPAL} h): entrenamiento {D['i1']} "
      f"({D['fechas'][0]:%Y-%m-%d} a {D['fechas'][D['i1']-1]:%Y-%m-%d})")
print(f"                                   validacion    {D['i2']-D['i1']} "
      f"({D['fechas'][D['i1']]:%Y-%m-%d} a {D['fechas'][D['i2']-1]:%Y-%m-%d})")
print(f"                                   prueba        {len(D['yte'])} "
      f"({D['fechas'][D['i2']]:%Y-%m-%d} a {D['fechas'][-1]:%Y-%m-%d})")

# ------------------------------------------------- 2. AJUSTE DE HIPERPARAMETROS
c.encabezado("2. AJUSTE DE HIPERPARAMETROS (criterio: RMSE en validacion, ug/m3)")
rejilla = {"forma": ["directa", "residual"], "hidden_layer_sizes": [(32,), (64,), (64, 32)], "alpha": [0.1, 1.0, 3.0]}
res, t0 = [], time.time()
for forma, h, a in itertools.product(*rejilla.values()):
    m, pred, htr_, hva_, ep = ajustar(D, h, a, 3e-3, forma)
    rv = rmse(D["yva"], pred(D["Xva"], D["pm_va"]))
    res.append((rv, forma, h, a, ep))
    print(f"  forma={forma:<9} capas={str(h):<9} alpha={a:<4} mejor_epoca={ep:<3} RMSE_val={rv:.2f}")
res.sort(key=lambda r: r[0])
_, forma_opt, h_opt, a_opt, _ = res[0]
lr_opt = 3e-3
print(f"({len(res)} configuraciones, {time.time()-t0:.0f} s)")
print(f"RMSE de validacion de referencia: persistencia = {rmse(D['yva'], D['pm_va']):.2f} | "
      f"ARX lineal = {rmse(D['yva'], baseline_arx(D)(D['Lva'])):.2f}")
print(f">> Mejor configuracion: forma={forma_opt}, capas={h_opt}, alpha={a_opt}, lr={lr_opt}")

# ------------------------------------------------------ 3. ENTRENAMIENTO FINAL
c.encabezado("3. ENTRENAMIENTO FINAL (horizonte principal)")
red, pred_nn, htr, hva, mejor_ep = ajustar(D, h_opt, a_opt, lr_opt, forma_opt)
print(f"Arquitectura: {D['Xtr'].shape[1]} -> {' -> '.join(map(str, h_opt))} -> 1 | "
      f"parametros: {sum(w.size for w in red.coefs_) + sum(b.size for b in red.intercepts_)}")
print(f"Epocas entrenadas: {len(htr)} | mejor epoca (validacion): {mejor_ep}")

# ------------------------------------------------------------ 4. EVALUACION
c.encabezado(f"4. EVALUACION EN PRUEBA - PRONOSTICO A {H_PRINCIPAL} HORAS")
def mets(nombre, y, p):
    return (nombre, r2_score(y, p), rmse(y, p), mean_absolute_error(y, p), np.corrcoef(y, p)[0, 1])
p_nn, p_ar, p_pers = pred_nn(D["Xte"], D["pm_te"]), baseline_arx(D)(D["Lte"]), D["pm_te"]
tab = pd.DataFrame([mets("Persistencia (PM2.5 actual)", D["yte"], p_pers),
                    mets("Autorregresivo lineal ARX", D["yte"], p_ar),
                    mets("Red neuronal NARX (propuesta)", D["yte"], p_nn)],
                   columns=["Modelo", "R2", "RMSE", "MAE", "R"])
print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
rm = tab["RMSE"].values
print(f"\nHabilidad respecto a la persistencia (1 - RMSE/RMSE_pers): NARX = {1-rm[2]/rm[0]:+.1%} | "
      f"ARX = {1-rm[1]/rm[0]:+.1%}")
print(f"Mejora de la red NARX sobre el ARX lineal: {1-rm[2]/rm[1]:+.1%} en RMSE")
err = D["yte"] - p_nn
print(f"Error NARX: media = {err.mean():+.2f} ug/m3 | desviacion estandar = {err.std():.2f} ug/m3")

# Barrido de horizontes (forma y alpha se re-sintonizan en validacion para cada horizonte)
print("\nRMSE de prueba (ug/m3) por horizonte de pronostico (forma y alpha re-sintonizados en validacion):")
print(f"  {'h':>3} | {'forma':>8} | {'alpha':>5} | {'Persistencia':>12} | {'ARX':>7} | {'NARX':>7} | {'R2 NARX':>8}")
barrido = []
for h in HORIZONTES:
    Dh = preparar(h)
    cand = []
    for forma in ["directa", "residual"]:
        for a in [0.1, 1.0, 10.0]:
            mh, ph, *_ = ajustar(Dh, h_opt, a, lr_opt, forma)
            cand.append((rmse(Dh["yva"], ph(Dh["Xva"], Dh["pm_va"])), forma, a, ph))
    _, f_h, a_h, ph = min(cand, key=lambda t: t[0])
    pn, pa = ph(Dh["Xte"], Dh["pm_te"]), baseline_arx(Dh)(Dh["Lte"])
    fila = (h, rmse(Dh["yte"], Dh["pm_te"]), rmse(Dh["yte"], pa), rmse(Dh["yte"], pn), r2_score(Dh["yte"], pn))
    barrido.append(fila)
    print(f"  {h:>3} | {f_h:>8} | {a_h:>5} | {fila[1]:>12.2f} | {fila[2]:>7.2f} | {fila[3]:>7.2f} | {fila[4]:>8.4f}")
barrido = np.array(barrido)

# Clasificacion derivada: alerta si PM2.5(t+h) > 150 ug/m3 (umbral del modelo calibrado en validacion)
print(f"\nSistema de ALERTA (PM2.5 en t+{H_PRINCIPAL} h > {UMBRAL_ALERTA:.0f} ug/m3) - clasificacion binaria:")
real, real_va = (D["yte"] > UMBRAL_ALERTA).astype(int), (D["yva"] > UMBRAL_ALERTA).astype(int)
print(f"  Horas de prueba con alerta real: {real.sum()} de {len(real)} ({real.mean():.1%})")
cand_u = np.arange(60, 201, 2.5)
def umbral_opt(p_val):
    return float(cand_u[int(np.argmax([f1_score(real_va, p_val > u) for u in cand_u]))])
t_nn, t_pers = umbral_opt(pred_nn(D["Xva"], D["pm_va"])), umbral_opt(D["pm_va"])
print(f"  Umbral de pronostico calibrado en validacion (max F1): NARX = {t_nn:.1f} | persistencia = {t_pers:.1f} ug/m3")
for nom, p, u in [("Persistencia", p_pers, t_pers), ("NARX", p_nn, t_nn)]:
    pred = (p > u).astype(int)
    print(f"  {nom:<13} exactitud={np.mean(pred==real):.4f} precision={precision_score(real, pred):.4f} "
          f"sensibilidad={recall_score(real, pred):.4f} F1={f1_score(real, pred):.4f} AUC={roc_auc_score(real, p):.4f}")
cm = confusion_matrix(real, (p_nn > t_nn).astype(int))
print(f"  Matriz de confusion NARX [VN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} VP={cm[1,1]}]")

# Autocorrelacion del error (diagnostico estilo MATLAB)
e0 = err - err.mean()
ac = np.array([np.sum(e0[:len(e0)-k] * e0[k:]) for k in range(49)]) / np.sum(e0**2)
lim = 1.96 / np.sqrt(len(e0))

# ------------------------------------------------------------------ 5. FIGURAS
# 5a. Vista general de los datos
fig, ax = plt.subplots(1, 2, figsize=(16, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})
fig.suptitle("Ejemplo 3 · Series de tiempo: exploración del conjunto de datos PM2.5 (Beijing, 2010–2014)",
             fontsize=14, fontweight="bold", y=1.03)
f_all, y_all = D["fechas"], D["y"]
a = ax[0]
for i0, i1_, col, nom in [(0, D["i1"], c.AZUL, "Entrenamiento (70 %)"), (D["i1"], D["i2"], c.NARANJA, "Validación (15 %)"),
                          (D["i2"], len(y_all), c.VERDE, "Prueba (15 %)")]:
    a.plot(f_all[i0:i1_], y_all[i0:i1_], lw=.5, color=col, label=nom)
a.axhline(UMBRAL_ALERTA, color=c.ROJO, ls="--", lw=1.2, label=f"Umbral de alerta ({UMBRAL_ALERTA:.0f} µg/m³)")
a.set_ylabel("PM2.5 (µg/m³)"); a.set_title("Serie horaria y partición cronológica"); a.legend(ncol=4, loc="upper center")
a = ax[1]
mes = pd.Series(y_all, index=f_all).groupby(f_all.month)
a.boxplot([g.values for _, g in mes], showfliers=False, patch_artist=True,
          boxprops=dict(facecolor="#BBD3EA"), medianprops=dict(color=c.ROJO, lw=2))
a.set_xticks(range(1, 13), ["E", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
a.set_xlabel("Mes"); a.set_ylabel("PM2.5 (µg/m³)"); a.set_title("Estacionalidad (temporada de calefacción)")
fig.tight_layout(); c.guardar(fig, "ej3_datos.png")

# 5b. Resultados
fig, ax = plt.subplots(2, 3, figsize=(16, 9.2))
fig.suptitle(f"Ejemplo 3 · Series de tiempo (NARX): pronóstico de PM2.5 a {H_PRINCIPAL} horas",
             fontsize=14, fontweight="bold", y=1.0)
a = ax[0, 0]
a.plot(range(1, len(htr)+1), htr, color=c.AZUL, lw=2, label="Entrenamiento")
a.plot(range(1, len(hva)+1), hva, color=c.NARANJA, lw=2, label="Validación")
a.axvline(mejor_ep, color=c.ROJO, ls="--", label=f"Mejor época = {mejor_ep}")
a.set_xlabel("Época"); a.set_ylabel("MSE (escala normalizada)"); a.set_title("Rendimiento durante el entrenamiento"); a.legend()

a = ax[0, 1]                                         # ventana de 14 dias con el episodio mas severo
ventana = 24 * 14
pico = int(np.argmax(pd.Series(D["yte"]).rolling(ventana).max().values))
ini = max(0, min(pico - ventana // 2, len(D["yte"]) - ventana)); sl = slice(ini, ini + ventana)
a.fill_between(D["f_te"][sl], 0, D["yte"][sl], color="#D6DBDF", label="Real")
a.plot(D["f_te"][sl], p_nn[sl], color=c.AZUL, lw=1.8, label=f"NARX (t+{H_PRINCIPAL} h)")
a.plot(D["f_te"][sl], p_pers[sl], color=c.NARANJA, lw=1, alpha=.8, label="Persistencia")
a.axhline(UMBRAL_ALERTA, color=c.ROJO, ls="--", lw=1)
a.set_ylabel("PM2.5 (µg/m³)"); a.set_title("Pronóstico vs. real (14 días de prueba)"); a.legend(loc="upper right")
a.tick_params(axis="x", rotation=25)

a = ax[0, 2]
hb = a.hexbin(D["yte"], p_nn, gridsize=45, bins="log", cmap="Blues", mincnt=1, extent=(0, 500, 0, 500))
m_ = 500; a.plot([0, m_], [0, m_], "k--", lw=1, label="Ajuste ideal")
a.set_xlim(0, m_); a.set_ylim(0, m_); a.set_xlabel("PM2.5 real (µg/m³)"); a.set_ylabel("PM2.5 pronosticado (µg/m³)")
a.set_title(f"Regresión (R = {np.corrcoef(D['yte'], p_nn)[0,1]:.3f}, R² = {r2_score(D['yte'], p_nn):.3f})")
a.legend(loc="upper left"); plt.colorbar(hb, ax=a, label="Horas (log)")

a = ax[1, 0]
for j, nom, col, mk in [(1, "Persistencia", c.NARANJA, "s"), (2, "ARX lineal", c.GRIS, "^"), (3, "NARX (MLP)", c.AZUL, "o")]:
    a.plot(barrido[:, 0], barrido[:, j], marker=mk, lw=2.2, color=col, label=nom)
a.set_xlabel("Horizonte de pronóstico (h)"); a.set_ylabel("RMSE (µg/m³)"); a.set_title("Error vs. horizonte"); a.legend()

a = ax[1, 1]
a.bar(range(49), ac, color=c.AZUL, width=.7)
a.axhline(lim, color=c.ROJO, ls="--"); a.axhline(-lim, color=c.ROJO, ls="--", label="Límites de confianza 95 %")
a.set_xlabel("Retardo (h)"); a.set_ylabel("Autocorrelación"); a.set_title("Autocorrelación del error"); a.legend()

a = ax[1, 2]
a.imshow(cm, cmap="Blues"); a.grid(False)
for (i, j), v in np.ndenumerate(cm):
    a.text(j, i, f"{v}\n({v/cm.sum():.1%})", ha="center", va="center", fontsize=12, fontweight="bold",
           color="white" if v > cm.max() / 2 else "black")
a.set_xticks([0, 1], ["Sin alerta", "Alerta"]); a.set_yticks([0, 1], ["Sin alerta", "Alerta"])
a.set_xlabel("Predicho"); a.set_ylabel("Real")
a.set_title(f"Alerta > {UMBRAL_ALERTA:.0f} µg/m³  (AUC = {roc_auc_score(real, p_nn):.3f})")
fig.tight_layout(); c.guardar(fig, "ej3_resultados.png")
print("\n[OK] Ejemplo 3 finalizado.")
