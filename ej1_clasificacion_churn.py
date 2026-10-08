"""
EJEMPLO 1 - RED NEURONAL TIPO PATTERN RECOGNITION (CLASIFICACION)
Problema comercial: predecir la fuga de clientes (churn) de una empresa de telecomunicaciones.
Dataset : IBM Telco Customer Churn (7,043 clientes, 20 variables, clase Yes/No).
Modelo  : Perceptron multicapa (MLPClassifier, scikit-learn) con ajuste de hiperparametros.
Autor   : Kevin del Jesus Gonzalez Maas - Sistemas Inteligentes - FI-UAC
"""
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix,
                             precision_recall_curve, average_precision_score)
import comun as c

URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
       "master/data/Telco-Customer-Churn.csv")
c.estilo_figuras()

# ---------------------------------------------------------------- 1. DATOS
c.encabezado("1. CARGA Y PREPROCESAMIENTO DE DATOS")
df = pd.read_csv(c.descargar(URL, "telco_churn.csv"))
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
y = (df["Churn"] == "Yes").astype(int).values
X = pd.get_dummies(df.drop(columns=["customerID", "Churn"]), drop_first=True).astype(float)
print(f"Registros: {len(df)} | Variables de entrada tras codificacion: {X.shape[1]}")
print(f"Clase positiva (abandona): {y.mean():.1%} | Clase negativa (permanece): {1-y.mean():.1%}")

# Division estratificada 70 / 15 / 15 (entrenamiento / validacion / prueba)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=c.SEMILLA)
X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=c.SEMILLA)
esc = StandardScaler().fit(X_tr)                 # el escalador solo ve entrenamiento
X_tr_s, X_va_s, X_te_s = esc.transform(X_tr), esc.transform(X_va), esc.transform(X_te)
print(f"Entrenamiento: {len(y_tr)} | Validacion: {len(y_va)} | Prueba: {len(y_te)}")

# El desbalance (26.5 % vs 73.5 %) se trata con el UMBRAL de decision (seccion 3),
# sin duplicar muestras: asi la validacion interna del early stopping no se contamina.
X_bal, y_bal = X_tr_s, y_tr

# ------------------------------------------------- 2. AJUSTE DE HIPERPARAMETROS
c.encabezado("2. AJUSTE DE HIPERPARAMETROS (busqueda en rejilla, criterio: AUC en validacion)")
rejilla = {"hidden_layer_sizes": [(8,), (16,), (32,), (32, 16)],
           "alpha": [1e-1, 1.0, 3.0, 10.0],
           "learning_rate_init": [1e-3, 5e-3]}
resultados = []
for h, a, lr in itertools.product(*rejilla.values()):
    m = MLPClassifier(hidden_layer_sizes=h, alpha=a, learning_rate_init=lr, activation="relu",
                      solver="adam", early_stopping=True, validation_fraction=0.15,
                      n_iter_no_change=12, max_iter=300, random_state=c.SEMILLA)
    m.fit(X_bal, y_bal)
    auc = roc_auc_score(y_va, m.predict_proba(X_va_s)[:, 1])
    resultados.append((auc, h, a, lr, m.n_iter_))
    print(f"  capas={str(h):<10} alpha={a:<6} lr={lr:<6} epocas={m.n_iter_:<3} AUC_val={auc:.4f}")
resultados.sort(key=lambda t: -t[0])
auc_mejor, h_opt, a_opt, lr_opt, _ = resultados[0]
print(f"\n>> Mejor configuracion: capas={h_opt}, alpha={a_opt}, lr={lr_opt}  (AUC_val={auc_mejor:.4f})")

# ------------------------------------------------------ 3. ENTRENAMIENTO FINAL
c.encabezado("3. ENTRENAMIENTO FINAL")
red = MLPClassifier(hidden_layer_sizes=h_opt, alpha=a_opt, learning_rate_init=lr_opt,
                    activation="relu", solver="adam", early_stopping=True,
                    validation_fraction=0.15, n_iter_no_change=12, max_iter=300,
                    random_state=c.SEMILLA).fit(X_bal, y_bal)
print(f"Arquitectura: {X.shape[1]} -> {' -> '.join(map(str, h_opt))} -> 1 (sigmoide)")
print(f"Epocas ejecutadas: {red.n_iter_} | Mejor puntaje de validacion interna: {red.best_validation_score_:.4f}")

# Umbral de decision optimo (maximiza F1 sobre validacion)
p_va = red.predict_proba(X_va_s)[:, 1]
umbrales = np.linspace(0.2, 0.85, 66)
f1s = [f1_score(y_va, p_va >= t) for t in umbrales]
t_opt = float(umbrales[int(np.argmax(f1s))])
print(f"Umbral de decision optimo (max F1 en validacion): {t_opt:.2f}")

# ------------------------------------------------------------ 4. EVALUACION
c.encabezado("4. EVALUACION SOBRE EL CONJUNTO DE PRUEBA (datos nunca vistos)")
p_te = red.predict_proba(X_te_s)[:, 1]
y_hat = (p_te >= t_opt).astype(int)
logit = LogisticRegression(max_iter=1000).fit(X_bal, y_bal)
p_log = logit.predict_proba(X_te_s)[:, 1]
f1s_log = [f1_score(y_va, logit.predict_proba(X_va_s)[:, 1] >= t) for t in umbrales]
t_log = float(umbrales[int(np.argmax(f1s_log))])
base = DummyClassifier(strategy="most_frequent").fit(X_tr_s, y_tr)

def fila(nombre, yh, p):
    return (nombre, accuracy_score(y_te, yh), precision_score(y_te, yh, zero_division=0),
            recall_score(y_te, yh), f1_score(y_te, yh), roc_auc_score(y_te, p) if p is not None else 0.5)

tabla = pd.DataFrame([fila("Clase mayoritaria (referencia)", base.predict(X_te_s), None),
                      fila("Regresion logistica (umbral propio)", (p_log >= t_log).astype(int), p_log),
                      fila("Red neuronal MLP (propuesta)", y_hat, p_te)],
                     columns=["Modelo", "Exactitud", "Precision", "Sensibilidad", "F1", "AUC"])
print(tabla.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
cm = confusion_matrix(y_te, y_hat)
tn, fp, fn, tp = cm.ravel()
print(f"\nMatriz de confusion  [VN={tn}  FP={fp}  FN={fn}  VP={tp}]")
print(f"Especificidad: {tn/(tn+fp):.4f} | Valor predictivo negativo: {tn/(tn+fn):.4f}")
print(f"AUC (red neuronal): {roc_auc_score(y_te, p_te):.4f} | Precision promedio (AP): {average_precision_score(y_te, p_te):.4f}")

# Impacto economico ilustrativo (supuestos explicitos, NO datos reales de la empresa)
costo_perder, costo_campana, exito = 600.0, 80.0, 0.30   # valor del cliente, costo por oferta, exito de retencion
def beneficio(tp_, fp_):  # se contacta a todo cliente marcado como "abandona"
    return tp_ * (exito * costo_perder - costo_campana) - fp_ * costo_campana
b_red = beneficio(tp, fp)
b_todos = beneficio(int(y_te.sum()), int((1 - y_te).sum()))
print(f"\nImpacto economico ilustrativo (cliente = ${costo_perder:.0f}, oferta = ${costo_campana:.0f}, exito de retencion = {exito:.0%}):")
print(f"  Campana guiada por la red neuronal : ${b_red:>9,.0f}")
print(f"  Campana a todos los clientes       : ${b_todos:>9,.0f}")
print(f"  Sin campana                        : ${0:>9,.0f}")

# ------------------------------------------------------------ 5. IMPORTANCIA
imp = permutation_importance(red, X_te_s, y_te, scoring="roc_auc", n_repeats=5,
                             random_state=c.SEMILLA)
orden = np.argsort(imp.importances_mean)[::-1][:10]
print("\nTop 10 variables mas influyentes (importancia por permutacion, caida de AUC):")
for i in orden:
    print(f"  {X.columns[i]:<42} {imp.importances_mean[i]:.4f}")

# ------------------------------------------------------------------ 6. FIGURAS
fig, ax = plt.subplots(2, 3, figsize=(16, 9.2))
fig.suptitle("Ejemplo 1 · Pattern Recognition: predicción de fuga de clientes (Telco Churn)",
             fontsize=14, fontweight="bold", y=1.0)
a = ax[0, 0]
a.plot(red.loss_curve_, color=c.AZUL, lw=2, label="Pérdida (entrenamiento)")
a.set_xlabel("Época"); a.set_ylabel("Entropía cruzada"); a.set_title("Curva de aprendizaje")
a2 = a.twinx(); a2.grid(False)
a2.plot(red.validation_scores_, color=c.NARANJA, lw=2, label="Exactitud (validación)")
a2.set_ylabel("Exactitud validación"); a2.spines["right"].set_visible(True)
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
a.legend(h1 + h2, l1 + l2, loc="center right")

a = ax[0, 1]
for p, nom, col in [(p_te, f"MLP  (AUC = {roc_auc_score(y_te, p_te):.3f})", c.AZUL),
                    (p_log, f"Reg. logística  (AUC = {roc_auc_score(y_te, p_log):.3f})", c.GRIS)]:
    fpr, tpr, _ = roc_curve(y_te, p); a.plot(fpr, tpr, lw=2.2, color=col, label=nom)
a.plot([0, 1], [0, 1], "--", color="k", lw=1, label="Azar")
a.set_xlabel("Tasa de falsos positivos"); a.set_ylabel("Tasa de verdaderos positivos")
a.set_title("Curva ROC (prueba)"); a.legend(loc="lower right")

a = ax[0, 2]
im = a.imshow(cm, cmap="Blues"); a.grid(False)
for (i, j), v in np.ndenumerate(cm):
    a.text(j, i, f"{v}\n({v/cm.sum():.1%})", ha="center", va="center", fontsize=12,
           color="white" if v > cm.max() / 2 else "black", fontweight="bold")
a.set_xticks([0, 1], ["Permanece", "Abandona"]); a.set_yticks([0, 1], ["Permanece", "Abandona"])
a.set_xlabel("Clase predicha"); a.set_ylabel("Clase real"); a.set_title(f"Matriz de confusión (umbral {t_opt:.2f})")

a = ax[1, 0]
a.plot(umbrales, f1s, color=c.VERDE, lw=2.2, label="F1 (validación)")
a.axvline(t_opt, color=c.ROJO, ls="--", label=f"Umbral óptimo = {t_opt:.2f}")
a.set_xlabel("Umbral de decisión"); a.set_ylabel("F1"); a.set_title("Selección del umbral"); a.legend()

a = ax[1, 1]
pr, rc, _ = precision_recall_curve(y_te, p_te)
a.plot(rc, pr, color=c.AZUL, lw=2.2, label=f"MLP  (AP = {average_precision_score(y_te, p_te):.3f})")
a.axhline(y_te.mean(), color="k", ls="--", lw=1, label="Referencia (prevalencia)")
a.set_xlabel("Sensibilidad (recall)"); a.set_ylabel("Precisión"); a.set_title("Curva Precisión–Sensibilidad"); a.legend()

a = ax[1, 2]
nombres = [X.columns[i].replace("_", " ")[:30] for i in orden][::-1]
a.barh(nombres, imp.importances_mean[orden][::-1], xerr=imp.importances_std[orden][::-1],
       color=c.AZUL, alpha=.9)
a.set_xlabel("Caída del AUC al permutar"); a.set_title("Variables más influyentes"); a.tick_params(axis="y", labelsize=8)
fig.tight_layout()
c.guardar(fig, "ej1_resultados.png")
print("\n[OK] Ejemplo 1 finalizado.")
