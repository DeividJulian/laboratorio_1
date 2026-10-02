"""
Laboratorio 1 - Minería de Datos (CRISP-DM)
Regresión lineal múltiple para tres escenarios: Dólar, Glucosa y Energía.

Fases CRISP-DM cubiertas en este script:
  1. Comprensión del negocio  -> definida en CONFIG (objetivo y variables)
  2. Comprensión de los datos -> estadísticos descriptivos, nulos, correlación
  3. Preparación de datos     -> verificación de nulos/duplicados, partición train/test
  4. Modelado                 -> LinearRegression (scikit-learn)
  5. Evaluación               -> MSE, RMSE, R², coeficientes, p-valores, importancia
  6. Despliegue               -> exportación .pkl (joblib) usada por app.py (Flask)

Uso:  python entrenamiento.py
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent
DATA, MODELOS, GRAFICAS = BASE / "data", BASE / "modelos", BASE / "graficas"
for d in (MODELOS, GRAFICAS):
    d.mkdir(exist_ok=True)

SEMILLA = 42
TEST_SIZE = 0.20

CONFIG = {
    "dolar": {
        "archivo": "dolar_data.csv",
        "nombre": "Precio del dólar",
        "objetivo": "Precio_Dolar",
        "unidad": "COP",
        "features": ["Dia", "Inflacion", "Tasa_interes"],
    },
    "glucosa": {
        "archivo": "glucosa_data.csv",
        "nombre": "Nivel de glucosa en sangre",
        "objetivo": "Nivel_Glucosa",
        "unidad": "mg/dL",
        "features": ["Edad", "IMC", "Actividad_Fisica"],
    },
    "energia": {
        "archivo": "energia_data.csv",
        "nombre": "Consumo de energía eléctrica",
        "objetivo": "Consumo_Energia",
        "unidad": "kWh",
        "features": ["Temperatura", "Hora", "Dia_Semana"],
    },
}


def ols_resumen(X, y, modelo):
    """Errores estándar, estadístico t y p-valor de cada coeficiente (OLS clásico)."""
    n, k = X.shape
    Xc = np.column_stack([np.ones(n), X.values])
    beta = np.r_[modelo.intercept_, modelo.coef_]
    resid = y.values - Xc @ beta
    sigma2 = resid @ resid / (n - k - 1)
    cov = sigma2 * np.linalg.inv(Xc.T @ Xc)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=n - k - 1))
    return pd.DataFrame({"Variable": ["Intercepto"] + list(X.columns), "Coeficiente": beta,
                         "Error_est": se, "t": t, "p_valor": p})


def graficar(df, cfg, clave, y_test, y_pred):
    obj, feats = cfg["objetivo"], cfg["features"]
    # 1) Dispersión de cada variable independiente vs. dependiente (con recta de tendencia)
    fig, axes = plt.subplots(1, len(feats), figsize=(5.2 * len(feats), 4.2))
    for ax, f in zip(axes, feats):
        muestra = df.sample(min(len(df), 2000), random_state=SEMILLA)
        ax.scatter(muestra[f], muestra[obj], s=10, alpha=0.4, color="#2b6cb0")
        m, b = np.polyfit(df[f], df[obj], 1)
        xs = np.linspace(df[f].min(), df[f].max(), 100)
        ax.plot(xs, m * xs + b, color="#c53030", lw=2, label=f"tendencia (r = {df[f].corr(df[obj]):.2f})")
        ax.set_xlabel(f); ax.set_ylabel(f"{obj} ({cfg['unidad']})"); ax.legend(loc="best"); ax.grid(alpha=.3)
    fig.suptitle(f"{cfg['nombre']}: relación de cada variable con {obj}", fontweight="bold")
    fig.tight_layout(); fig.savefig(GRAFICAS / f"{clave}_dispersion.png", dpi=130); plt.close(fig)

    # 2) Matriz de correlación
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title(f"Correlación - {cfg['nombre']}")
    fig.tight_layout(); fig.savefig(GRAFICAS / f"{clave}_correlacion.png", dpi=130); plt.close(fig)

    # 3) Real vs. predicho y residuos
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(y_test, y_pred, s=10, alpha=.5, color="#2b6cb0")
    lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[0].plot(lim, lim, "r--", label="Predicción perfecta")
    axes[0].set_xlabel("Valor real"); axes[0].set_ylabel("Valor predicho")
    axes[0].set_title("Real vs. predicho (conjunto de prueba)"); axes[0].legend(); axes[0].grid(alpha=.3)
    res = y_test - y_pred
    axes[1].scatter(y_pred, res, s=10, alpha=.5, color="#2f855a")
    axes[1].axhline(0, color="r", ls="--")
    axes[1].set_xlabel("Valor predicho"); axes[1].set_ylabel("Residuo"); axes[1].set_title("Residuos"); axes[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(GRAFICAS / f"{clave}_evaluacion.png", dpi=130); plt.close(fig)


resultados = {}
for clave, cfg in CONFIG.items():
    print("=" * 70)
    print(f"EJERCICIO: {cfg['nombre'].upper()}")
    print("=" * 70)
    df = pd.read_csv(DATA / cfg["archivo"])
    obj, feats = cfg["objetivo"], cfg["features"]

    # --- Comprensión y preparación de los datos ---
    print(f"Filas x columnas: {df.shape} | nulos: {int(df.isna().sum().sum())} | duplicados: {int(df.duplicated().sum())}")
    print(df.describe().T.round(3).to_string())
    print("\nCorrelación con la variable dependiente:")
    corr = df.corr()[obj].drop(obj)
    print(corr.round(3).to_string())

    X, y = df[feats], df[obj]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEMILLA)

    # --- Modelado ---
    modelo = LinearRegression().fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    y_pred_tr = modelo.predict(X_train)

    # --- Evaluación ---
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2_tr = r2_score(y_train, y_pred_tr)
    print(f"\nMSE={mse:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  R2(test)={r2:.4f}  R2(train)={r2_tr:.4f}")

    tabla = ols_resumen(X_train, y_train, modelo)
    # Coeficientes estandarizados (beta*sd_x/sd_y): permiten comparar el impacto entre variables
    estd = {f: modelo.coef_[i] * X_train[f].std() / y_train.std() for i, f in enumerate(feats)}
    tabla["Coef_estandarizado"] = [np.nan] + [estd[f] for f in feats]
    print("\nCoeficientes:")
    print(tabla.round(5).to_string(index=False))
    mayor = max(estd, key=lambda k: abs(estd[k]))
    print(f"-> Variable de mayor impacto (|beta estandarizado|): {mayor} ({estd[mayor]:.3f})")

    graficar(df, cfg, clave, y_test, y_pred)

    # --- Despliegue: exportar el modelo con joblib ---
    paquete = {
        "modelo": modelo,
        "features": feats,
        "objetivo": obj,
        "unidad": cfg["unidad"],
        "rangos": {f: [float(df[f].min()), float(df[f].max())] for f in feats},
        "medias": {f: float(df[f].mean()) for f in feats},
        "metricas": {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2},
    }
    joblib.dump(paquete, MODELOS / f"modelo_{clave}.pkl")

    resultados[clave] = {
        "n": len(df), "n_train": len(X_train), "n_test": len(X_test),
        "MSE": mse, "RMSE": rmse, "MAE": mae, "R2_test": r2, "R2_train": r2_tr,
        "intercepto": float(modelo.intercept_),
        "coeficientes": {f: float(c) for f, c in zip(feats, modelo.coef_)},
        "p_valores": {r.Variable: float(r.p_valor) for r in tabla.itertuples()},
        "estandarizados": {k: float(v) for k, v in estd.items()},
        "correlaciones": {k: float(v) for k, v in corr.items()},
        "mayor_impacto": mayor,
    }

(BASE / "resultados.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False))
print("\nModelos guardados en", MODELOS)
