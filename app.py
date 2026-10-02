"""
Interfaz web (Flask) para predecir con los tres modelos exportados (.pkl).
Ejecutar:   python app.py      y abrir   http://127.0.0.1:5000
"""
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

BASE = Path(__file__).parent
app = Flask(__name__)

EJERCICIOS = {
    "dolar": {"titulo": "Precio del dólar", "icono": "💵",
              "ayuda": {"Dia": "Número de día (1 a 500 en los datos)",
                        "Inflacion": "Tasa de inflación diaria (ej. 0.02)",
                        "Tasa_interes": "Tasa de interés diaria (ej. 5.0)"}},
    "glucosa": {"titulo": "Nivel de glucosa en sangre", "icono": "🩸",
                "ayuda": {"Edad": "Edad del paciente en años",
                          "IMC": "Índice de masa corporal",
                          "Actividad_Fisica": "Horas semanales de ejercicio"}},
    "energia": {"titulo": "Consumo de energía eléctrica", "icono": "⚡",
                "ayuda": {"Temperatura": "Temperatura en °C",
                          "Hora": "Hora del día (1 a 24)",
                          "Dia_Semana": "1 = Lunes ... 7 = Domingo"}},
}

# Los modelos se cargan una sola vez al iniciar la aplicación
MODELOS = {k: joblib.load(BASE / "modelos" / f"modelo_{k}.pkl") for k in EJERCICIOS}


def info_publica():
    salida = {}
    for k, e in EJERCICIOS.items():
        m = MODELOS[k]
        salida[k] = {
            "titulo": e["titulo"], "icono": e["icono"], "objetivo": m["objetivo"], "unidad": m["unidad"],
            "metricas": m["metricas"],
            "campos": [{"nombre": f, "ayuda": e["ayuda"][f], "min": m["rangos"][f][0],
                        "max": m["rangos"][f][1], "media": m["medias"][f]} for f in m["features"]],
        }
    return salida


@app.route("/")
def index():
    return render_template_string(PLANTILLA, info=info_publica())


@app.route("/predecir", methods=["POST"])
def predecir():
    datos = request.get_json(force=True)
    clave = datos.get("ejercicio")
    if clave not in MODELOS:
        return jsonify(error="Ejercicio no válido"), 400
    m = MODELOS[clave]
    try:
        valores = {f: float(datos["valores"][f]) for f in m["features"]}
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Verifica que todos los campos tengan un valor numérico."), 400
    X = pd.DataFrame([valores], columns=m["features"])
    pred = float(m["modelo"].predict(X)[0])
    avisos = [f"{f} = {v:g} está fuera del rango de entrenamiento ({m['rangos'][f][0]:.3g} a {m['rangos'][f][1]:.3g}); "
              f"la predicción es una extrapolación." for f, v in valores.items()
              if not (m["rangos"][f][0] <= v <= m["rangos"][f][1])]
    return jsonify(prediccion=round(pred, 2), objetivo=m["objetivo"], unidad=m["unidad"], avisos=avisos)


PLANTILLA = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Predicciones con Regresión Lineal</title>
<style>
 :root{--bg:#f4f6fb;--card:#fff;--ink:#1a202c;--mut:#667085;--pri:#2b6cb0;--ok:#2f855a;--warn:#b7791f}
 *{box-sizing:border-box} body{margin:0;font-family:system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink)}
 header{background:linear-gradient(135deg,#2b6cb0,#2c5282);color:#fff;padding:28px 20px;text-align:center}
 header h1{margin:0 0 6px;font-size:1.6rem} header p{margin:0;opacity:.85}
 main{max-width:760px;margin:24px auto;padding:0 16px}
 .tabs{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px}
 .tab{flex:1;min-width:150px;padding:14px;border:2px solid #d5dbe8;border-radius:12px;background:#fff;cursor:pointer;font-size:1rem;text-align:center}
 .tab.act{border-color:var(--pri);background:#ebf4ff;font-weight:600}
 .card{background:var(--card);border-radius:14px;padding:22px;box-shadow:0 2px 10px rgba(0,0,0,.06)}
 label{display:block;font-weight:600;margin-top:14px} small{color:var(--mut)}
 input{width:100%;padding:11px;margin-top:6px;border:1px solid #cbd5e0;border-radius:8px;font-size:1rem}
 input:focus{outline:2px solid var(--pri);border-color:var(--pri)}
 button.go{margin-top:20px;width:100%;padding:13px;border:0;border-radius:10px;background:var(--pri);color:#fff;font-size:1.05rem;cursor:pointer}
 button.go:hover{background:#2c5282}
 #res{margin-top:18px;display:none;text-align:center;padding:18px;border-radius:12px;background:#f0fff4;border:1px solid #9ae6b4}
 #res .v{font-size:2.2rem;font-weight:700;color:var(--ok)} #res .o{color:var(--mut)}
 .av{margin-top:10px;color:var(--warn);font-size:.9rem;text-align:left}
 .err{background:#fff5f5;border-color:#feb2b2!important} .err .v{color:#c53030;font-size:1.1rem}
 .met{margin-top:16px;font-size:.88rem;color:var(--mut);text-align:center}
</style></head><body>
<header><h1>Predicciones con Regresión Lineal Múltiple</h1><p>Laboratorio 1 · Minería de Datos (CRISP-DM)</p></header>
<main>
 <div class="tabs" id="tabs"></div>
 <div class="card"><h2 id="tit" style="margin-top:0"></h2><div id="campos"></div>
  <button class="go" onclick="predecir()">Predecir</button>
  <div id="res"></div><div class="met" id="met"></div></div>
</main>
<script>
const INFO = {{ info|tojson }}; let actual = Object.keys(INFO)[0];
function render(){
  document.getElementById('tabs').innerHTML = Object.entries(INFO).map(([k,e]) =>
    `<button class="tab ${k===actual?'act':''}" onclick="actual='${k}';render()">${e.icono} ${e.titulo}</button>`).join('');
  const e = INFO[actual];
  document.getElementById('tit').textContent = `${e.icono} ${e.titulo}`;
  document.getElementById('campos').innerHTML = e.campos.map(c =>
    `<label for="${c.nombre}">${c.nombre} <small>— ${c.ayuda}</small></label>
     <input id="${c.nombre}" type="number" step="any" placeholder="Rango de datos: ${c.min.toPrecision(3)} a ${c.max.toPrecision(3)}">`).join('');
  document.getElementById('res').style.display='none';
  const m = e.metricas;
  document.getElementById('met').textContent = `Desempeño del modelo (prueba): R² = ${m.R2.toFixed(3)} · RMSE = ${m.RMSE.toFixed(2)} ${e.unidad}`;
}
async function predecir(){
  const e = INFO[actual], valores = {}, r = document.getElementById('res');
  e.campos.forEach(c => valores[c.nombre] = document.getElementById(c.nombre).value);
  const resp = await fetch('/predecir',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ejercicio:actual,valores})});
  const d = await resp.json(); r.style.display='block';
  if(!resp.ok){ r.className='err'; r.innerHTML=`<div class="v">${d.error}</div>`; return; }
  r.className=''; r.id='res';
  r.innerHTML = `<div class="o">${d.objetivo}</div><div class="v">${d.prediccion.toLocaleString('es-CO',{minimumFractionDigits:2})} ${d.unidad}</div>` +
    d.avisos.map(a=>`<div class="av">⚠️ ${a}</div>`).join('');
}
render();
</script></body></html>"""

if __name__ == "__main__":
    app.run(debug=True)
