const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, HeadingLevel,
  AlignmentType, LevelFormat, BorderStyle, WidthType, ShadingType, PageBreak,
} = require("docx");

const R = JSON.parse(fs.readFileSync("resultados.json", "utf8"));
const f2 = (x) => x.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const f4 = (x) => x.toFixed(4);
const pv = (p) => (p < 0.001 ? "< 0.001" : p.toFixed(3));

const P = (text, opts = {}) =>
  new Paragraph({ spacing: { after: 120, line: 300 }, alignment: AlignmentType.JUSTIFIED, ...opts,
    children: Array.isArray(text) ? text : [new TextRun(text)] });
const B = (t) => new TextRun({ text: t, bold: true });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const bullet = (runs) => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 80, line: 290 },
  children: Array.isArray(runs) ? runs : [new TextRun(runs)] });

const border = { style: BorderStyle.SINGLE, size: 4, color: "BFC7D5" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (t, w, head) => new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    shading: { fill: head ? "2B6CB0" : "FFFFFF", type: ShadingType.CLEAR, color: "auto" },
    margins: { top: 70, bottom: 70, left: 100, right: 100 },
    children: [new Paragraph({ alignment: head ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: [new TextRun({ text: String(t), bold: head, color: head ? "FFFFFF" : "000000", size: 20 })] })],
  });
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, widths[i], true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, widths[i], false)) }))] });
}
const spacer = () => new Paragraph({ spacing: { after: 100 }, children: [] });

function img(file, wPx, caption) {
  const sizes = { dispersion: [2028, 546], evaluacion: [1430, 546], correlacion: [715, 585] };
  const kind = Object.keys(sizes).find((k) => file.includes(k));
  const [w, h] = sizes[kind];
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(`graficas/${file}`),
        transformation: { width: wPx, height: Math.round((wPx * h) / w) },
        altText: { title: caption, description: caption, name: file } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: caption, italics: true, size: 20, color: "555555" })] }),
  ];
}

function coefTable(clave, varLabels) {
  const r = R[clave];
  const rows = [["Intercepto", f4(r.intercepto), pv(r.p_valores["Intercepto"]), "—"]];
  Object.keys(r.coeficientes).forEach((k) =>
    rows.push([k, f4(r.coeficientes[k]), pv(r.p_valores[k]), r.estandarizados[k].toFixed(3)]));
  return table(["Variable", "Coeficiente (β)", "p-valor", "β estandarizado"], rows, [3000, 2200, 1800, 2360]);
}
function metTable(clave, unit, useMse = true) {
  const r = R[clave];
  return table(["Métrica", "Valor"], [
    ["MSE (prueba)", f2(r.MSE)], ["RMSE (prueba)", `${f2(r.RMSE)} ${unit}`], ["MAE (prueba)", `${f2(r.MAE)} ${unit}`],
    ["R² (prueba)", f4(r.R2_test)], ["R² (entrenamiento)", f4(r.R2_train)],
    ["Registros (entrenamiento / prueba)", `${r.n_train} / ${r.n_test}`],
  ], [5000, 4360]);
}

const d = R.dolar, g = R.glucosa, e = R.energia;
const c = [];

// ---------- Portada ----------
c.push(
  new Paragraph({ spacing: { before: 2400, after: 200 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "LABORATORIO 1 – MINERÍA DE DATOS", bold: true, size: 40, color: "2B6CB0" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text: "Regresión lineal múltiple bajo la metodología CRISP-DM", size: 30 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1600 },
    children: [new TextRun({ text: "Predicción del precio del dólar, del nivel de glucosa y del consumo de energía eléctrica", size: 24, color: "555555" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 }, children: [new TextRun({ text: "Estudiante(s): ______________________________", size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 }, children: [new TextRun({ text: "Docente: ______________________________", size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Fecha: ______________________________", size: 24 })] }),
  new Paragraph({ children: [new PageBreak()] }),
);

// ---------- 1. Introducción ----------
c.push(H1("1. Introducción y objetivo"),
  P("Este informe documenta la aplicación de las fases de la metodología CRISP-DM (comprensión del negocio, comprensión y preparación de los datos, modelado, evaluación y despliegue) a tres problemas de predicción de variables cuantitativas: el precio del dólar, el nivel de glucosa en sangre y el consumo de energía eléctrica. En la fase de modelado se ajustó, para cada caso, un modelo de regresión lineal múltiple; se interpretaron sus coeficientes, se evaluó su desempeño, se exportó el modelo entrenado y se construyó una interfaz web básica para realizar predicciones."),
  P([B("Herramientas: "), new TextRun("Python 3, pandas, scikit-learn (LinearRegression), scipy, matplotlib/seaborn, joblib (exportación .pkl) y Flask (interfaz web).")]),
  H2("1.1 Metodología aplicada"),
  bullet([B("Comprensión del negocio: "), new TextRun("se definió, para cada dataset, la variable dependiente a predecir y las variables independientes disponibles.")]),
  bullet([B("Comprensión de los datos: "), new TextRun("estadísticos descriptivos, búsqueda de valores nulos y duplicados, y correlaciones con la variable dependiente.")]),
  bullet([B("Preparación de los datos: "), new TextRun("los tres datasets no presentaron nulos ni duplicados, por lo que no fue necesario imputar ni eliminar registros. Se dividieron los datos en 80 % para entrenamiento y 20 % para prueba (semilla aleatoria 42).")]),
  bullet([B("Modelado: "), new TextRun("regresión lineal múltiple por mínimos cuadrados ordinarios (LinearRegression).")]),
  bullet([B("Evaluación: "), new TextRun("MSE, RMSE, MAE y R² sobre el conjunto de prueba; coeficientes, p-valores y coeficientes estandarizados.")]),
  bullet([B("Despliegue: "), new TextRun("exportación de cada modelo con joblib (.pkl) y aplicación web Flask.")]),
  H2("1.2 Cómo leer las métricas"),
  P("El MSE es el promedio de los errores al cuadrado y está en unidades al cuadrado de la variable dependiente; el RMSE es su raíz cuadrada y está en las mismas unidades de la variable (más fácil de interpretar). El R² indica la proporción de la variabilidad de la variable dependiente explicada por el modelo (1 es un ajuste perfecto). Para comparar el impacto entre variables que están en escalas distintas se usó el coeficiente estandarizado (β × desviación estándar de X ÷ desviación estándar de Y), que indica cuántas desviaciones estándar cambia Y cuando X cambia una desviación estándar."),
);

// ---------- 2. Dólar ----------
c.push(H1("2. Ejercicio 1: Predicción del precio del dólar"),
  P(`El dataset contiene ${d.n} registros diarios con las variables Dia, Inflacion y Tasa_interes como predictoras y Precio_Dolar como variable dependiente (rango aproximado entre 3,940 y 6,554). Las correlaciones con el precio fueron: Dia = ${d.correlaciones.Dia.toFixed(3)}, Inflacion = ${d.correlaciones.Inflacion.toFixed(3)} y Tasa_interes = ${d.correlaciones.Tasa_interes.toFixed(3)}.`),
  H2("2.1 Modelo y coeficientes"),
  P(`Ecuación ajustada: Precio_Dolar = ${f2(d.intercepto)} + ${f2(d.coeficientes.Dia)}·Dia ${f2(d.coeficientes.Inflacion)}·Inflacion ${f2(d.coeficientes.Tasa_interes)}·Tasa_interes.`),
  coefTable("dolar"), spacer(),
  P([B("Interpretación. "), new TextRun(`Dia: manteniendo constantes las demás variables, cada día adicional aumenta el precio del dólar en promedio ${f2(d.coeficientes.Dia)} unidades (COP); es un efecto positivo, de enorme magnitud relativa (β estandarizado = ${d.estandarizados.Dia.toFixed(3)}) y estadísticamente significativo (p < 0.001). Inflacion: el coeficiente es negativo (${f2(d.coeficientes.Inflacion)}), pero su p-valor (${pv(d.p_valores.Inflacion)}) es mayor que 0.05, por lo que no se puede afirmar un efecto distinto de cero; además, su β estandarizado (${d.estandarizados.Inflacion.toFixed(3)}) es insignificante. Tasa_interes: efecto prácticamente nulo (${f2(d.coeficientes.Tasa_interes)}, p = ${pv(d.p_valores.Tasa_interes)}), no significativo. El intercepto (${f2(d.intercepto)}) es el precio base estimado en el día 0.`)]),
  H2("2.2 Desempeño"),
  metTable("dolar", "COP"), spacer(),
  P(`El modelo explica el ${(d.R2_test * 100).toFixed(2)} % de la variabilidad del precio en el conjunto de prueba (R² = ${f4(d.R2_test)}), con un RMSE de aproximadamente ${f2(d.RMSE)} unidades, pequeño frente a un precio medio cercano a 5,212. El R² de entrenamiento (${f4(d.R2_train)}) es similar al de prueba, lo que indica que no hay sobreajuste.`),
  H2("2.3 Visualizaciones"),
  ...img("dolar_dispersion.png", 624, "Figura 1. Relación de Dia, Inflacion y Tasa_interes con Precio_Dolar."),
  ...img("dolar_evaluacion.png", 560, "Figura 2. Valores reales vs. predichos y residuos (dólar)."),
  P("La gráfica de dispersión confirma una tendencia lineal creciente casi perfecta entre Dia y el precio, mientras que Inflacion y Tasa_interes forman nubes de puntos sin patrón (pendientes casi horizontales)."),
  P([B("Observación metodológica: "), new TextRun("el dataset es una serie temporal (Dia es un índice de tiempo). Para el laboratorio se usó una partición aleatoria; en un caso real convendría validar con una partición cronológica y verificar que los residuos no estén autocorrelacionados. Además, casi todo el poder predictivo proviene de Dia, es decir, de la tendencia; Inflacion y Tasa_interes no aportan información en estos datos.")]),
);

// ---------- 3. Glucosa ----------
c.push(new Paragraph({ children: [new PageBreak()] }),
  H1("3. Ejercicio 2: Predicción del nivel de glucosa en sangre"),
  P(`El dataset contiene ${g.n} pacientes con Edad, IMC y Actividad_Fisica (horas semanales) como predictoras y Nivel_Glucosa (mg/dL) como variable dependiente. Las correlaciones con la glucosa fueron: Edad = ${g.correlaciones.Edad.toFixed(3)}, IMC = ${g.correlaciones.IMC.toFixed(3)} y Actividad_Fisica = ${g.correlaciones.Actividad_Fisica.toFixed(3)}.`),
  H2("3.1 Modelo y coeficientes"),
  P(`Ecuación ajustada: Nivel_Glucosa = ${f2(g.intercepto)} + ${f2(g.coeficientes.Edad)}·Edad + ${f2(g.coeficientes.IMC)}·IMC ${f2(g.coeficientes.Actividad_Fisica)}·Actividad_Fisica.`),
  coefTable("glucosa"), spacer(),
  P([B("Interpretación. "), new TextRun(`Edad: cada año adicional se asocia con un aumento promedio de ${f2(g.coeficientes.Edad)} mg/dL en la glucosa, manteniendo constantes las demás variables. IMC: cada punto adicional de IMC se asocia con ${f2(g.coeficientes.IMC)} mg/dL más de glucosa. Actividad_Fisica: cada hora semanal adicional de ejercicio se asocia con ${f2(Math.abs(g.coeficientes.Actividad_Fisica))} mg/dL menos de glucosa (efecto protector). Las tres variables son estadísticamente significativas (p < 0.001), por lo que todas aportan al modelo. El intercepto (${f2(g.intercepto)}) no tiene interpretación clínica directa porque corresponde a una edad e IMC de 0.`)]),
  H2("3.2 Desempeño"),
  metTable("glucosa", "mg/dL"), spacer(),
  P(`El modelo explica el ${(g.R2_test * 100).toFixed(1)} % de la variabilidad de la glucosa (R² = ${f4(g.R2_test)}) con un RMSE de ${f2(g.RMSE)} mg/dL. Es un ajuste moderado-bueno: el R² de entrenamiento (${f4(g.R2_train)}) y el de prueba son casi iguales, así que el modelo generaliza bien, pero queda cerca del 32 % de la variabilidad sin explicar (otros factores no incluidos, como genética, dieta o medicación, o relaciones no lineales).`),
  H2("3.3 Importancia de las variables"),
  ...img("glucosa_dispersion.png", 624, "Figura 3. Relación de Edad, IMC y Actividad_Fisica con Nivel_Glucosa."),
  ...img("glucosa_evaluacion.png", 560, "Figura 4. Valores reales vs. predichos y residuos (glucosa)."),
  P([B("Variable de mayor impacto: Edad. "), new TextRun(`Comparando los coeficientes estandarizados, Edad (${g.estandarizados.Edad.toFixed(3)}) supera ampliamente a Actividad_Fisica (${g.estandarizados.Actividad_Fisica.toFixed(3)}) e IMC (${g.estandarizados.IMC.toFixed(3)}). Esto coincide con la correlación simple (0.79) y con el estadístico t más alto del modelo. Nótese que el coeficiente bruto de Actividad_Fisica (${f2(g.coeficientes.Actividad_Fisica)}) es mayor en valor absoluto que el de Edad por estar en otra escala; por eso la importancia debe compararse con los coeficientes estandarizados y no con los brutos.`)]),
);

// ---------- 4. Energía ----------
c.push(new Paragraph({ children: [new PageBreak()] }),
  H1("4. Ejercicio 3: Predicción del consumo de energía eléctrica"),
  P(`El dataset contiene ${e.n.toLocaleString("en-US")} registros con Temperatura (°C), Hora (1 a 24) y Dia_Semana (1 = lunes, …, 7 = domingo) como predictoras y Consumo_Energia (kWh) como variable dependiente. Las correlaciones con el consumo fueron: Temperatura = ${e.correlaciones.Temperatura.toFixed(3)}, Hora = ${e.correlaciones.Hora.toFixed(3)} y Dia_Semana = ${e.correlaciones.Dia_Semana.toFixed(3)}.`),
  H2("4.1 Modelo y coeficientes"),
  P(`Ecuación ajustada: Consumo_Energia = ${f2(e.intercepto)} + ${f2(e.coeficientes.Temperatura)}·Temperatura + ${f2(e.coeficientes.Hora)}·Hora ${f2(e.coeficientes.Dia_Semana)}·Dia_Semana.`),
  coefTable("energia"), spacer(),
  P([B("Interpretación. "), new TextRun(`Temperatura: por cada grado centígrado adicional el consumo aumenta en promedio ${f2(e.coeficientes.Temperatura)} kWh (por ejemplo, mayor uso de climatización). Hora: cada hora avanzada en el día se asocia con ${f2(e.coeficientes.Hora)} kWh más. Dia_Semana: cada incremento de un día en la semana (de lunes hacia domingo) reduce el consumo en ${f2(Math.abs(e.coeficientes.Dia_Semana))} kWh, es decir, el consumo tiende a ser menor hacia el fin de semana. Las tres variables son significativas (p < 0.001).`)]),
  H2("4.2 Variable de mayor impacto"),
  P(`Temperatura es la variable de mayor impacto: su coeficiente estandarizado (${e.estandarizados.Temperatura.toFixed(3)}) es el más alto, seguido de Hora (${e.estandarizados.Hora.toFixed(3)}) y, muy por debajo, Dia_Semana (${e.estandarizados.Dia_Semana.toFixed(3)}). Una desviación estándar de aumento en la temperatura (≈ 5 °C) eleva el consumo en cerca de 0.78 desviaciones estándar. El mismo orden se observa en las correlaciones.`),
  H2("4.3 Desempeño"),
  metTable("energia", "kWh"), spacer(),
  P(`El modelo alcanza un R² de ${f4(e.R2_test)} en prueba (explica ≈ ${(e.R2_test * 100).toFixed(1)} % de la variabilidad) y un RMSE de ${f2(e.RMSE)} kWh, frente a un consumo medio de 400 kWh (error relativo ≈ 5 %). No hay sobreajuste (R² de entrenamiento = ${f4(e.R2_train)}).`),
  H2("4.4 Visualizaciones"),
  ...img("energia_dispersion.png", 624, "Figura 5. Relación de Temperatura, Hora y Dia_Semana con Consumo_Energia."),
  ...img("energia_correlacion.png", 300, "Figura 6. Matriz de correlación (energía)."),
  ...img("energia_evaluacion.png", 560, "Figura 7. Valores reales vs. predichos y residuos (energía)."),
  P([B("Limitación: "), new TextRun("Hora es una variable cíclica (la hora 24 está junto a la hora 1), pero el modelo lineal la trata como una escala continua con efecto constante por hora. Una mejora posible sería codificarla con seno/coseno o como variable categórica, y Dia_Semana como categórica, lo que permitiría capturar patrones diarios y de fin de semana no lineales.")]),
);

// ---------- 5. Exportación ----------
c.push(new Paragraph({ children: [new PageBreak()] }),
  H1("5. Exportación del modelo"),
  P("Para reutilizar un modelo entrenado sin tener que volver a entrenarlo existen dos opciones habituales en Python: pickle (módulo estándar, serializa cualquier objeto) y joblib (pensado para objetos con arreglos NumPy grandes, como los modelos de scikit-learn, con carga eficiente). Se utilizó joblib. Cada archivo .pkl guarda un diccionario con el modelo entrenado, el orden de las variables, los rangos de entrenamiento y las métricas, de modo que la aplicación web pueda validar las entradas."),
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "Guardar:  joblib.dump(paquete, 'modelos/modelo_glucosa.pkl')", font: "Consolas", size: 20 })] }),
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "Cargar:   p = joblib.load('modelos/modelo_glucosa.pkl'); p['modelo'].predict(X)", font: "Consolas", size: 20 })] }),
  spacer(),
  P("Archivos generados: modelo_dolar.pkl, modelo_glucosa.pkl y modelo_energia.pkl. Una advertencia práctica: los archivos pickle/joblib solo deben cargarse si provienen de una fuente de confianza, porque la deserialización puede ejecutar código arbitrario, y deben usarse con la misma versión de scikit-learn con la que se guardaron."),
  H1("6. Interfaz web"),
  P("Se desarrolló una aplicación Flask (app.py) con una única página. El usuario selecciona el ejercicio (Dólar, Glucosa o Energía), ingresa por teclado los valores de las variables independientes y presiona «Predecir»; el servidor carga el modelo .pkl correspondiente y devuelve la predicción en pantalla con su unidad. La aplicación valida que las entradas sean numéricas, advierte cuando un valor está fuera del rango de entrenamiento (extrapolación) y muestra el R² y el RMSE del modelo seleccionado."),
  P([B("Ejecución: "), new TextRun("pip install -r requirements.txt  →  python app.py  →  abrir http://127.0.0.1:5000.")]),
  P("Ejemplos de prueba realizados: Glucosa (Edad = 50, IMC = 25, Actividad = 4) → 142.18 mg/dL; Energía (Temperatura = 25 °C, Hora = 12, Dia_Semana = 3) → 401.25 kWh; Dólar (Dia = 501, Inflacion = 0.02, Tasa_interes = 5) → 6,458.64, con aviso de extrapolación por estar el día 501 fuera del rango 1–500."),
);

// ---------- 7. Conclusiones ----------
c.push(H1("7. Conclusiones"),
  bullet(`Dólar: el modelo es casi perfecto (R² = ${f4(d.R2_test)}, RMSE ≈ ${f2(d.RMSE)}), pero se debe a que el precio sigue una tendencia lineal en el tiempo; la inflación y la tasa de interés no resultaron significativas.`),
  bullet(`Glucosa: el modelo explica ≈ ${(g.R2_test * 100).toFixed(0)} % de la variabilidad (RMSE = ${f2(g.RMSE)} mg/dL). La edad es el factor más influyente; el IMC eleva y la actividad física reduce la glucosa, y los tres efectos son significativos.`),
  bullet(`Energía: R² = ${f4(e.R2_test)} y RMSE = ${f2(e.RMSE)} kWh. La temperatura es la variable con mayor impacto, seguida de la hora del día; el día de la semana tiene un efecto pequeño pero significativo.`),
  bullet("La regresión lineal múltiple es un modelo simple e interpretable que funcionó bien en los tres casos; sus principales limitaciones son el supuesto de linealidad y de efecto constante (relevante para la variable cíclica Hora) y el hecho de que las asociaciones encontradas no implican causalidad."),
  bullet("Se logró cubrir el ciclo CRISP-DM completo: desde la comprensión de los datos hasta el despliegue en una aplicación web funcional que consume los modelos exportados."),
  H1("8. Archivos entregados"),
  bullet("entrenamiento.py — código Python de los tres modelos (entrenamiento, métricas, gráficas y exportación)."),
  bullet("modelos/modelo_dolar.pkl, modelo_glucosa.pkl, modelo_energia.pkl — modelos exportados."),
  bullet("graficas/ — visualizaciones de cada ejercicio."),
  bullet("app.py — interfaz web Flask; requirements.txt y README.md — instrucciones de ejecución."),
);

const doc = new Document({
  creator: "Laboratorio Minería de Datos",
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: "Calibri", color: "2B6CB0" }, paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: "Calibri", color: "2C5282" }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }] },
  sections: [{ properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } }, children: c }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync("Informe_Laboratorio_1.docx", b); console.log("ok"); });
