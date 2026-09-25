"""
visor.py
Genera visor.html dentro de una carpeta de evaluacion (eval_test_* o eval_val_*)
para revisar en el navegador todas las predicciones del modelo: mascaras
(real vs prediccion), madurez y categoria OCDE, con filtros por tipo de error.

Abrir el visor: doble clic en <carpeta de evaluacion>/visor.html
(funciona sin internet ni servidor; los datos van incrustados en el HTML).

Uso:
    python scripts/visor.py --eval resultados/yolo_multitarea/<modelo>/eval_test_...
Para tener TODAS las imagenes, la evaluacion debe correrse con --n-vis alto, por ejemplo:
    python scripts/eval_yolo_multitarea.py --weights ... --conf-defecto 0.15 --n-vis 500
"""

import argparse
import json
import os

import pandas as pd

HTML = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>Visor de predicciones</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--tx:#1d2330;--sub:#5d6675;--ok:#2e7d32;--bad:#c62828;--bd:#dde1e7}
*{box-sizing:border-box}body{margin:0;font-family:Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--tx)}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid var(--bd);padding:12px 20px;z-index:5}
h1{font-size:18px;margin:0 0 8px}.sub{color:var(--sub);font-size:13px}
.filtros{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
.filtros button{border:1px solid var(--bd);background:#fff;border-radius:16px;padding:5px 12px;cursor:pointer;font-size:13px}
.filtros button.on{background:#1d2330;color:#fff;border-color:#1d2330}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(640px,1fr));gap:14px;padding:16px 20px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;overflow:hidden;cursor:pointer}
.card img{width:100%;display:block;background:#eee}
.info{padding:8px 10px;font-size:13px;display:grid;grid-template-columns:1fr 1fr;gap:4px}
.nom{grid-column:1/3;color:var(--sub);font-size:11px;word-break:break-all}
.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}
#modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:10;align-items:center;justify-content:center;flex-direction:column}
#modal img{max-width:96vw;max-height:84vh}#modal .cap{color:#fff;margin-top:10px;font-size:15px;text-align:center}
.vacio{padding:40px;color:var(--sub)}
</style></head><body>
<header><h1>Visor de predicciones</h1>
<div class="sub" id="resumen"></div>
<div class="sub">Cada imagen: original | anotacion real | prediccion del modelo. Verde = palta, rojo = defecto. Clic para ampliar; flechas para navegar; Esc para cerrar.</div>
<div class="filtros" id="filtros"></div></header>
<main id="grid"></main>
<div id="modal"><img id="mimg"><div class="cap" id="mcap"></div></div>
<script>
const D = __DATOS__;
const F = {
  "Todas": r => true,
  "Error de madurez": r => r.madurez_real !== r.madurez_pred,
  "Error de madurez > 1 nivel": r => Math.abs(r.madurez_real - r.madurez_pred) > 1,
  "Error OCDE": r => r.ocde_real !== r.ocde_pred,
  "Rechazada que pasa como Cat. I": r => r.ocde_real === "Rechazado" && r.ocde_pred === "Categoria I",
  "Con defecto real": r => r.ratio_real > 0,
  "Todo correcto": r => r.madurez_real === r.madurez_pred && r.ocde_real === r.ocde_pred,
};
let actual = "Todas", lista = [], idx = 0;
const fmt = x => (x*100).toFixed(1) + "%";
function tarjeta(r, i){
  const mOk = r.madurez_real === r.madurez_pred, oOk = r.ocde_real === r.ocde_pred;
  return `<div class="card" onclick="abrir(${i})">
    ${r.vis ? `<img loading="lazy" src="visualizaciones/${encodeURIComponent(r.vis)}">` : `<div class="vacio">Sin imagen (correr eval con --n-vis mayor)</div>`}
    <div class="info">
      <span>Madurez real: <b>${r.madurez_real}</b></span>
      <span class="${mOk?'ok':'bad'}">Prediccion: ${r.madurez_pred} (${Math.round(r.confianza_madurez*100)}%)</span>
      <span>OCDE real: <b>${r.ocde_real}</b> (${fmt(r.ratio_real)})</span>
      <span class="${oOk?'ok':'bad'}">Prediccion: ${r.ocde_pred} (${fmt(r.ratio_pred)})</span>
      <span class="nom">${r.archivo}</span></div></div>`;
}
function pintar(){
  lista = D.filas.filter(F[actual]);
  document.getElementById("grid").innerHTML = lista.length ? lista.map(tarjeta).join("") : '<div class="vacio">Ninguna imagen en este filtro.</div>';
  document.querySelectorAll("#filtros button").forEach(b => b.classList.toggle("on", b.dataset.f === actual));
}
function abrir(i){ idx = i; const r = lista[i]; if(!r.vis) return;
  document.getElementById("mimg").src = "visualizaciones/" + encodeURIComponent(r.vis);
  document.getElementById("mcap").textContent = `${i+1}/${lista.length}  |  ${r.archivo}  |  madurez ${r.madurez_real} -> ${r.madurez_pred}  |  OCDE ${r.ocde_real} -> ${r.ocde_pred}`;
  document.getElementById("modal").style.display = "flex"; }
document.getElementById("modal").onclick = () => document.getElementById("modal").style.display = "none";
document.addEventListener("keydown", e => { const m = document.getElementById("modal");
  if (m.style.display !== "flex") return;
  if (e.key === "Escape") m.style.display = "none";
  if (e.key === "ArrowRight" && idx < lista.length-1) abrir(idx+1);
  if (e.key === "ArrowLeft" && idx > 0) abrir(idx-1); });
document.getElementById("filtros").innerHTML = Object.keys(F).map(k =>
  `<button data-f="${k}" onclick="actual='${k}';pintar()">${k} (${D.filas.filter(F[k]).length})</button>`).join("");
const n = D.filas.length, mOk = D.filas.filter(r => r.madurez_real === r.madurez_pred).length, oOk = D.filas.filter(r => r.ocde_real === r.ocde_pred).length;
document.getElementById("resumen").textContent = `${D.titulo}  |  ${n} imagenes  |  madurez correcta: ${mOk} (${fmt(mOk/n)})  |  OCDE correcta: ${oOk} (${fmt(oOk/n)})  |  con imagen: ${D.filas.filter(r=>r.vis).length}`;
pintar();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    args = ap.parse_args()
    d = pd.read_csv(os.path.join(args.eval, "predicciones.csv"))
    vis_dir = os.path.join(args.eval, "visualizaciones")
    visibles = set(os.listdir(vis_dir)) if os.path.isdir(vis_dir) else set()
    filas = []
    for i, r in d.iterrows():
        nombre = f"{i:02d}_{os.path.splitext(r.archivo)[0][:40]}.jpg"
        fila = {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
        fila["vis"] = nombre if nombre in visibles else None
        filas.append(fila)
    titulo = os.path.relpath(os.path.abspath(args.eval))
    html = HTML.replace("__DATOS__", json.dumps({"titulo": titulo, "filas": filas}, ensure_ascii=False))
    salida = os.path.join(args.eval, "visor.html")
    with open(salida, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Visor generado: {salida}  ({sum(1 for x in filas if x['vis'])}/{len(filas)} con imagen)")


if __name__ == "__main__":
    main()
