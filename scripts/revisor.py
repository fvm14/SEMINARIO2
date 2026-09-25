"""
revisor.py
Genera data/pseudo/revision.html: herramienta para aprobar o rechazar
pseudo-etiquetas en el navegador (doble clic, funciona sin internet).

Teclas:  A o flecha derecha = aprobar  |  R = rechazar  |  C = corregir despues  |  S = saltar
         Z o Retroceso = volver a la anterior
Tu avance se guarda solo en el navegador; puedes cerrar y seguir otro dia.
Al terminar (o cuando quieras), boton "Exportar decisiones" -> descarga
revision_decisiones.csv; muevelo a data/pseudo/.

Criterio para aprobar: el contorno verde sigue el borde de la palta y cada
zona roja es un defecto real (sin inventar ni dejar fuera defectos
importantes). Si falta un defecto evidente o sobra uno grande, rechaza.

Uso:
    python scripts/revisor.py            (despues de generar_pseudoetiquetas.py)
"""

import argparse
import json
import os

import pandas as pd

HTML = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>Revision de pseudo-etiquetas</title>
<style>
:root{--bd:#dde1e7;--tx:#1d2330;--sub:#5d6675;--ok:#2e7d32;--bad:#c62828}
*{box-sizing:border-box}body{margin:0;font-family:Segoe UI,Arial,sans-serif;background:#f3f4f6;color:var(--tx)}
header{background:#fff;border-bottom:1px solid var(--bd);padding:10px 20px;display:flex;flex-wrap:wrap;gap:16px;align-items:center}
h1{font-size:17px;margin:0}.sub{color:var(--sub);font-size:13px}
.barra{flex:1;min-width:200px;height:10px;background:#e5e7eb;border-radius:5px;overflow:hidden}
.barra div{height:100%;background:#1d2330}
main{display:flex;flex-direction:column;align-items:center;padding:14px}
#img{max-width:98vw;max-height:70vh;border:1px solid var(--bd);background:#fff}
.meta{margin:10px 0;font-size:15px;text-align:center}
.meta .motivo{color:#b45309;font-weight:600}
.botones{display:flex;gap:10px;margin-top:6px}
button{font-size:15px;padding:9px 18px;border-radius:8px;border:1px solid var(--bd);background:#fff;cursor:pointer}
.co{background:#b45309;color:#fff;border-color:#b45309}.estado.corregir{color:#b45309}
.ap{background:var(--ok);color:#fff;border-color:var(--ok)}.re{background:var(--bad);color:#fff;border-color:var(--bad)}
.estado{font-weight:700}.estado.aprobada{color:var(--ok)}.estado.rechazada{color:var(--bad)}
.fin{padding:60px;font-size:18px;text-align:center}
select{font-size:13px;padding:4px}
</style></head><body>
<header><h1>Revision de pseudo-etiquetas</h1>
<div class="barra"><div id="prog"></div></div>
<span class="sub" id="cont"></span>
<select id="filtro"><option value="pend">Pendientes</option><option value="todas">Todas</option><option value="aprobada">Aprobadas</option><option value="rechazada">Rechazadas</option><option value="corregir">Para corregir</option></select>
<button onclick="exportar()">Exportar decisiones</button></header>
<main id="main"></main>
<script>
const D = __DATOS__;
const KEY = "revision_pseudo_" + D.id;
let dec = {}; try { dec = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) {}
let historial = [], i = 0;
const guardar = () => { try { localStorage.setItem(KEY, JSON.stringify(dec)); } catch(e) {} };
function vista(){ const f = document.getElementById("filtro").value;
  return D.filas.filter(r => f === "todas" ? true : f === "pend" ? !dec[r.archivo] : dec[r.archivo] === f); }
function pintar(){
  const v = vista(), hechas = Object.keys(dec).length, ap = Object.values(dec).filter(x => x === "aprobada").length;
  document.getElementById("prog").style.width = (100*hechas/D.filas.length) + "%";
  const co = Object.values(dec).filter(x => x === "corregir").length;
  document.getElementById("cont").textContent = `${hechas}/${D.filas.length} revisadas | ${ap} aprobadas | ${hechas-ap-co} rechazadas | ${co} para corregir`;
  if (!v.length) { document.getElementById("main").innerHTML = '<div class="fin">No quedan imagenes en este filtro. Pulsa "Exportar decisiones" y mueve el archivo a data/pseudo/.</div>'; return; }
  if (i >= v.length) i = v.length - 1; if (i < 0) i = 0;
  const r = v[i], d = dec[r.archivo];
  document.getElementById("main").innerHTML = `
    <img id="img" src="revision/${encodeURIComponent(r.archivo)}">
    <div class="meta">${i+1}/${v.length} en este filtro &nbsp;|&nbsp; <b>${r.archivo}</b> &nbsp;|&nbsp; madurez ${r.madurez}
      &nbsp;|&nbsp; ${r.n_defectos} defecto(s) &nbsp;|&nbsp; ${r.estado === "auto" ? "<b>CONTROL</b>: aprobada por el modelo" : `<span class="motivo">${r.motivo}</span>`}
      ${d ? `&nbsp;|&nbsp; <span class="estado ${d}">${d}</span>` : ""}</div>
    <div class="sub">Izquierda: foto original. Derecha: pseudo-etiqueta (verde = palta, rojo = defecto).</div>
    <div class="botones"><button onclick="atras()">&larr; Anterior (Z)</button>
      <button class="re" onclick="decidir('rechazada')">Rechazar (R)</button>
      <button class="co" onclick="decidir('corregir')">Corregir despues (C)</button>
      <button onclick="saltar()">Saltar (S)</button>
      <button class="ap" onclick="decidir('aprobada')">Aprobar (A)</button></div>`;
  const sig = v[i+1]; if (sig) { const p = new Image(); p.src = "revision/" + encodeURIComponent(sig.archivo); }
}
function decidir(x){ const v = vista(); if (!v.length) return; const r = v[i];
  historial.push([r.archivo, dec[r.archivo]]); dec[r.archivo] = x; guardar();
  if (document.getElementById("filtro").value !== "pend") i++; pintar(); }
function saltar(){ i++; pintar(); }
function atras(){ const h = historial.pop(); if (h) { if (h[1]) dec[h[0]] = h[1]; else delete dec[h[0]]; guardar();
  const f = document.getElementById("filtro").value; if (f !== "pend") i = Math.max(0, i-1); else i = 0; } else i = Math.max(0, i-1); pintar(); }
function exportar(){ const filas = ["archivo,decision"].concat(Object.entries(dec).map(([a, d]) => `${a},${d}`));
  const b = new Blob([filas.join("\n") + "\n"], {type: "text/csv"}); const a = document.createElement("a");
  a.href = URL.createObjectURL(b); a.download = "revision_decisiones.csv"; a.click(); }
document.getElementById("filtro").onchange = () => { i = 0; pintar(); };
document.addEventListener("keydown", e => { const k = e.key.toLowerCase();
  if (k === "a" || e.key === "ArrowRight") decidir("aprobada");
  else if (k === "r") decidir("rechazada"); else if (k === "c") decidir("corregir"); else if (k === "s") saltar();
  else if (k === "z" || e.key === "Backspace" || e.key === "ArrowLeft") { e.preventDefault(); atras(); } });
pintar();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join("data", "pseudo"))
    args = ap.parse_args()
    d = pd.read_csv(os.path.join(args.dir, "pseudoetiquetas.csv"))
    d = d[d.en_revision].copy()
    # Orden: 1) muestra de control de las automaticas (mide su error),
    # 2) defectos rectangulares, 3) resto de dudosas en orden aleatorio intercalando
    # niveles de madurez, para que una revision parcial sea representativa.
    d["grupo"] = 2
    d.loc[d.estado == "auto", "grupo"] = 0
    d.loc[d.motivo.fillna("").str.contains("rectangular"), "grupo"] = 1
    d["azar"] = d.sample(frac=1, random_state=42).reset_index().reset_index().set_index("index")["level_0"]
    d["turno"] = d.groupby(["grupo", "madurez"]).azar.rank()
    d = d.sort_values(["grupo", "turno", "madurez"])
    filas = [{"archivo": r.archivo, "estado": r.estado, "motivo": r.motivo if isinstance(r.motivo, str) else "",
              "madurez": int(r.madurez), "n_defectos": int(r.n_defectos)} for r in d.itertuples()]
    ident = f"{len(filas)}_{filas[0]['archivo'] if filas else ''}"
    html = HTML.replace("__DATOS__", json.dumps({"id": ident, "filas": filas}, ensure_ascii=False))
    salida = os.path.join(args.dir, "revision.html")
    with open(salida, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Herramienta de revision: {salida}  ({len(filas)} imagenes: "
          f"{int((d.estado == 'revisar').sum())} dudosas + {int((d.estado == 'auto').sum())} de control)")


if __name__ == "__main__":
    main()
