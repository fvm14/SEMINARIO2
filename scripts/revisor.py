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
<html lang="es"><head><meta charset="utf-8"><title>Revisor</title>
<style>img{max-width:100%;max-height:72vh;display:block}body{margin:8px;font-family:sans-serif;font-size:14px}</style>
</head><body>
<div>
Grupo: <select id="grupo"></select>
Estado: <select id="filtro"><option value="pend">Pendientes</option><option value="todas">Todas</option><option value="aprobada">Aprobadas</option><option value="rechazada">Rechazadas</option><option value="corregir">Para corregir</option></select>
<button onclick="exportar()">Exportar CSV</button>
<span id="cuentas"></span>
</div>
<div id="ficha"></div>
<div id="foto"></div>
<div>
<button onclick="atras()">Anterior (Z)</button>
<button onclick="decidir('rechazada')">Rechazar (R)</button>
<button onclick="decidir('corregir')">Corregir despues (C)</button>
<button onclick="saltar()">Saltar (S)</button>
<button onclick="decidir('aprobada')">Aprobar (A)</button>
</div>
<p>Izquierda: original. Derecha: pseudo-etiqueta (verde = palta, rojo = defecto). Aprobar si el contorno verde sigue la palta y cada zona roja es un defecto real. Rechazar si falta un defecto evidente o sobra uno grande.</p>
<script>
const D = __DATOS__;
const KEY = "revision_pseudo_" + D.id;
const GRUPOS = [["control", "Control"], ["rectangular", "Rectangulares"], ["dudosa", "Dudosas"], ["todos", "Todos"]];
let dec = {}; try { dec = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) {}
let historial = [], i = 0;
const $ = id => document.getElementById(id);
const guardar = () => { try { localStorage.setItem(KEY, JSON.stringify(dec)); } catch(e) {} };
const delGrupo = () => { const g = $("grupo").value; return g === "todos" ? D.filas : D.filas.filter(r => r.grupo === g); };
function vista(){ const f = $("filtro").value;
  return delGrupo().filter(r => f === "todas" ? true : f === "pend" ? !dec[r.archivo] : dec[r.archivo] === f); }
function pintar(){
  const G = delGrupo(), v = vista();
  const n = e => G.filter(r => dec[r.archivo] === e).length;
  const ap = n("aprobada"), re = n("rechazada"), co = n("corregir"), hechas = ap + re + co;
  $("cuentas").textContent = ` Revisadas ${hechas}/${G.length} | aprobadas ${ap} | rechazadas ${re} | corregir ${co} | aprobacion ${hechas ? (100 * ap / hechas).toFixed(1) + "%" : "-"}`;
  if (!v.length) { $("foto").textContent = "No quedan imagenes con este grupo y estado."; $("ficha").textContent = ""; return; }
  if (i >= v.length) i = v.length - 1; if (i < 0) i = 0;
  const r = v[i], d = dec[r.archivo];
  $("ficha").textContent = `${i + 1}/${v.length} | ${r.archivo} | madurez ${r.madurez} | defectos ${r.n_defectos} | ${r.grupo}${r.motivo ? " | " + r.motivo : ""} | decision: ${d || "pendiente"}`;
  $("foto").innerHTML = `<img src="revision/${encodeURIComponent(r.archivo)}" alt="">`;
  const sig = v[i + 1]; if (sig) { const p = new Image(); p.src = "revision/" + encodeURIComponent(sig.archivo); }
}
function decidir(x){ const v = vista(); if (!v.length) return; const r = v[i];
  historial.push([r.archivo, dec[r.archivo]]); dec[r.archivo] = x; guardar();
  if ($("filtro").value !== "pend") i++; pintar(); }
function saltar(){ i++; pintar(); }
function atras(){ const h = historial.pop(); if (h) { if (h[1]) dec[h[0]] = h[1]; else delete dec[h[0]]; guardar();
  if ($("filtro").value !== "pend") i = Math.max(0, i - 1); else i = 0; } else i = Math.max(0, i - 1); pintar(); }
function exportar(){ const filas = ["archivo,decision"].concat(Object.entries(dec).map(([a, d]) => `${a},${d}`));
  const b = new Blob([filas.join("\n") + "\n"], {type: "text/csv"}); const a = document.createElement("a");
  a.href = URL.createObjectURL(b); a.download = "revision_decisiones.csv"; a.click(); }
$("grupo").innerHTML = GRUPOS.map(([k, t]) => { const c = k === "todos" ? D.filas.length : D.filas.filter(r => r.grupo === k).length;
  return c ? `<option value="${k}">${t} (${c})</option>` : ""; }).join("");
$("grupo").onchange = $("filtro").onchange = () => { i = 0; pintar(); };
document.addEventListener("keydown", e => { if (e.target.tagName === "SELECT") return; const k = e.key.toLowerCase();
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
              "madurez": int(r.madurez), "n_defectos": int(r.n_defectos),
              "grupo": ["control", "rectangular", "dudosa"][int(r.grupo)]} for r in d.itertuples()]
    ident = f"{len(filas)}_{filas[0]['archivo'] if filas else ''}"
    html = HTML.replace("__DATOS__", json.dumps({"id": ident, "filas": filas}, ensure_ascii=False))
    salida = os.path.join(args.dir, "revision.html")
    with open(salida, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Herramienta de revision: {salida}  ({len(filas)} imagenes: "
          f"{int((d.estado == 'revisar').sum())} dudosas + {int((d.estado == 'auto').sum())} de control)")


if __name__ == "__main__":
    main()
