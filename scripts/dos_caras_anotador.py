"""
dos_caras_anotador.py
Genera data/dos_caras/anotador.html: pagina local para anotar los defectos del
lado faltante sin instalar nada (se abre con doble clic en Chrome o Edge).

Cada imagen trae el contorno de la palta pre-anotado. Se dibujan los defectos
arrastrando el mouse alrededor de cada uno; al terminar se exporta un JSON que
`dos_caras_labelme.py --importar --anotador <json>` convierte a YOLO-seg.

Uso:
    python scripts/dos_caras_anotador.py
"""

import glob
import json
import os

import cv2

DIR = os.path.join("data", "dos_caras")
IMG = os.path.join(DIR, "por_anotar", "images")
LBL = os.path.join(DIR, "por_anotar", "labels")

HTML = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>Anotador de defectos</title>
<style>
body{font-family:sans-serif;margin:8px;font-size:14px}
#barra{margin-bottom:6px}
#lienzo{border:1px solid #000;cursor:crosshair}
#caja{overflow:auto;max-height:calc(100vh - 110px);display:inline-block}
button{margin-right:4px}
</style></head><body>
<div id="barra">
<button id="ant">&lt; Anterior (A)</button>
<button id="sig">Siguiente (D) &gt;</button>
<button id="lista">Marcar lista (Enter)</button>
<button id="deshacer">Borrar ultimo defecto (Z)</button>
<button id="modo">Modo: defecto (P cambia a palta)</button>
<button id="zoom">Zoom (+)</button>
<button id="exportar">Exportar JSON</button>
<label>Cargar JSON <input type="file" id="cargar" accept=".json"></label>
<div id="info"></div>
</div>
<div id="caja"><canvas id="lienzo"></canvas></div>
<p>Arrastra el mouse alrededor de cada defecto (se cierra al soltar). Clic derecho sobre un defecto lo borra.
Si el fruto esta sano, solo marca la imagen como lista. Todo se guarda en el navegador; al terminar, exporta el JSON.</p>
<script>
const DATOS = __DATOS__;
const CLAVE = "anotador_dos_caras_v1";
let estado = {};
try { estado = JSON.parse(localStorage.getItem(CLAVE) || "{}"); } catch (e) { estado = {}; }
for (const d of DATOS) {
  if (!estado[d.archivo]) estado[d.archivo] = {palta: d.palta, defectos: [], lista: false};
}
let i = 0, escala = 1, modo = "defecto", trazo = null;
const cv = document.getElementById("lienzo"), cx = cv.getContext("2d");
const img = new Image();
function guardar() { try { localStorage.setItem(CLAVE, JSON.stringify(estado)); } catch (e) {} }
function actual() { return estado[DATOS[i].archivo]; }
function cargar() {
  img.onload = () => { ajustar(); dibujar(); };
  img.src = "por_anotar/images/" + DATOS[i].archivo;
}
function ajustar() {
  const base = Math.min(1, (window.innerHeight - 120) / img.height);
  cv.width = img.width * base * escala; cv.height = img.height * base * escala;
  cv.dataset.k = base * escala;
}
function poligono(p, borde, relleno) {
  if (!p || p.length < 2) return;
  const k = +cv.dataset.k;
  cx.beginPath(); cx.moveTo(p[0][0] * k, p[0][1] * k);
  for (const q of p.slice(1)) cx.lineTo(q[0] * k, q[1] * k);
  cx.closePath(); cx.fillStyle = relleno; cx.fill(); cx.strokeStyle = borde; cx.lineWidth = 2; cx.stroke();
}
function dibujar() {
  const a = actual();
  cx.drawImage(img, 0, 0, cv.width, cv.height);
  poligono(a.palta, "#00c000", "rgba(0,0,0,0)");
  for (const d of a.defectos) poligono(d, "#ff0000", "rgba(255,0,0,0.25)");
  if (trazo) poligono(trazo, modo === "palta" ? "#00c000" : "#ff0000", "rgba(0,0,0,0)");
  const listas = Object.values(estado).filter(e => e.lista).length;
  document.getElementById("info").textContent =
    `Imagen ${i + 1} de ${DATOS.length}: ${DATOS[i].archivo} | defectos: ${a.defectos.length} | ` +
    `${a.lista ? "LISTA" : "pendiente"} | listas: ${listas} de ${DATOS.length}`;
  document.getElementById("modo").textContent = "Modo: " + modo + (modo === "defecto" ? " (P cambia a palta)" : " (P vuelve a defecto)");
}
function punto(e) { const r = cv.getBoundingClientRect(), k = +cv.dataset.k; return [(e.clientX - r.left) / k, (e.clientY - r.top) / k]; }
function dentro(p, q) {
  let c = false;
  for (let a = 0, b = q.length - 1; a < q.length; b = a++) {
    if ((q[a][1] > p[1]) !== (q[b][1] > p[1]) && p[0] < (q[b][0] - q[a][0]) * (p[1] - q[a][1]) / (q[b][1] - q[a][1]) + q[a][0]) c = !c;
  }
  return c;
}
cv.addEventListener("mousedown", e => { if (e.button === 0) { trazo = [punto(e)]; } });
cv.addEventListener("mousemove", e => {
  if (!trazo) return;
  const p = punto(e), u = trazo[trazo.length - 1];
  if (Math.hypot(p[0] - u[0], p[1] - u[1]) > 2) { trazo.push(p); dibujar(); }
});
window.addEventListener("mouseup", () => {
  if (!trazo) return;
  if (trazo.length >= 3) {
    if (modo === "palta") actual().palta = trazo; else actual().defectos.push(trazo);
    guardar();
  }
  trazo = null; dibujar();
});
cv.addEventListener("contextmenu", e => {
  e.preventDefault();
  const p = punto(e), a = actual();
  for (let k = a.defectos.length - 1; k >= 0; k--) if (dentro(p, a.defectos[k])) { a.defectos.splice(k, 1); guardar(); dibujar(); return; }
});
function ir(d) { i = Math.max(0, Math.min(DATOS.length - 1, i + d)); cargar(); }
function marcarLista() { actual().lista = true; guardar(); ir(1); }
document.getElementById("ant").onclick = () => ir(-1);
document.getElementById("sig").onclick = () => ir(1);
document.getElementById("lista").onclick = marcarLista;
document.getElementById("deshacer").onclick = () => { actual().defectos.pop(); guardar(); dibujar(); };
document.getElementById("modo").onclick = () => { modo = modo === "defecto" ? "palta" : "defecto"; dibujar(); };
document.getElementById("zoom").onclick = () => { escala = escala >= 2 ? 1 : escala + 0.5; ajustar(); dibujar(); };
document.getElementById("exportar").onclick = () => {
  const salida = {version: 1, imagenes: {}};
  for (const d of DATOS) {
    const e = estado[d.archivo];
    salida.imagenes[d.archivo] = {ancho: d.ancho, alto: d.alto, palta: e.palta, defectos: e.defectos, lista: e.lista};
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(salida)], {type: "application/json"}));
  a.download = "anotaciones_dos_caras.json"; a.click();
};
document.getElementById("cargar").onchange = ev => {
  const f = ev.target.files[0]; if (!f) return;
  f.text().then(t => {
    const j = JSON.parse(t);
    for (const [k, v] of Object.entries(j.imagenes || {})) if (estado[k]) estado[k] = {palta: v.palta, defectos: v.defectos, lista: v.lista};
    guardar(); dibujar();
  });
};
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  const t = e.key.toLowerCase();
  if (t === "a") ir(-1);
  else if (t === "d") ir(1);
  else if (t === "enter") marcarLista();
  else if (t === "z") { actual().defectos.pop(); guardar(); dibujar(); }
  else if (t === "p") { modo = modo === "defecto" ? "palta" : "defecto"; dibujar(); }
  else if (t === "+") { escala = escala >= 2 ? 1 : escala + 0.5; ajustar(); dibujar(); }
});
window.addEventListener("resize", () => { ajustar(); dibujar(); });
const primera = DATOS.findIndex(d => !estado[d.archivo].lista);
i = primera < 0 ? 0 : primera;
cargar();
</script></body></html>
"""


def main():
    datos = []
    for ruta in sorted(glob.glob(os.path.join(IMG, "*.jpg"))):
        nombre = os.path.basename(ruta)
        alto, ancho = cv2.imread(ruta).shape[:2]
        palta = []
        txt = os.path.join(LBL, nombre[:-4] + ".txt")
        if os.path.exists(txt):
            for linea in open(txt):
                v = linea.split()
                if len(v) >= 7 and v[0] == "0":
                    xy = [float(t) for t in v[1:]]
                    palta = [[round(xy[k] * ancho, 1), round(xy[k + 1] * alto, 1)] for k in range(0, len(xy), 2)]
        datos.append({"archivo": nombre, "ancho": ancho, "alto": alto, "palta": palta})
    salida = os.path.join(DIR, "anotador.html")
    with open(salida, "w", encoding="utf-8") as f:
        f.write(HTML.replace("__DATOS__", json.dumps(datos)))
    print(f"{salida} con {len(datos)} imagenes")


if __name__ == "__main__":
    main()
