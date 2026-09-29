"""
probar_modelo_gui.py
Ventana (tkinter) para probar los modelos con fotos, sin emulador.

- Lista desplegable con los modelos encontrados en el proyecto (multitarea en
  PyTorch, TFLite u ONNX, y los de tarea unica de Seminario I / linea base).
- Tres paneles: original | anotacion real (si la foto tiene poligonos en
  data/raw/labels_seg) | prediccion del modelo.
- Por foto: categoria OCDE y % de defecto (predicho vs real), IoU, precision y
  recall de defecto por pixel, madurez predicha vs real y la probabilidad de
  cada nivel, y el tiempo.
- Metricas acumuladas de las fotos analizadas (o de toda la carpeta con
  "Evaluar carpeta"): madurez (accuracy, F1 macro, +-1 nivel, matriz de
  confusion) y defectos (IoU por pixel, acierto OCDE, recall de Rechazado,
  MAE del ratio), mas la latencia media.

Uso:
    .venv\\Scripts\\python scripts\\probar_modelo_gui.py --fotos data\\prueba_test
Teclas: flecha derecha/izquierda = siguiente/anterior foto.
"""

import argparse
import base64
import glob
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inferencia_movil import cargar_backend, letterbox, postprocesar  # noqa: E402
from ocde import calcular_ratio, clasificar_ocde, filtrar_defecto_por_roi, fruto_completo, CATEGORIAS_OCDE  # noqa: E402
from rasterize_utils import poligonos_a_mascaras  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS_SEG = os.path.join(RAIZ, "data", "raw", "labels_seg")
S1_TRAIN = os.path.join(RAIZ, "resultados", "linea_base_s1", "s1_train_stems.txt")
LADO_VISTA = 380
CONF_PALTA = 0.35
COLORES_OCDE = {"Categoria I": "#2E7D32", "Categoria II": "#B26A00", "Rechazado": "#C62828"}

# Modelos conocidos: (ruta relativa o patron, nombre visible, umbral de defecto calibrado en val)
CATALOGO = [
    ("modelos_movil/*dynamic_range*.tflite", "Multitarea Ronda 1 · TFLite dynamic range (el de la app)", 0.05),
    ("weights/ronda1_best.pt", "Multitarea Ronda 1 · PyTorch original", 0.05),
    ("modelos_movil/*float16*.tflite", "Multitarea Ronda 1 · TFLite FP16", 0.05),
    ("modelos_movil/*integer_quant*.tflite", "Multitarea Ronda 1 · TFLite INT8", 0.05),
    ("modelos_movil/*.onnx", "Multitarea Ronda 1 · ONNX FP32", 0.05),
    ("weights/limpio_best.pt", "Multitarea limpio (generó las pseudoetiquetas)", 0.15),
    ("weights/s1_defectos_best.pt", "Seminario I · solo defectos (split viejo: vio 77/108 fotos de test)", 0.20),
    ("resultados/linea_base_s1/yolov8s_replica_s1/weights/best.pt", "Línea base · réplica de Seminario I (split por fruto)", None),
    ("resultados/linea_base_s1/yolov8s_ablacion/weights/best.pt", "Línea base · ablación sin madurez (split por fruto)", None),
]


def buscar_modelos():
    """Devuelve [(nombre, ruta, umbral)] con los modelos del catalogo que existen, mas otros .pt/.tflite/.onnx sueltos."""
    encontrados, vistos = [], set()
    for patron, nombre, umbral in CATALOGO:
        for ruta in sorted(os.path.normpath(p) for p in glob.glob(os.path.join(RAIZ, patron))):
            if ruta not in vistos:
                if umbral is None:  # linea base: leer el umbral calibrado si ya existe
                    umbral = umbral_calibrado(os.path.dirname(os.path.dirname(ruta)))
                encontrados.append((nombre, ruta, umbral))
                vistos.add(ruta)
    for patron in ("weights/*.pt", "modelos_movil/*.tflite", "modelos_movil/*.onnx"):
        for ruta in sorted(os.path.normpath(p) for p in glob.glob(os.path.join(RAIZ, patron))):
            if ruta not in vistos:
                encontrados.append((os.path.relpath(ruta, RAIZ), ruta, 0.15))
                vistos.add(ruta)
    return encontrados


def umbral_calibrado(carpeta_run):
    ruta = os.path.join(carpeta_run, "calibracion_defecto_val_resumen.csv")
    try:
        import pandas as pd
        r = pd.read_csv(ruta)
        return float(r.sort_values(["iou_defecto", "acc_ocde"], ascending=False).iloc[0].umbral)
    except Exception:
        return 0.15


# ---------------------------------------------------------------- modelos

class ModeloExportado:
    """TFLite u ONNX multitarea: mismo pipeline que la app (inferencia_movil)."""
    tipo = "multitarea exportado"

    def __init__(self, ruta):
        self.backend = cargar_backend(ruta, 4)
        self.tiene_madurez = True

    def predecir(self, img, conf_defecto):
        x, r, pad = letterbox(img, self.backend.imgsz)
        t0 = time.perf_counter()
        det, proto, mad = self.backend(x)
        ms = (time.perf_counter() - t0) * 1000
        palta, defecto = postprocesar(det, proto, img.shape[:2], r, pad, CONF_PALTA, conf_defecto)
        return palta, defecto, np.asarray(mad, dtype=float).ravel(), ms


class ModeloPyTorch:
    """best.pt de Ultralytics: multitarea (con cabezal de madurez) o de tarea unica."""

    def __init__(self, ruta):
        import torch  # noqa: F401
        import yolo_multitarea  # noqa: F401  (registra la clase para cargar el checkpoint)
        from ultralytics import YOLO
        self.modelo = YOLO(ruta)
        self.tiene_madurez = None  # se sabe al primer predict
        self.tipo = "PyTorch"

    def predecir(self, img, conf_defecto):
        import torch
        from yolo_multitarea import buscar_modelo_multitarea
        t0 = time.perf_counter()
        r = self.modelo.predict(img, imgsz=800, conf=min(CONF_PALTA, conf_defecto), retina_masks=True, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000
        probs = None
        try:
            mt = buscar_modelo_multitarea(self.modelo.predictor.model)
            probs = torch.softmax(mt.madurez_logits.float(), dim=1)[0].cpu().numpy()
            self.tiene_madurez = True
        except TypeError:
            self.tiene_madurez = False
        alto, ancho = img.shape[:2]
        palta, defecto = np.zeros((alto, ancho), bool), np.zeros((alto, ancho), bool)
        if r.masks is not None:
            for mk, c, s in zip(r.masks.data.cpu().numpy() > 0.5, r.boxes.cls.cpu().numpy().astype(int),
                                r.boxes.conf.cpu().numpy()):
                if c == 0 and s >= CONF_PALTA:
                    palta |= mk
                elif c == 1 and s >= conf_defecto:
                    defecto |= mk
        self.tipo = "PyTorch multitarea" if self.tiene_madurez else "PyTorch solo defectos (sin madurez)"
        return palta, defecto, probs, ms


def cargar_modelo(ruta):
    return ModeloPyTorch(ruta) if ruta.endswith(".pt") else ModeloExportado(ruta)


# ---------------------------------------------------------------- analisis

def madurez_real(ruta):
    try:
        return int(os.path.basename(ruta).split("_")[4].split(".")[0])
    except (IndexError, ValueError):
        return None


def anotacion_real(ruta, alto, ancho):
    txt = os.path.join(LABELS_SEG, os.path.splitext(os.path.basename(ruta))[0] + ".txt")
    if not os.path.exists(txt):
        return None, None
    p, d = poligonos_a_mascaras(txt, alto, ancho)
    return p.astype(bool), d.astype(bool)


def analizar(modelo, ruta, conf_defecto):
    img = cv2.imread(ruta)
    if img is None:
        raise ValueError(f"No se pudo leer {ruta}")
    palta, defecto, probs, ms = modelo.predecir(img, conf_defecto)
    fruto = fruto_completo(palta, defecto)
    defecto = filtrar_defecto_por_roi(fruto, defecto)
    ratio = calcular_ratio(fruto, defecto)
    res = {"img": img, "fruto": fruto, "defecto": defecto, "ratio": ratio, "ms": ms,
           "ocde": CATEGORIAS_OCDE[clasificar_ocde(ratio)], "probs": probs,
           "madurez": int(np.argmax(probs)) + 1 if probs is not None else None,
           "madurez_real": madurez_real(ruta)}
    gt_p, gt_d = anotacion_real(ruta, *img.shape[:2])
    res["gt_fruto"], res["gt_defecto"] = gt_p, gt_d
    if gt_p is not None:
        res["ratio_real"] = calcular_ratio(gt_p, gt_d)
        res["ocde_real"] = CATEGORIAS_OCDE[clasificar_ocde(res["ratio_real"])]
        res["tp"] = int((defecto & gt_d).sum())
        res["fp"] = int((defecto & ~gt_d).sum())
        res["fn"] = int((~defecto & gt_d).sum())
    return res


def resumen_metricas(resultados):
    """Metricas acumuladas sobre una lista de resultados (sin las mascaras)."""
    txt = [f"Fotos analizadas: {len(resultados)}"]
    mad = [(r["madurez_real"], r["madurez"]) for r in resultados if r["madurez"] and r["madurez_real"]]
    if mad:
        real, pred = np.array(mad).T
        txt += ["", "MADUREZ",
                f"  Accuracy: {np.mean(real == pred) * 100:.1f}%   F1 macro: "
                f"{f1_score(real, pred, labels=[1, 2, 3, 4, 5], average='macro', zero_division=0):.3f}   "
                f"±1 nivel: {np.mean(np.abs(real - pred) <= 1) * 100:.1f}%",
                "  Matriz de confusión (filas real 1-5, columnas predicho 1-5):"]
        for i, fila in enumerate(confusion_matrix(real, pred, labels=[1, 2, 3, 4, 5]), 1):
            txt.append(f"    {i}: " + " ".join(f"{v:3d}" for v in fila))
    elif resultados:
        txt += ["", "MADUREZ: este modelo no la predice"]
    con_gt = [r for r in resultados if "ocde_real" in r]
    if con_gt:
        tp, fp, fn = (sum(r[k] for r in con_gt) for k in ("tp", "fp", "fn"))
        rech = [r for r in con_gt if r["ocde_real"] == "Rechazado"]
        txt += ["", f"DEFECTOS ({len(con_gt)} fotos con anotación)",
                f"  IoU por píxel: {tp / max(1, tp + fp + fn):.3f}   precisión: {tp / max(1, tp + fp):.3f}   "
                f"recall: {tp / max(1, tp + fn):.3f}",
                f"  Acierto OCDE: {np.mean([r['ocde'] == r['ocde_real'] for r in con_gt]) * 100:.1f}%   "
                f"MAE del ratio: {np.mean([abs(r['ratio'] - r['ratio_real']) for r in con_gt]) * 100:.2f} pts",
                f"  Recall de Rechazado: " + (f"{np.mean([r['ocde'] == 'Rechazado' for r in rech]) * 100:.1f}% "
                                              f"({sum(r['ocde'] == 'Rechazado' for r in rech)} de {len(rech)})"
                                              if rech else "sin rechazados reales")]
    if resultados:
        txt += ["", f"Latencia media: {np.mean([r['ms'] for r in resultados]):.0f} ms por foto"]
    return "\n".join(txt)


# ---------------------------------------------------------------- dibujo

def a_photoimage(img_bgr):
    h, w = img_bgr.shape[:2]
    r = LADO_VISTA / max(h, w)
    img = cv2.resize(img_bgr, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)
    ok, png = cv2.imencode(".png", img)
    return tk.PhotoImage(data=base64.b64encode(png.tobytes()))


def superponer(img_bgr, fruto, defecto):
    capa = img_bgr.copy()
    capa[fruto] = (0.6 * capa[fruto] + 0.4 * np.array([67, 160, 46])).astype(np.uint8)
    capa[defecto] = (0.35 * capa[defecto] + 0.65 * np.array([38, 38, 220])).astype(np.uint8)
    for m, color in ((fruto, (67, 160, 46)), (defecto, (38, 38, 220))):
        contornos, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(capa, contornos, -1, color, 2)
    return capa


# ---------------------------------------------------------------- ventana

class App:
    def __init__(self, raiz):
        self.raiz = raiz
        self.fotos, self.i = [], 0
        self.cache = {}  # (ruta_modelo, umbral, foto) -> resultado sin mascaras
        self.modelo, self.ruta_modelo = None, None
        self.s1_train = set()
        if os.path.exists(S1_TRAIN):
            self.s1_train = {l.strip() for l in open(S1_TRAIN, encoding="utf-8") if l.strip()}
        raiz.title("PaltaScan - prueba de modelos")

        barra = tk.Frame(raiz)
        barra.pack(fill="x", padx=8, pady=(6, 2))
        tk.Label(barra, text="Modelo:").pack(side="left")
        self.modelos = buscar_modelos()
        self.combo = ttk.Combobox(barra, state="readonly", width=62,
                                  values=[n for n, _, _ in self.modelos] + ["Otro archivo..."])
        self.combo.pack(side="left", padx=4)
        self.combo.bind("<<ComboboxSelected>>", lambda e: self.elegir_modelo())
        tk.Label(barra, text="Umbral defecto:").pack(side="left", padx=(8, 0))
        self.umbral = tk.DoubleVar(value=0.05)
        sp = tk.Spinbox(barra, from_=0.01, to=0.9, increment=0.05, width=5, textvariable=self.umbral,
                        command=self.mostrar)
        sp.pack(side="left", padx=4)
        sp.bind("<Return>", lambda e: self.mostrar())

        barra2 = tk.Frame(raiz)
        barra2.pack(fill="x", padx=8, pady=2)
        tk.Button(barra2, text="Abrir foto(s)", command=self.abrir_fotos).pack(side="left")
        tk.Button(barra2, text="Abrir carpeta", command=self.abrir_carpeta).pack(side="left", padx=4)
        self.btn_eval = tk.Button(barra2, text="Evaluar carpeta completa", command=self.evaluar_todo)
        self.btn_eval.pack(side="left", padx=4)
        tk.Button(barra2, text="Siguiente >", command=lambda: self.mover(1)).pack(side="right")
        tk.Button(barra2, text="< Anterior", command=lambda: self.mover(-1)).pack(side="right", padx=4)
        self.lbl_modelo = tk.Label(raiz, anchor="w", fg="#555", justify="left")
        self.lbl_modelo.pack(fill="x", padx=8)

        imgs = tk.Frame(raiz)
        imgs.pack(padx=8, pady=4)
        self.paneles = []
        for titulo in ("Original", "Anotación real", "Modelo (verde: fruto, rojo: defecto)"):
            lbl = tk.Label(imgs, text=titulo, compound="top")
            lbl.pack(side="left", padx=3)
            self.paneles.append(lbl)

        abajo = tk.Frame(raiz)
        abajo.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        foto = tk.LabelFrame(abajo, text="Esta foto", padx=8, pady=4)
        foto.pack(side="left", fill="both", expand=True)
        self.lbl_cat = tk.Label(foto, font=("Segoe UI", 16, "bold"), fg="white", bg="#777", padx=10)
        self.lbl_cat.pack(anchor="w", pady=(2, 4))
        self.lbl_foto = tk.Label(foto, font=("Consolas", 10), justify="left", anchor="nw")
        self.lbl_foto.pack(anchor="w")
        self.canvas = tk.Canvas(foto, width=330, height=120, highlightthickness=0)
        self.canvas.pack(anchor="w", pady=4)
        acum = tk.LabelFrame(abajo, text="Métricas acumuladas (fotos ya analizadas con este modelo y umbral)",
                             padx=8, pady=4)
        acum.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self.lbl_acum = tk.Label(acum, font=("Consolas", 10), justify="left", anchor="nw")
        self.lbl_acum.pack(anchor="w")

        raiz.bind("<Right>", lambda e: self.mover(1))
        raiz.bind("<Left>", lambda e: self.mover(-1))
        if self.modelos:
            self.combo.current(0)
            self.elegir_modelo()

    # --- modelo
    def elegir_modelo(self):
        idx = self.combo.current()
        if idx == len(self.modelos):
            ruta = filedialog.askopenfilename(filetypes=[("Modelo", "*.pt *.tflite *.onnx")])
            if not ruta:
                return
            nombre, umbral = os.path.basename(ruta), 0.15
            self.modelos.append((nombre, ruta, umbral))
            self.combo.config(values=[n for n, _, _ in self.modelos] + ["Otro archivo..."])
            self.combo.current(len(self.modelos) - 1)
        else:
            nombre, ruta, umbral = self.modelos[idx]
        self.lbl_modelo.config(text=f"Cargando {nombre}...")
        self.raiz.update_idletasks()
        try:
            self.modelo, self.ruta_modelo = cargar_modelo(ruta), ruta
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Modelo", f"No se pudo cargar {ruta}:\n{e}")
            return
        self.umbral.set(umbral)
        self.info_modelo()
        self.mostrar()

    def info_modelo(self):
        mb = os.path.getsize(self.ruta_modelo) / 1e6
        aviso = ""
        if "s1_defectos" in self.ruta_modelo:
            aviso = "\n⚠ Este modelo se entrenó con el split viejo: sus métricas en test están infladas (vio la mayoría de estas fotos)."
        self.lbl_modelo.config(text=f"{os.path.relpath(self.ruta_modelo, RAIZ)} · {mb:.1f} MB · {self.modelo.tipo}{aviso}")

    # --- fotos
    def abrir_fotos(self):
        rutas = filedialog.askopenfilenames(filetypes=[("Imágenes", "*.jpg *.jpeg *.png")])
        if rutas:
            self.fotos, self.i = list(rutas), 0
            self.mostrar()

    def abrir_carpeta(self, carpeta=None):
        carpeta = carpeta or filedialog.askdirectory()
        if carpeta:
            self.fotos = sorted(sum((glob.glob(os.path.join(carpeta, p)) for p in ("*.jpg", "*.jpeg", "*.png")), []))
            self.i = 0
            self.mostrar()

    def mover(self, paso):
        if self.fotos:
            self.i = (self.i + paso) % len(self.fotos)
            self.mostrar()

    def clave(self, ruta):
        return (self.ruta_modelo, round(float(self.umbral.get()), 3), ruta)

    def guardar_cache(self, ruta, r):
        self.cache[self.clave(ruta)] = {k: v for k, v in r.items()
                                        if k not in ("img", "fruto", "defecto", "gt_fruto", "gt_defecto")}

    def mostrar(self):
        if not self.fotos or self.modelo is None or self.btn_eval["state"] == "disabled":
            return
        ruta = self.fotos[self.i]
        try:
            r = analizar(self.modelo, ruta, float(self.umbral.get()))
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Análisis", str(e))
            return
        self.guardar_cache(ruta, r)
        self.info_modelo()

        img = r["img"]
        self._fotos = [a_photoimage(img)]
        if r["gt_fruto"] is not None:
            self._fotos.append(a_photoimage(superponer(img, r["gt_fruto"], r["gt_defecto"])))
        else:
            self._fotos.append(a_photoimage(np.full_like(img, 235)))
        self._fotos.append(a_photoimage(superponer(img, r["fruto"], r["defecto"])))
        for lbl, ph in zip(self.paneles, self._fotos):
            lbl.config(image=ph)
        self.paneles[1].config(text="Anotación real" if r["gt_fruto"] is not None else "Anotación real (no hay)")

        cat = r["ocde"]
        self.lbl_cat.config(text=cat.replace("Categoria", "Categoría"), bg=COLORES_OCDE.get(cat, "#777"))
        stem = os.path.splitext(os.path.basename(ruta))[0]
        lineas = [f"Foto {self.i + 1}/{len(self.fotos)}: {os.path.basename(ruta)[:48]}"]
        if "s1_defectos" in self.ruta_modelo and stem in self.s1_train:
            lineas.append("⚠ el modelo de Seminario I se entrenó con esta foto")
        lineas.append("")
        lineas.append("DEFECTOS")
        if "ocde_real" in r:
            ok = "OK" if r["ocde"] == r["ocde_real"] else "X"
            lineas.append(f"  OCDE: {r['ocde']}  | real: {r['ocde_real']}  [{ok}]")
            lineas.append(f"  % defecto: {r['ratio'] * 100:.1f}%  | real: {r['ratio_real'] * 100:.1f}%")
            tp, fp, fn = r["tp"], r["fp"], r["fn"]
            if tp + fp + fn:
                lineas.append(f"  IoU {tp / (tp + fp + fn):.2f} · precisión {tp / max(1, tp + fp):.2f} · "
                              f"recall {tp / max(1, tp + fn):.2f}")
            else:
                lineas.append("  sin defecto real ni predicho (correcto)")
        else:
            lineas.append(f"  OCDE: {r['ocde']}  | % defecto: {r['ratio'] * 100:.1f}%  (sin anotación para comparar)")
        lineas.append("")
        lineas.append("MADUREZ")
        if r["madurez"]:
            real = r["madurez_real"]
            extra = f"  | real: {real}  [{'OK' if real == r['madurez'] else 'X'}]" if real else ""
            lineas.append(f"  Nivel {r['madurez']} ({r['probs'].max() * 100:.0f}% de confianza){extra}")
        else:
            lineas.append("  este modelo no predice madurez")
        lineas.append("")
        lineas.append(f"Tiempo de inferencia: {r['ms']:.0f} ms")
        self.lbl_foto.config(text="\n".join(lineas))
        self.dibujar_probs(r)
        self.actualizar_acumulado()

    def dibujar_probs(self, r):
        c = self.canvas
        c.delete("all")
        if r["probs"] is None:
            return
        c.create_text(0, 8, anchor="w", text="Probabilidad por nivel de madurez", font=("Segoe UI", 9))
        for n, p in enumerate(r["probs"]):
            y = 22 + n * 19
            color = "#2E7D32" if n + 1 == r["madurez_real"] else ("#1565C0" if n + 1 == r["madurez"] else "#90A4AE")
            c.create_text(10, y + 7, text=str(n + 1), font=("Segoe UI", 9))
            c.create_rectangle(22, y, 22 + 250 * p, y + 14, fill=color, outline="")
            c.create_text(280, y + 7, anchor="w", text=f"{p * 100:.0f}%", font=("Segoe UI", 9))
        c.create_text(0, 118, anchor="sw", text="verde = nivel real · azul = predicho", font=("Segoe UI", 8), fill="#666")

    def actualizar_acumulado(self):
        clave = (self.ruta_modelo, round(float(self.umbral.get()), 3))
        res = [v for k, v in self.cache.items() if k[:2] == clave]
        self.lbl_acum.config(text=resumen_metricas(res))

    def evaluar_todo(self):
        if not self.fotos or self.modelo is None:
            messagebox.showinfo("Evaluar", "Primero abra una carpeta de fotos.")
            return
        pendientes = [f for f in self.fotos if self.clave(f) not in self.cache]
        if not pendientes:
            self.actualizar_acumulado()
            return
        self.btn_eval.config(state="disabled")
        umbral = float(self.umbral.get())

        def trabajo():
            for n, ruta in enumerate(pendientes, 1):
                try:
                    r = analizar(self.modelo, ruta, umbral)
                except Exception:  # noqa: BLE001
                    continue
                self.raiz.after(0, self.guardar_cache, ruta, r)
                if n % 5 == 0 or n == len(pendientes):
                    self.raiz.after(0, self.progreso, n, len(pendientes))
            self.raiz.after(0, self.fin_evaluacion)

        threading.Thread(target=trabajo, daemon=True).start()

    def fin_evaluacion(self):
        self.btn_eval.config(state="normal", text="Evaluar carpeta completa")
        self.actualizar_acumulado()

    def progreso(self, n, total):
        self.btn_eval.config(text=f"Evaluando {n}/{total}...")
        self.actualizar_acumulado()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fotos", default="")
    args = ap.parse_args()
    raiz = tk.Tk()
    app = App(raiz)
    if args.fotos:
        if os.path.isdir(args.fotos):
            app.abrir_carpeta(args.fotos)
        else:
            app.fotos = [args.fotos]
            app.mostrar()
    raiz.mainloop()


if __name__ == "__main__":
    main()
