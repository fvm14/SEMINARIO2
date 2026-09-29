# PaltaScan — app Android (Etapas 3 y 4)

App Android nativa (Kotlin) que evalua una palta desde una foto **en el propio
celular, sin internet**: segmenta la palta y sus defectos, clasifica la
madurez (1-5) y asigna la categoria OCDE. Guarda un historial por lote que se
puede exportar a CSV.


## Estructura

```
PaltaScan/app/src/main/java/pe/edu/ulima/paltascan/
  ml/            Nucleo del modelo, port de scripts/inferencia_movil.py y scripts/ocde.py
    Configuracion.kt  umbrales (palta 0.35, defecto 0.05 para ronda1, IoU 0.7, ROI 20 px)
    Letterbox.kt      geometria del letterbox (igual que en Python)
    LectorSalidas.kt  identifica las 5 salidas del .tflite por su forma
    Postproceso.kt    confianza, NMS por clase, mascara = sigmoide(coef . prototipos)
    Ocde.kt           fruto completo, filtro ROI, ratio defecto/fruto, categoria
    Analizador.kt     pipeline completo con TFLite y tiempos por etapa
  datos/         SQLite (historial), CSV y utilidades de imagen
  ui/            Inicio, Resultado, Historial
PaltaScan/app/src/test/   tests de la logica en la PC (sin celular ni modelo)
```

Todo lo de `ml/` excepto `Analizador.kt` es Kotlin puro, asi que se prueba en la
PC con `gradlew test`.

## Pantallas

1. **Inicio**: lote actual (y boton "Nuevo lote"), "Tomar foto", "Elegir de
   galeria" e "Historial". Si falta el modelo, lo avisa y desactiva el
   analisis.
2. **Resultado**: tarjeta con la categoria OCDE (color segun la categoria),
   foto original junto a la foto con el fruto en verde y los defectos en rojo,
   madurez con su confianza, % de area con defecto y tiempo por etapa
   (preproceso, inferencia y postproceso).
3. **Historial del lote**: conteo por categoria y por madurez, lista de paltas
   (tocar una abre su resultado) y "Exportar CSV".

## Poner el modelo (pendiente: lo tiene el companero)

1. Generar el TFLite desde `best.pt` de `yolov8s_mt_ronda1`
   (`scripts/exportar_movil.py` → ONNX → TFLite dynamic-range, ver Etapa 2).
2. Copiarlo como `PaltaScan/app/src/main/assets/palta_multitarea.tflite`.
   Esta en `.gitignore`: no se sube al repo.
3. Si se usa otro modelo, ajustar `CONF_DEFECTO` en `Configuracion.kt` segun
   su calibracion.

## Compilar y probar

Requisitos ya presentes en la PC: Android Studio (con su JDK), Android SDK 36
y un emulador (`Medium_Phone`).

```bash
cd app_movil/PaltaScan
# (Git Bash) usar el JDK de Android Studio
export JAVA_HOME="/c/Program Files/Android/Android Studio/jbr"
./gradlew testDebugUnitTest   # 13 tests de la logica
./gradlew assembleDebug       # APK en app/build/outputs/apk/debug/
```

O abrir la carpeta `app_movil/PaltaScan` en Android Studio y darle a Run.

## Verificar contra Python (cuando llegue el modelo)

```bash
python scripts/referencia_app.py --modelo <ruta>.tflite --fotos <carpeta> --conf-defecto 0.05
```

Analizar las mismas fotos en la app: la categoria OCDE y la madurez deben
coincidir, y el ratio puede diferir en milesimas, porque la app trabaja a la
resolucion del letterbox.

## Diferencias con Python (conscientes)

- La foto se reduce a un lado maximo de 1600 px antes de analizarla, y las
  mascaras y el ratio se calculan en el espacio del letterbox (maximo
  800x800) en vez de a la resolucion original. El kernel ROI se escala en
  proporcion.
- Modo foto por foto (no video en vivo): con segmentacion a 800 px, el video
  en tiempo real no es realista en un celular.

## Pendiente

- [ ] Poner el modelo y medir la latencia real en un celular (gama media y alta).
- [ ] Comparar app vs. `referencia_app.py` sobre las fotos de test.
- [ ] Revisar las pantallas contra `frontend/PaltaScan — Pantallas.pdf`.
- [ ] Umbrales OCDE con fuente citable (hoy 0.094 / 0.141, igual que `scripts/ocde.py`).
