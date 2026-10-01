# PaltaScan: funcionalidades (v0.3.0)

App Android que analiza fotos de palta Hass **en el celular, sin conexión**. Usa
el modelo multitarea YOLOv8s-seg exportado a TFLite (rango dinámico, 12.3 MB)
y corre con LiteRT 1.4.2.

## Estilo

Barra superior verde con el título en blanco, fondo blanco, texto negro y botones planos con esquinas rectas (verde para la acción principal y negro para las demás).

## Pantallas

### 1. Pantalla principal

**Cabecera:** el nombre de la app y un subtítulo. Mientras carga el modelo se
lee "Cargando modelo…". Si falta el modelo, aparece un aviso y los botones
quedan deshabilitados.

**Lote actual:** se ve el lote donde se guardarán los análisis. El botón
**Cambiar** abre una lista con:
- "Sin lote";
- los lotes existentes;
- "+ Nuevo lote", que pide nombre y productor.

La app recuerda el lote elegido aunque se cierre.

**Botones:**
- **Tomar foto:** abre la cámara del sistema.
- **Galería:** abre el selector de fotos de Android.
- **Historial:** abre la segunda pantalla.

**Resultado:** después de analizar una foto, en la misma pantalla se ve:
- la foto, con un interruptor para alternar entre la original y la versión con
  las **marcas de defectos** (máscaras superpuestas);
- la **categoría OCDE** (Cat. I, Cat. II o Rechazado), con su color;
- el **% del área con defecto**;
- la **madurez**: nivel del 1 al 5, su nombre y la confianza;
- el **tiempo de análisis**;
- los **avisos**, si los hay;
- el mensaje "Guardado en <lote>".

### 2. Historial

- **Selector:** "Todos los análisis" o un lote concreto.
- **Resumen:** total, cantidad por categoría y cantidad por nivel de madurez.
- **Exportar CSV:** funciona con todos los análisis o con un lote.
- **Exportar PDF:** solo con un lote. Genera un informe con el resumen y una
  tabla con miniaturas.
- Ambos archivos se comparten con el menú de Android (WhatsApp, Drive, correo,
  etc.).
- **Lista de análisis:** cada fila muestra la miniatura, la categoría, la
  fecha, la madurez y el % de defecto.
- **Detalle:** al tocar una fila se abre el mismo resultado de la pantalla
  principal, con los botones:
  - **Mover a lote**;
  - **Eliminar**, que pide confirmación y borra el registro y las fotos;
  - **Cerrar**.

## Validación de la foto

Se calcula después de la inferencia.

| Caso | Tipo | Efecto |
|---|---|---|
| No hay palta | Bloqueante | No se guarda y se muestra el motivo |
| Más de una palta | Bloqueante | No se guarda y se muestra el motivo |
| Palta cortada en el borde | Bloqueante | No se guarda y se muestra el motivo |
| Imagen oscura (luminancia media < 70) | Aviso | Se guarda con el aviso |
| Imagen borrosa (varianza del Laplaciano < 40) | Aviso | Se guarda con el aviso |
| Confianza de madurez < 0.6 | Aviso | Se guarda con el aviso |

## Análisis

1. **Preproceso:** corrige la orientación EXIF, reduce la imagen y aplica
   letterbox a 640×640.
2. **Inferencia** en un único modelo con dos tareas:
   - segmentación de la palta y de sus defectos;
   - clasificación de la madurez en 5 clases.
3. **Postproceso:** NMS, máscaras y cálculo de la razón entre píxeles con
   defecto y píxeles de la palta.
4. **Clasificación OCDE** según esa razón, con los umbrales de Cat. I, Cat. II
   y Rechazado.

Los tiempos de preproceso, inferencia y postproceso se registran por separado.
En el Xiaomi 11 Lite 5G NE, cada foto tarda ~0.9 s en total.

## Datos

- **Base SQLite local (v2):**
  - tabla `lotes`: nombre, productor y fecha;
  - tabla `inspecciones`: fotos, madurez, probabilidad, razón, categoría,
    tiempos, avisos y `lote_id` (anulable).
- La migración desde la v1 conserva el historial.
- Las fotos (original y con marcas) se guardan en el almacenamiento interno de
  la app.
- No se envía nada a internet.

## Pruebas

- **20 pruebas unitarias:** validación de la captura, postproceso, OCDE, CSV,
  etc.
- **Prueba instrumentada `ConcordanciaTest`:** compara la app con Python en las
  108 imágenes de prueba.
  - Madurez: 108/108.
  - Categoría OCDE: 107/108.

## Pendiente (Fase 2)

- Cámara en vivo con CameraX:
  - chequeo de luz y enfoque en tiempo real;
  - modo continuo con conteo por categoría.
- Medir la latencia con el delegado GPU.
