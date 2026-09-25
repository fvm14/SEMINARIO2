Modelo alternativo (archivado): ResNet34 + decoder FPN + cabezal de madurez,
segmentacion semantica por pixel. Se reemplazo por YOLOv8s-seg multitarea
(scripts/yolo_multitarea.py) para mantener continuidad con Seminario I y con
la Etapa 2 del roadmap de Seminario II. Sigue siendo ejecutable:
    python scripts/alternativa_resnet34/train.py --epochs 2 --max-batches 5
