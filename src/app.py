# from ultralytics import YOLO
# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = YOLO("yolov8n.pt")

# checkpoint_dir = "C:/Users/carlo/OneDrive/Documentos/Red Neural Saves/"

# model.train(
#     data="C:/Users/carlo/Downloads/Packages-V4.v7-final-no-augmentation.yolov8/data.yaml", 
#     epochs=300, 
#     device=device,
#     save=True,  # Guarda los checkpoints
#     save_period=5,  # Guardar checkpoint cada 5 epochs (ajusta según prefieras)
#     project=checkpoint_dir,  # Directorio donde guardar los checkpoints
#     name="yolov8_checkpoint",  # Nombre del proyecto para identificar checkpoints
# )

from ultralytics import YOLO
import torch

# Definir el dispositivo de entrenamiento
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ruta del checkpoint más reciente 
checkpoint_path = "C:/Users/carlo/OneDrive/Documentos/Red Neural Saves/yolov8_checkpoint/weights/last.pt"

# Cargar el modelo desde el checkpoint
model = YOLO(checkpoint_path)

# Continuar el entrenamiento desde el checkpoint cargado
model.train(
    data="C:/Users/carlo/Downloads/Packages-V4.v7-final-no-augmentation.yolov8/data.yaml", 
    epochs=23,  # Número total de epochs que faltan para completar 300
    device=device,
    save=True,  # Continuar guardando checkpoints
    save_period=5,  # Guardar checkpoint cada 5 epochs
    project="C:/Users/carlo/OneDrive/Documentos/Red Neural Saves/",  # Directorio donde guardar los checkpoints
    name="yolov8_checkpoint"  # Nombre del proyecto para identificar checkpoints
)

