import cv2 
from ultralytics import YOLO

if __name__ == "__main__":
    # Cargar el modelo
    model = YOLO("C:/Users/carlo/OneDrive/Documentos/Red Neural Saves/yolov8_checkpoint2/weights/best.pt")

    # Cargar la imagen
    img_name = "1"
    image = cv2.imread(f"C:/Users/carlo/OneDrive/Documentos/Red Neural/{img_name}.jpg")

    # Realizar la predicción
    pred = model.predict(image)[0]
    #Dibujar la predicción
    pred_image = pred.plot()

    # Guardar la imagen con la predicción
    cv2.imwrite(f"C:/Users/carlo/OneDrive/Documentos/Red Neural Results/{img_name}.jpg", pred_image)
    
    # Mostrar resultados
    cv2.imshow("Predicción", pred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()