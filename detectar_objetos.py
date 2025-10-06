import cv2
import os
from ultralytics import YOLO
import time

class DetectorObjetos:
    def __init__(self, model_path='runs/detect/train/weights/best.pt'):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(0)
        self.conf_threshold = 0.5
        self.colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), 
                      (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    def dibujar_detecciones(self, frame, results):
        """Dibuja las detecciones en el frame"""
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.conf[0] > self.conf_threshold:
                    # Coordenadas
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Clase y confianza
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = self.model.names[cls]
                    
                    # Color basado en la clase
                    color = self.colors[cls % len(self.colors)]
                    
                    # Dibujar bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Dibujar etiqueta
                    label = f"{class_name} {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def ejecutar_deteccion(self):
        """Ejecuta la detección en tiempo real"""
        print("Iniciando detección en tiempo real...")
        print("Presiona 'q' para salir")
        print("Presiona '+' para aumentar confianza")
        print("Presiona '-' para disminuir confianza")
        
        fps_time = time.time()
        frame_count = 0
        fps = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Realizar detección
            results = self.model(frame, verbose=False)
            
            # Dibujar detecciones
            frame_con_detecciones = self.dibujar_detecciones(frame.copy(), results)
            
            # Calcular y mostrar FPS
            frame_count += 1
            if time.time() - fps_time >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_time = time.time()
            
            cv2.putText(frame_con_detecciones, f"FPS: {fps}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_con_detecciones, f"Conf: {self.conf_threshold:.2f}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Detección de Objetos - YOLO", frame_con_detecciones)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+'):
                self.conf_threshold = min(0.9, self.conf_threshold + 0.05)
            elif key == ord('-'):
                self.conf_threshold = max(0.1, self.conf_threshold - 0.05)
        
        self.cap.release()
        cv2.destroyAllWindows()

# Uso
if __name__ == "__main__":
    # Buscar el modelo entrenado automáticamente
    model_path = 'runs/detect/train/weights/best.pt'
    
    if not os.path.exists(model_path):
        print("Modelo no encontrado. Asegúrate de haber entrenado primero.")
        exit()
    
    detector = DetectorObjetos(model_path)
    detector.ejecutar_deteccion()