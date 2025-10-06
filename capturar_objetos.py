import cv2
import os
import uuid
import json
from ultralytics import YOLO

class CapturadorObjetos:
    def __init__(self):
        # Usar modelo pre-entrenado para ayuda en captura
        self.model_detector = YOLO('yolov8n.pt')
        self.cap = cv2.VideoCapture(0)
        self.output_dir = "dataset_objetos"
        self.clase_actual = ""
        self.anotaciones = {}
        
    def crear_directorio_clase(self, nombre_clase):
        path_imagenes = os.path.join(self.output_dir, nombre_clase, "images")
        path_anotaciones = os.path.join(self.output_dir, nombre_clase, "annotations")
        os.makedirs(path_imagenes, exist_ok=True)
        os.makedirs(path_anotaciones, exist_ok=True)
        return path_imagenes, path_anotaciones
    
    def dibujar_cuadro_seleccion(self, frame, x1, y1, x2, y2):
        """Dibuja el cuadro de selección en el frame"""
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, self.clase_actual, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    def capturar_manual(self, nombre_clase, num_fotos=100):
        """Captura manual seleccionando el área del objeto"""
        self.clase_actual = nombre_clase
        path_imagenes, path_anotaciones = self.crear_directorio_clase(nombre_clase)
        contador = 0
        
        print(f"Captura MANUAL - {num_fotos} fotos para: {nombre_clase}")
        print("Instrucciones:")
        print("1. Presiona 'c' para capturar la imagen")
        print("2. Usa el mouse para seleccionar el área del objeto")
        print("3. Presiona 'q' para terminar")
        
        # Variables para selección con mouse
        drawing = False
        ix, iy = -1, -1
        x1, y1, x2, y2 = -1, -1, -1, -1
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing, ix, iy, x1, y1, x2, y2
            
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                ix, iy = x, y
                
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    x1, y1, x2, y2 = min(ix, x), min(iy, y), max(ix, x), max(iy, y)
                    
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                x1, y1, x2, y2 = min(ix, x), min(iy, y), max(ix, x), max(iy, y)
        
        cv2.namedWindow("Capturar Objetos")
        cv2.setMouseCallback("Capturar Objetos", mouse_callback)
        
        while contador < num_fotos:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame_copy = frame.copy()
            
            # Dibujar cuadro de selección si existe
            if x1 != -1 and y1 != -1 and x2 != -1 and y2 != -1:
                self.dibujar_cuadro_seleccion(frame_copy, x1, y1, x2, y2)
            
            # Mostrar instrucciones
            cv2.putText(frame_copy, f"Capturas: {contador}/{num_fotos}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_copy, "Presiona 'c' para capturar, 'q' para salir", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Capturar Objetos", frame_copy)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') and x1 != -1:
                # Guardar imagen
                filename = f"{uuid.uuid4()}.jpg"
                img_path = os.path.join(path_imagenes, filename)
                cv2.imwrite(img_path, frame)
                
                # Guardar anotación (formato YOLO)
                h, w = frame.shape[:2]
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                # Guardar anotación en archivo .txt
                anno_path = os.path.join(path_anotaciones, filename.replace('.jpg', '.txt'))
                with open(anno_path, 'w') as f:
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                
                contador += 1
                print(f"Captura {contador}/{num_fotos} guardada")
                
            elif key == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"Captura manual completada para {nombre_clase}")

# Uso
if __name__ == "__main__":
    capturador = CapturadorObjetos()
    
    print("¿Qué tipo de captura prefieres?")
    print("1. Captura manual (seleccionas el área del objeto)")
    opcion = input("Selecciona 1: ")
    
    nombre_clase = input("Ingresa el nombre del objeto (ej: 'telefono', 'vaso', 'libro'): ")
    num_fotos = int(input("Número de fotos a capturar (recomendado: 100-200): "))
    
    if opcion == "1":
        capturador.capturar_manual(nombre_clase, num_fotos)