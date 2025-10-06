import os
import yaml
import shutil
from ultralytics import YOLO
from sklearn.model_selection import train_test_split

class EntrenadorObjetos:
    def __init__(self, dataset_path="dataset_objetos"):
        self.dataset_path = dataset_path
        self.yolo_dataset_path = "yolo_dataset_objetos"
        
    def preparar_dataset_yolo(self):
        """Prepara el dataset en formato YOLO"""
        print("Preparando dataset en formato YOLO...")
        
        # Crear estructura de directorios
        os.makedirs(f"{self.yolo_dataset_path}/images/train", exist_ok=True)
        os.makedirs(f"{self.yolo_dataset_path}/images/val", exist_ok=True)
        os.makedirs(f"{self.yolo_dataset_path}/labels/train", exist_ok=True)
        os.makedirs(f"{self.yolo_dataset_path}/labels/val", exist_ok=True)
        
        # Obtener todas las clases
        clases = [d for d in os.listdir(self.dataset_path) 
                 if os.path.isdir(os.path.join(self.dataset_path, d))]
        
        # Mapeo de clases a IDs
        class_mapping = {clase: idx for idx, clase in enumerate(clases)}
        
        # Recolectar todas las imágenes y anotaciones
        all_images = []
        all_annotations = []
        
        for clase in clases:
            images_dir = os.path.join(self.dataset_path, clase, "images")
            annotations_dir = os.path.join(self.dataset_path, clase, "annotations")
            
            for img_file in os.listdir(images_dir):
                if img_file.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(images_dir, img_file)
                    anno_path = os.path.join(annotations_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
                    
                    if os.path.exists(anno_path):
                        all_images.append(img_path)
                        all_annotations.append(anno_path)
        
        # Dividir en train/val
        train_imgs, val_imgs, train_annos, val_annos = train_test_split(
            all_images, all_annotations, test_size=0.2, random_state=42
        )
        
        # Copiar archivos a estructura YOLO
        for img_path, anno_path in zip(train_imgs, train_annos):
            shutil.copy(img_path, f"{self.yolo_dataset_path}/images/train/")
            shutil.copy(anno_path, f"{self.yolo_dataset_path}/labels/train/")
        
        for img_path, anno_path in zip(val_imgs, val_annos):
            shutil.copy(img_path, f"{self.yolo_dataset_path}/images/val/")
            shutil.copy(anno_path, f"{self.yolo_dataset_path}/labels/val/")
        
        # Crear archivo dataset.yaml
        dataset_config = {
            'path': os.path.abspath(self.yolo_dataset_path),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(clases),
            'names': clases
        }
        
        with open('dataset_config.yaml', 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"Dataset preparado con {len(clases)} clases: {clases}")
        return 'dataset_config.yaml'
    
    def entrenar_modelo(self, epochs=100, imgsz=640):
        """Entrena el modelo YOLO"""
        print("Iniciando entrenamiento...")
        
        # Preparar dataset
        config_path = self.preparar_dataset_yolo()
        
        # Cargar modelo base
        model = YOLO('yolov8n.pt')  # Puedes usar 'yolov8s.pt' para mejor precisión
        
        # Entrenar
        results = model.train(
            data=config_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=16,
            patience=20,
            device='cuda' if os.getenv('CUDA_VISIBLE_DEVICES') else 'cpu',
            lr0=0.01,
            augment=True,
            degrees=10,  # Rotación
            translate=0.1,  # Traslación
            scale=0.5,  # Escala
            shear=2.0,  # Inclinación
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0
        )
        
        print("¡Entrenamiento completado!")
        return results

# Uso
if __name__ == "__main__":
    entrenador = EntrenadorObjetos()
    
    epochs = int(input("Número de épocas (recomendado: 100): ") or "100")
    imgsz = int(input("Tamaño de imagen (recomendado: 640): ") or "640")
    
    resultados = entrenador.entrenar_modelo(epochs=epochs, imgsz=imgsz)