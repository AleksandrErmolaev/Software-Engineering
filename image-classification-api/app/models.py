import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from PIL import Image
import io
import requests


class ImageClassifier:
    def __init__(self):
        self.model = ResNet50(weights='imagenet')
        self.input_size = (224, 224)
        print("✅ Модель ResNet50 загружена успешно!")

    def load_image_from_url(self, url: str) -> Image.Image:
        """Загрузка изображения из URL"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
            return img.convert('RGB')
        except Exception as e:
            raise Exception(f"Ошибка загрузки изображения из URL: {e}")

    def load_image_from_file(self, file_content: bytes) -> Image.Image:
        """Загрузка изображения из файла в памяти"""
        try:
            img = Image.open(io.BytesIO(file_content))
            return img.convert('RGB')
        except Exception as e:
            raise Exception(f"Ошибка загрузки изображения: {e}")

    def preprocess_image(self, img: Image.Image) -> np.ndarray:
        """Предобработка изображения для модели"""
        try:
            # Изменение размера
            img = img.resize(self.input_size)
            # Конвертация в массив
            img_array = image.img_to_array(img)
            # Добавление батч-измерения
            img_array = np.expand_dims(img_array, axis=0)
            # Предобработка для ResNet50
            img_array = preprocess_input(img_array)
            return img_array
        except Exception as e:
            raise Exception(f"Ошибка предобработки изображения: {e}")

    def predict(self, img_array: np.ndarray, top_k: int = 5) -> list:
        """Предсказание классов изображения"""
        try:
            predictions = self.model.predict(img_array, verbose=0)
            decoded_predictions = decode_predictions(predictions, top=top_k)[0]

            # Форматируем результат
            result = []
            for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
                result.append({
                    "rank": i + 1,
                    "class_id": imagenet_id,
                    "class_name": label,
                    "confidence": float(score),
                    "confidence_percentage": f"{score:.2%}"
                })

            return result
        except Exception as e:
            raise Exception(f"Ошибка предсказания: {e}")

    def predict_from_file(self, file_content: bytes, top_k: int = 5) -> dict:
        """Полный пайплайн предсказания из файла"""
        try:
            # Загрузка изображения
            img = self.load_image_from_file(file_content)
            # Предобработка
            img_array = self.preprocess_image(img)
            # Предсказание
            predictions = self.predict(img_array, top_k)

            return {
                "status": "success",
                "predictions": predictions,
                "top_prediction": predictions[0] if predictions else None
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def predict_from_url(self, url: str, top_k: int = 5) -> dict:
        """Полный пайплайн предсказания из URL"""
        try:
            # Загрузка изображения
            img = self.load_image_from_url(url)
            # Предобработка
            img_array = self.preprocess_image(img)
            # Предсказание
            predictions = self.predict(img_array, top_k)

            return {
                "status": "success",
                "predictions": predictions,
                "top_prediction": predictions[0] if predictions else None
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }


classifier = ImageClassifier()