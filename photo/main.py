import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import cv2


class ImageClassifier:
    def __init__(self):
        self.model = ResNet50(weights='imagenet')
        print("Модель ResNet50 загружена успешно!")

    def load_image_from_url(self, url):
        """Загрузка изображения из URL"""
        try:
            response = requests.get(url)
            img = image.load_img(BytesIO(response.content), target_size=(224, 224))
            return img
        except Exception as e:
            print(f"Ошибка загрузки изображения: {e}")
            return None

    def load_image_from_file(self, file_path):
        """Загрузка изображения из файла"""
        try:
            img = image.load_img(file_path, target_size=(224, 224))
            return img
        except Exception as e:
            print(f"Ошибка загрузки изображения: {e}")
            return None

    def predict_image(self, img):
        """Классификация изображения"""
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        img_array = preprocess_input(img_array)

        predictions = self.model.predict(img_array)

        decoded_predictions = decode_predictions(predictions, top=5)[0]

        return decoded_predictions

    def display_results(self, img, predictions):
        """Отображение изображения и результатов классификации"""
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Входное изображение')

        plt.subplot(1, 2, 2)
        classes = [pred[1] for pred in predictions]
        probabilities = [pred[2] for pred in predictions]

        colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
        bars = plt.barh(classes, probabilities, color=colors)
        plt.xlabel('Вероятность')
        plt.title('Топ-5 предсказаний')
        plt.gca().invert_yaxis()

        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            plt.text(prob + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{prob:.2%}', va='center')

        plt.tight_layout()
        plt.show()


def main():
    classifier = ImageClassifier()



    print("\n=== Классификация изображения из файла ===")
    file_path = "./photo1.jpg" 
    img = classifier.load_image_from_file(file_path)

    if img is not None:
        predictions = classifier.predict_image(img)
        classifier.display_results(img, predictions)

        print("\nТоп-5 предсказаний:")
        for i, (imagenet_id, label, score) in enumerate(predictions):
            print(f"{i+1}. {label}: {score:.2%}")


if __name__ == "__main__":
    main()
