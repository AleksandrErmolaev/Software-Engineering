import requests
import json
import os

BASE_URL = "http://localhost:8000"


def test_health():
    """Тест здоровья API"""
    response = requests.get(f"{BASE_URL}/health")
    print("✅ Health Check:", response.json())
    assert response.status_code == 200


def test_predict_file():
    """Тест предсказания из файла"""
    test_image_path = "static/images/test_image.jpg"

    if not os.path.exists(test_image_path):
        print("⚠️  Тестовое изображение не найдено, создаем простое...")
        # Создаем простое тестовое изображение
        from PIL import Image
        img = Image.new('RGB', (224, 224), color='red')
        os.makedirs("static/images", exist_ok=True)
        img.save(test_image_path)

    with open(test_image_path, "rb") as f:
        files = {"file": ("test_image.jpg", f, "image/jpeg")}
        response = requests.post(f"{BASE_URL}/predict?top_k=3", files=files)

    print("✅ File Prediction Test:")
    print(json.dumps(response.json(), indent=2))
    assert response.status_code == 200


def test_predict_url():
    """Тест предсказания из URL"""
    test_url = "https://github.com/opencv/opencv/raw/master/samples/data/basketball1.png"

    response = requests.post(
        f"{BASE_URL}/predict/url",
        params={"url": test_url, "top_k": 3}
    )

    print("✅ URL Prediction Test:")
    print(json.dumps(response.json(), indent=2))
    assert response.status_code == 200


if __name__ == "__main__":
    print("🚀 Запуск тестов API...")
    test_health()
    test_predict_file()
    test_predict_url()
    print("🎉 Все тесты пройдены!")