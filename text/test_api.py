import pytest
from fastapi.testclient import TestClient
from main import app
import json

# Инициализация тестового клиента
client = TestClient(app)


class TestSentimentAnalysisAPI:
    """Тесты для API анализа тональности"""

    def test_root_endpoint(self):
        """Тест корневого эндпоинта"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "Sentiment Analysis API"

    def test_health_check(self):
        """Тест проверки здоровья API"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] == True

    def test_analyze_sentiment_positive(self):
        """Тест анализа положительного текста"""
        test_data = {"text": "Я очень доволен этим продуктом, он прекрасен!"}
        response = client.post("/analyze", json=test_data)

        assert response.status_code == 200
        data = response.json()

        assert "text" in data
        assert "label" in data
        assert "score" in data
        assert "sentiment" in data
        assert data["text"] == test_data["text"]
        assert data["score"] > 0.5  # Высокая уверенность

    def test_analyze_sentiment_negative(self):
        """Тест анализа отрицательного текста"""
        test_data = {"text": "Это ужасный продукт, я разочарован"}
        response = client.post("/analyze", json=test_data)

        assert response.status_code == 200
        data = response.json()

        assert data["text"] == test_data["text"]
        assert data["score"] > 0.5

    def test_analyze_sentiment_empty_text(self):
        """Тест с пустым текстом"""
        test_data = {"text": ""}
        response = client.post("/analyze", json=test_data)

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    def test_analyze_batch(self):
        """Тест пакетного анализа"""
        test_data = {
            "texts": [
                "Это прекрасный день!",
                "Я ненавижу эту погоду",
                "Нормальный такой день"
            ]
        }
        response = client.post("/analyze-batch", json=test_data)

        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert len(data["results"]) == 3

        for result in data["results"]:
            assert "text" in result
            assert "label" in result
            assert "score" in result
            assert "sentiment" in result
            assert result["score"] > 0.5

    def test_analyze_batch_empty_list(self):
        """Тест пакетного анализа с пустым списком"""
        test_data = {"texts": []}
        response = client.post("/analyze-batch", json=test_data)

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    def test_analyze_batch_with_empty_texts(self):
        """Тест пакетного анализа с пустыми текстами"""
        test_data = {
            "texts": [
                "Полезный продукт",
                "",  # Пустой текст
                "   ",  # Текст только с пробелами
                "Не очень понравилось"
            ]
        }
        response = client.post("/analyze-batch", json=test_data)

        assert response.status_code == 200
        data = response.json()

        # Должны остаться только 2 валидных текста
        assert len(data["results"]) == 2
        assert data["results"][0]["text"] == "Полезный продукт"
        assert data["results"][1]["text"] == "Не очень понравилось"

    def test_response_structure(self):
        """Тест структуры ответа"""
        test_data = {"text": "Тестовый текст для анализа"}
        response = client.post("/analyze", json=test_data)

        assert response.status_code == 200
        data = response.json()

        # Проверяем наличие всех ожидаемых полей
        expected_fields = {"text", "label", "score", "sentiment"}
        assert set(data.keys()) == expected_fields

        # Проверяем типы данных
        assert isinstance(data["text"], str)
        assert isinstance(data["label"], str)
        assert isinstance(data["score"], float)
        assert isinstance(data["sentiment"], str)

        # Проверяем диапазон вероятности
        assert 0 <= data["score"] <= 1


class TestIntegration:
    """Интеграционные тесты"""

    def test_multiple_requests(self):
        """Тест множественных запросов"""
        test_texts = [
            {"text": "Отлично!"},
            {"text": "Плохо"},
            {"text": "Нормально"}
        ]

        for test_data in test_texts:
            response = client.post("/analyze", json=test_data)
            assert response.status_code == 200
            data = response.json()
            assert data["text"] == test_data["text"]


def test_performance():
    """Тест производительности с большим количеством текстов"""
    import time

    # Генерируем тестовые данные
    test_texts = ["Тестовый текст " + str(i) for i in range(10)]
    test_data = {"texts": test_texts}

    start_time = time.time()
    response = client.post("/analyze-batch", json=test_data)
    end_time = time.time()

    assert response.status_code == 200
    processing_time = end_time - start_time

    # Проверяем что обработка заняла разумное время
    assert processing_time < 30.0  # 30 секунд максимум для 10 текстов

    data = response.json()
    assert len(data["results"]) == 10


if __name__ == "__main__":
    # Запуск тестов
    pytest.main([__file__, "-v"])