import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    """Тест главной страницы"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_check():
    """Тест проверки здоровья"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "api_status" in data
    assert "ollama_status" in data

def test_models_endpoint():
    """Тест получения списка моделей"""
    response = client.get("/models")
    assert response.status_code == 200

def test_generate_endpoint():
    """Тест генерации текста"""
    test_data = {
        "prompt": "Ответь кратко: привет!",
        "max_tokens": 10,
        "temperature": 0.1
    }
    
    response = client.post("/generate", json=test_data)
    
    # API должен вернуть ответ, даже если Ollama не запущен
    assert response.status_code in [200, 503, 500]
    
    if response.status_code == 200:
        data = response.json()
        assert "response" in data
        assert "success" in data

def test_generate_invalid_data():
    """Тест с некорректными данными"""
    invalid_data = {"wrong_field": "test"}
    
    response = client.post("/generate", json=invalid_data)
    assert response.status_code == 422  # Validation error
