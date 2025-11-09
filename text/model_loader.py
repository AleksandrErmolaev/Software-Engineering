import os
import requests
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "cointegrated/rubert-tiny-sentiment-balanced"
LOCAL_MODEL_PATH = "./models/rubert-tiny-sentiment-balanced"


def download_model_with_retry():
    """Скачивание модели с повторными попытками"""
    os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

    if os.path.exists(os.path.join(LOCAL_MODEL_PATH, "config.json")):
        logger.info("Используется локальная копия модели")
        return LOCAL_MODEL_PATH

    logger.info("Скачивание модели...")

    try:
        # Скачиваем модель через transformers
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            cache_dir=LOCAL_MODEL_PATH,
            force_download=True,
            resume_download=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=LOCAL_MODEL_PATH,
            force_download=True,
            resume_download=True
        )

        model.save_pretrained(LOCAL_MODEL_PATH)
        tokenizer.save_pretrained(LOCAL_MODEL_PATH)
        logger.info("Модель успешно скачана и сохранена локально")

    except Exception as e:
        logger.warning(f"Ошибка при скачивании: {e}")
        return None

    return LOCAL_MODEL_PATH


def create_classifier():
    """Создание классификатора с обработкой ошибок"""
    max_retries = 3

    for attempt in range(max_retries):
        try:
            # Пытаемся использовать локальную модель
            if os.path.exists(LOCAL_MODEL_PATH):
                logger.info(f"Попытка {attempt + 1}: Загрузка локальной модели")
                classifier = pipeline(
                    "sentiment-analysis",
                    model=LOCAL_MODEL_PATH,
                    tokenizer=LOCAL_MODEL_PATH
                )
                return classifier
            else:
                # Скачиваем модель
                model_path = download_model_with_retry()
                if model_path:
                    classifier = pipeline(
                        "sentiment-analysis",
                        model=model_path,
                        tokenizer=model_path
                    )
                    return classifier
                else:
                    # Если скачивание не удалось, пробуем напрямую
                    logger.info(f"Попытка {attempt + 1}: Прямая загрузка модели")
                    classifier = pipeline(
                        "sentiment-analysis",
                        model=MODEL_NAME,
                        max_retries=max_retries
                    )
                    return classifier

        except Exception as e:
            logger.warning(f"Попытка {attempt + 1} не удалась: {e}")
            if attempt == max_retries - 1:
                raise e
            import time
            time.sleep(5)

    raise Exception("Не удалось загрузить модель после всех попыток")