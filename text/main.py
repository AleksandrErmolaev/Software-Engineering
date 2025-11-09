from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
from model_loader import create_classifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sentiment Analysis API",
    description="API для анализа тональности текста на русском языке",
    version="1.0.0"
)

classifier = None


@app.on_event("startup")
async def load_model():
    global classifier
    try:
        logger.info("Загрузка модели sentiment analysis...")
        classifier = create_classifier()

        test_text = "Это тестовый текст"
        result = classifier([test_text])
        logger.info(f"Модель успешно загружена. Тестовый результат: {result[0]}")

    except Exception as e:
        logger.error(f"Критическая ошибка при загрузке модели: {e}")
        raise


class TextItem(BaseModel):
    text: str


class AnalysisResult(BaseModel):
    text: str
    label: str
    score: float
    sentiment: Optional[str] = None


class BatchRequest(BaseModel):
    texts: List[str]


class BatchResponse(BaseModel):
    results: List[AnalysisResult]


@app.get("/")
async def root():
    return {"message": "Sentiment Analysis API", "status": "active"}


@app.get("/health")
async def health_check():
    if classifier is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    try:
        test_result = classifier(["Тест"])[0]
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_ready": True
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Модель не работает: {e}")


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_sentiment(item: TextItem):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    try:
        if not item.text.strip():
            raise HTTPException(status_code=400, detail="Текст не может быть пустым")

        result = classifier([item.text])[0]

        sentiment_map = {
            "POSITIVE": "positive",
            "NEGATIVE": "negative",
            "NEUTRAL": "neutral"
        }

        return AnalysisResult(
            text=item.text,
            label=result['label'],
            score=result['score'],
            sentiment=sentiment_map.get(result['label'], "unknown")
        )
    except Exception as e:
        logger.error(f"Ошибка при анализе текста: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {e}")


@app.post("/analyze-batch", response_model=BatchResponse)
async def analyze_sentiment_batch(request: BatchRequest):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="Список текстов не может быть пустым")

        valid_texts = [text for text in request.texts if text.strip()]
        if not valid_texts:
            raise HTTPException(status_code=400, detail="Нет валидных текстов для анализа")

        results = classifier(valid_texts)

        sentiment_map = {
            "POSITIVE": "positive",
            "NEGATIVE": "negative",
            "NEUTRAL": "neutral"
        }

        analysis_results = []
        for text, result in zip(valid_texts, results):
            analysis_results.append(
                AnalysisResult(
                    text=text,
                    label=result['label'],
                    score=result['score'],
                    sentiment=sentiment_map.get(result['label'], "unknown")
                )
            )

        return BatchResponse(results=analysis_results)
    except Exception as e:
        logger.error(f"Ошибка при пакетном анализе: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)