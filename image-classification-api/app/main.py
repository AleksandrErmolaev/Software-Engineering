from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List

from .models import classifier
from .schema import PredictionResponse, ErrorResponse, HealthResponse

app = FastAPI(
    title="Image Classification API",
    description="API для классификации изображений с использованием ResNet50",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Image Classification API работает! Перейдите на /docs для документации."}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка статуса API и модели"""
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name="ResNet50"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_image(
        file: UploadFile = File(..., description="Изображение для классификации"),
        top_k: int = Query(5, ge=1, le=10, description="Количество топ-предсказаний")
):
    """
    Классификация изображения из файла
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="Файл должен быть изображением"
        )

    try:
        contents = await file.read()

        result = classifier.predict_from_file(contents, top_k)

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])

        return PredictionResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")


@app.post("/predict/url")
async def predict_from_url(
        url: str = Query(..., description="URL изображения для классификации"),
        top_k: int = Query(5, ge=1, le=10, description="Количество топ-предсказаний")
):
    """
    Классификация изображения по URL
    """
    try:
        result = classifier.predict_from_url(url, top_k)

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])

        return PredictionResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки URL: {str(e)}")


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            status="error",
            message=exc.detail
        ).dict()
    )


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)