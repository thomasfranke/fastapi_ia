from fastapi import FastAPI
from app.presentation.age_classification.routes import router as age_classification_router
from app.presentation.hate_speech.routes import router as hate_speech_router

app = FastAPI()

app.include_router(age_classification_router, prefix="/ia", tags=["Age Rating"])
app.include_router(hate_speech_router, prefix="/ia", tags=["Hate Speech Detection"])
