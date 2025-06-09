from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class HateSpeechRequest(BaseModel):
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=5000, 
        description="Texto para análise de hate speech"
    )

class HateSpeechDetectionResponse(BaseModel):
    success: bool
    is_hate_speech: bool
    message: Optional[str] = None
    error: Optional[str] = None
    text_length: Optional[int] = None

class ClassificationDetail(BaseModel):
    category: str
    confidence: float
    is_hate_speech: bool

class HateSpeechAnalysisResponse(BaseModel):
    success: bool
    analysis: Optional[dict] = None
    error: Optional[str] = None

class HateSpeechAnalysisDetail(BaseModel):
    text: str
    is_hate_speech: bool
    confidence_score: float
    detected_categories: List[str]
    classifications: List[ClassificationDetail]
    analysis_timestamp: str
    model_version: str
    fallback_triggered: bool
    error_message: Optional[str] = None