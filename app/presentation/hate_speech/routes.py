# ===== 7. app/presentation/hate_speech/routes.py =====
from fastapi import APIRouter, HTTPException, Depends
from app.presentation.hate_speech.controller import HateSpeechController
from app.presentation.hate_speech.schemas import (
    HateSpeechRequest,
    HateSpeechDetectionResponse,
    HateSpeechAnalysisResponse
)
from app.domain.usecases.detect_hate_speech_usecase import DetectHateSpeechUseCase, AnalyzeHateSpeechUseCase
from app.infrastructure.huggingface_hate_speech_service import HuggingFaceHateSpeechService

router = APIRouter(prefix="/hate_speech", tags=["Hate Speech Detection"])

# Dependency Injection
def get_hate_speech_service():
    return HuggingFaceHateSpeechService()

def get_detect_usecase(service = Depends(get_hate_speech_service)):
    return DetectHateSpeechUseCase(service)

def get_analyze_usecase(service = Depends(get_hate_speech_service)):
    return AnalyzeHateSpeechUseCase(service)

def get_controller(
    detect_usecase = Depends(get_detect_usecase),
    analyze_usecase = Depends(get_analyze_usecase)
):
    return HateSpeechController(detect_usecase, analyze_usecase)

# Endpoints
@router.post("/detect", response_model=HateSpeechDetectionResponse)
async def detect_hate_speech(
    request: HateSpeechRequest,
    controller: HateSpeechController = Depends(get_controller)
):
    """
    Detecta se o texto contém discurso de ódio
    
    - **text**: Texto a ser analisado (1-5000 caracteres)
    
    Retorna:
    - **is_hate_speech**: true se detectado hate speech, false caso contrário
    - **success**: indica se a análise foi bem-sucedida
    """
    try:
        return await controller.detect_hate_speech(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze", response_model=HateSpeechAnalysisResponse)
async def analyze_hate_speech(
    request: HateSpeechRequest,
    controller: HateSpeechController = Depends(get_controller)
):
    """
    Análise detalhada de discurso de ódio
    
    - **text**: Texto a ser analisado (1-5000 caracteres)
    
    Retorna análise completa com:
    - Classificações por categoria
    - Scores de confiança
    - Categorias detectadas
    - Metadados da análise
    """
    try:
        return await controller.analyze_hate_speech(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))