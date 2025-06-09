from app.domain.usecases.detect_hate_speech_usecase import DetectHateSpeechUseCase, AnalyzeHateSpeechUseCase
from app.presentation.hate_speech.schemas import (
    HateSpeechRequest, 
    HateSpeechDetectionResponse, 
    HateSpeechAnalysisResponse
)
import logging

logger = logging.getLogger(__name__)

class HateSpeechController:
    """
    Controller para endpoints de hate speech
    """
    
    def __init__(
        self, 
        detect_usecase: DetectHateSpeechUseCase,
        analyze_usecase: AnalyzeHateSpeechUseCase
    ):
        self.detect_usecase = detect_usecase
        self.analyze_usecase = analyze_usecase
    
    async def detect_hate_speech(self, request: HateSpeechRequest) -> HateSpeechDetectionResponse:
        """
        Detecta hate speech no texto
        """
        logger.info(f"Requisição de detecção recebida para texto com {len(request.text)} caracteres")
        
        result = self.detect_usecase.execute(request.text)
        
        return HateSpeechDetectionResponse(**result)
    
    async def analyze_hate_speech(self, request: HateSpeechRequest) -> HateSpeechAnalysisResponse:
        """
        Análise detalhada de hate speech
        """
        logger.info(f"Requisição de análise recebida para texto com {len(request.text)} caracteres")
        
        result = self.analyze_usecase.execute(request.text)
        
        return HateSpeechAnalysisResponse(**result)
