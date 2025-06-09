# ===== 3. app/domain/usecases/detect_hate_speech_usecase.py =====
from app.domain.services.hate_speech_detection_service import HateSpeechDetectionService
from app.domain.entities.hate_speech_analysis import HateSpeechAnalysis
import logging

logger = logging.getLogger(__name__)

class DetectHateSpeechUseCase:
    """
    Caso de uso para detecção de hate speech
    """
    
    def __init__(self, hate_speech_service: HateSpeechDetectionService):
        self.hate_speech_service = hate_speech_service
    
    def execute(self, text: str) -> dict:
        """
        Executa a detecção de hate speech
        
        Args:
            text (str): Texto para análise
            
        Returns:
            dict: Resultado da detecção
        """
        try:
            # Validação de entrada
            if not text or not text.strip():
                return {
                    "success": False,
                    "is_hate_speech": False,
                    "error": "Texto vazio ou inválido"
                }
            
            # Executar detecção
            is_hate_speech = self.hate_speech_service.detect_hate_speech(text)
            
            logger.info(f"Hate speech detectado: {is_hate_speech} para texto: {text[:50]}...")
            
            return {
                "success": True,
                "is_hate_speech": is_hate_speech,
                "text_length": len(text),
                "message": "Análise concluída com sucesso"
            }
            
        except Exception as e:
            logger.error(f"Erro no caso de uso de detecção: {e}")
            return {
                "success": False,
                "is_hate_speech": False,
                "error": str(e),
                "message": "Erro na análise - assumindo conteúdo seguro"
            }


class AnalyzeHateSpeechUseCase:
    """
    Caso de uso para análise detalhada de hate speech
    """
    
    def __init__(self, hate_speech_service: HateSpeechDetectionService):
        self.hate_speech_service = hate_speech_service
    
    def execute(self, text: str) -> dict:
        """
        Executa análise detalhada
        
        Args:
            text (str): Texto para análise
            
        Returns:
            dict: Análise detalhada
        """
        try:
            # Validação de entrada
            if not text or not text.strip():
                return {
                    "success": False,
                    "analysis": None,
                    "error": "Texto vazio ou inválido"
                }
            
            # Executar análise
            analysis = self.hate_speech_service.analyze_text(text)
            
            logger.info(f"Análise concluída para texto: {text[:50]}...")
            
            return {
                "success": True,
                "analysis": {
                    "text": analysis.text,
                    "is_hate_speech": analysis.is_hate_speech,
                    "confidence_score": analysis.confidence_score,
                    "detected_categories": analysis.detected_categories,
                    "classifications": [
                        {
                            "category": c.category,
                            "confidence": c.confidence,
                            "is_hate_speech": c.is_hate_speech
                        }
                        for c in analysis.classifications
                    ],
                    "analysis_timestamp": analysis.analysis_timestamp.isoformat(),
                    "model_version": analysis.model_version,
                    "fallback_triggered": analysis.fallback_triggered,
                    "error_message": analysis.error_message
                }
            }
            
        except Exception as e:
            logger.error(f"Erro no caso de uso de análise: {e}")
            return {
                "success": False,
                "analysis": None,
                "error": str(e)
            }
