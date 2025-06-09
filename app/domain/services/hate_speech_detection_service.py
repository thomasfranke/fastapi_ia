# ===== 2. app/domain/services/hate_speech_detection_service.py =====
from abc import ABC, abstractmethod
from app.domain.entities.hate_speech_analysis import HateSpeechAnalysis

class HateSpeechDetectionService(ABC):
    """
    Interface do Domain Service para detecção de discurso de ódio
    """
    
    @abstractmethod
    def detect_hate_speech(self, text: str) -> bool:
        """
        Detecta se o texto contém discurso de ódio
        
        Args:
            text (str): Texto a ser analisado
            
        Returns:
            bool: True se contém discurso de ódio, False caso contrário
        """
        pass
    
    @abstractmethod
    def analyze_text(self, text: str) -> HateSpeechAnalysis:
        """
        Retorna análise completa do texto
        
        Args:
            text (str): Texto a ser analisado
            
        Returns:
            HateSpeechAnalysis: Análise detalhada
        """
        pass