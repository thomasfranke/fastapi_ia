from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class HateSpeechClassification:
    """Entidade para uma classificação individual"""
    category: str
    confidence: float
    is_hate_speech: bool

@dataclass
class HateSpeechAnalysis:
    """Entidade principal para análise de hate speech"""
    text: str
    is_hate_speech: bool
    confidence_score: float
    classifications: List[HateSpeechClassification]
    detected_categories: List[str]
    analysis_timestamp: datetime
    model_version: str
    fallback_triggered: bool = False
    error_message: Optional[str] = None

    def get_primary_classification(self) -> Optional[HateSpeechClassification]:
        """Retorna a classificação com maior confiança"""
        if not self.classifications:
            return None
        return max(self.classifications, key=lambda x: x.confidence)