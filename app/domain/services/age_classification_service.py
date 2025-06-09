# app/domain/services/age_classification_service.py
from abc import ABC, abstractmethod

class AgeClassificationService(ABC):
    @abstractmethod
    def classify(self, text: str) -> int:
        pass