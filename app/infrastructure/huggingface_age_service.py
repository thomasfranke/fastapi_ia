from app.domain.services.age_classification_service import AgeClassificationService
from transformers import pipeline
import torch

class HuggingFaceAgeService(AgeClassificationService):
    def __init__(self):
        
        # RoBERTa Large para zero-shot classification
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",  # Melhor para zero-shot
            device="mps"
        )
        
        # Labels para classificação
        self.age_labels = [
            "conteúdo adequado para todas as idades",
            "conteúdo infantil e educativo",
            "conteúdo com violência leve ou aventura",
            "conteúdo com conflitos e suspense",
            "conteúdo com violência moderada",
            "conteúdo com violência intensa ou temas adultos",
            "conteúdo extremamente violento ou perturbador"
        ]
        
        # Mapeamento de labels para idades
        self.label_to_age = {
            "conteúdo adequado para todas as idades": 0,
            "conteúdo infantil e educativo": 0,
            "conteúdo com violência leve ou aventura": 10,
            "conteúdo com conflitos e suspense": 12,
            "conteúdo com violência moderada": 14,
            "conteúdo com violência intensa ou temas adultos": 16,
            "conteúdo extremamente violento ou perturbador": 18
        }
    
    def classify(self, text: str) -> int:
        try:
            result = self.classifier(text, self.age_labels)
            
            top_label = result['labels'][0]
            confidence = result['scores'][0]
            
            print(f"Top classification: {top_label}")
            print(f"Confidence: {confidence:.4f}")
            print(f"All scores: {dict(zip(result['labels'][:3], result['scores'][:3]))}")
            print(f"Text: {text[:100]}...")
            
            age = self.label_to_age.get(top_label, 10)
            
            # Ajustar baseado na confiança
            if confidence < 0.3:  # Baixa confiança
                age = max(age - 2, 0)  # Ser mais conservador
                print(f"Baixa confiança, ajustando idade para: {age}")
            elif confidence > 0.8:  # Alta confiança
                print(f"Alta confiança na classificação: {age}")
            
            return age
            
        except Exception as e:
            print(f"Erro na classificação: {e}")
            # Fallback: análise simples baseada em comprimento e complexidade
            word_count = len(text.split())
            if word_count > 100:
                return 12  # Textos longos tendem a ser mais complexos
            else:
                return 10  # Padrão seguro
    
    def get_detailed_analysis(self, text: str) -> dict:
        """Método adicional para análise detalhada"""
        try:
            result = self.classifier(text, self.age_labels)
            
            analysis = {
                "text": text,
                "classifications": []
            }
            
            for label, score in zip(result['labels'], result['scores']):
                analysis["classifications"].append({
                    "category": label,
                    "confidence": round(score, 4),
                    "age_rating": self.label_to_age.get(label, 10)
                })
            
            return analysis
            
        except Exception as e:
            return {"error": str(e), "text": text}
