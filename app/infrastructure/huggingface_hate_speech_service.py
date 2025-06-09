from app.domain.services.hate_speech_detection_service import HateSpeechDetectionService
from app.domain.entities.hate_speech_analysis import HateSpeechAnalysis, HateSpeechClassification
from transformers import pipeline
from datetime import datetime
import torch
import logging

logger = logging.getLogger(__name__)

class HuggingFaceHateSpeechService(HateSpeechDetectionService):
    """
    Implementação melhorada do serviço de detecção usando Hugging Face
    """
    
    MODEL_VERSION = "1.1.0"
    
    def __init__(self):
        self._initialize_models()
        self._setup_configuration()
    
    def _initialize_models(self):
        """Inicializa os modelos de ML"""
        try:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            
            # Múltiplos modelos para melhor detecção
            self.models = {}
            
            # Modelo 1: BERT para toxicidade (mais sensível)
            try:
                self.models['toxic_bert'] = pipeline(
                    "text-classification",
                    model="unitary/toxic-bert",
                    device=device
                )
                logger.info("Toxic BERT carregado com sucesso")
            except Exception as e:
                logger.warning(f"Erro ao carregar Toxic BERT: {e}")
                self.models['toxic_bert'] = None
            
            # Modelo 2: Modelo alternativo para hate speech
            try:
                self.models['hate_speech'] = pipeline(
                    "text-classification", 
                    model="martin-ha/toxic-comment-model",
                    device=device
                )
                logger.info("Hate Speech model carregado com sucesso")
            except Exception as e:
                logger.warning(f"Erro ao carregar hate speech model: {e}")
                self.models['hate_speech'] = None
            
            # Modelo 3: Zero-shot para análise contextual
            try:
                self.models['zero_shot'] = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=device
                )
                logger.info("Zero-shot model carregado com sucesso")
            except Exception as e:
                logger.warning(f"Erro ao carregar zero-shot: {e}")
                self.models['zero_shot'] = None
            
            logger.info(f"Modelos inicializados no device: {device}")
            
        except Exception as e:
            logger.error(f"Erro geral ao inicializar modelos: {e}")
            raise
    
    def _setup_configuration(self):
        """Configura labels e thresholds com valores mais sensíveis"""
        
        # Labels em português mais específicos
        self.hate_speech_labels = [
            "discurso de ódio extremo e violento",
            "incitação à violência e eliminação",
            "desumanização e comparação com pragas",
            "linguagem discriminatória severa",
            "ameaças e intimidação",
            "bullying e assédio grave",
            "conteúdo tóxico moderado",
            "linguagem ofensiva leve",
            "crítica construtiva",
            "conteúdo neutro e respeitoso"
        ]
        
        # Thresholds mais baixos para maior sensibilidade
        self.hate_threshold = 0.4  # Reduzido de 0.65
        self.toxic_threshold = 0.5  # Reduzido de 0.7
        
        # Indicadores de hate speech
        self.hate_indicators = [
            "discurso de ódio extremo e violento",
            "incitação à violência e eliminação", 
            "desumanização e comparação com pragas",
            "linguagem discriminatória severa",
            "ameaças e intimidação",
            "bullying e assédio grave",
            "conteúdo tóxico moderado"
        ]
        
        # Palavras-chave expandidas e mais específicas
        self.hate_keywords = [
            # Violência e eliminação
            "eliminar", "eliminadas", "eliminados", "exterminar", "extermínio",
            "matar", "morrer", "morte", "assassinar", "acabar com",
            
            # Desumanização
            "praga", "pragas", "parasita", "parasitas", "lixo", "escória",
            "verme", "vermes", "animal", "animais", "coisa", "coisas",
            
            # Ódio e discriminação
            "ódio", "odio", "nojo", "repugnante", "asqueroso", "nojento",
            "inferior", "inferiores", "superiores", "raça", "espécie",
            
            # Violência direta
            "violência", "agressão", "atacar", "destruir", "aniquilar",
            "sumir", "desaparecer", "banir", "expulsar"
        ]
        
        # Padrões perigosos (frases completas)
        self.dangerous_patterns = [
            "deveriam ser eliminad",
            "são uma praga",
            "não merecem viver",
            "mundo seria melhor sem",
            "deveria morrer", 
            "não são humanos",
            "raça inferior",
            "merecem sofrer"
        ]
    
    def detect_hate_speech(self, text: str) -> bool:
        """
        Detecção melhorada com múltiplas camadas
        """
        if not text or not text.strip():
            return False
            
        text_lower = text.lower()
        logger.info(f"=== ANALISANDO: {text} ===")
        
        # Camada 1: Detecção por padrões perigosos (mais rápida e precisa)
        pattern_detected = self._detect_dangerous_patterns(text_lower)
        if pattern_detected:
            logger.warning(f"PADRÃO PERIGOSO DETECTADO: {pattern_detected}")
            return True
        
        # Camada 2: Detecção por palavras-chave
        keyword_detected = self._detect_keywords(text_lower)
        if keyword_detected:
            logger.warning(f"PALAVRA-CHAVE DETECTADA: {keyword_detected}")
            # Se tem palavra-chave + contexto violento, é hate speech
            if self._has_violent_context(text_lower):
                return True
        
        # Camada 3: Modelos de ML
        ml_detected = self._detect_with_ml_models(text)
        if ml_detected:
            logger.warning("HATE SPEECH DETECTADO POR ML")
            return True
        
        logger.info("NENHUM HATE SPEECH DETECTADO")
        return False
    
    def _detect_dangerous_patterns(self, text: str) -> str:
        """Detecta padrões específicos perigosos"""
        for pattern in self.dangerous_patterns:
            if pattern in text:
                return pattern
        return None
    
    def _detect_keywords(self, text: str) -> str:
        """Detecta palavras-chave"""
        for keyword in self.hate_keywords:
            if keyword in text:
                return keyword
        return None
    
    def _has_violent_context(self, text: str) -> bool:
        """Verifica se há contexto violento"""
        violent_words = ["eliminar", "matar", "morrer", "violência", "destruir", "praga", "inferior"]
        count = sum(1 for word in violent_words if word in text)
        return count >= 2  # Se tem 2+ palavras violentas, é contexto violento
    
    def _detect_with_ml_models(self, text: str) -> bool:
        """Detecção usando modelos de ML"""
        detections = []
        
        # Teste cada modelo disponível
        for model_name, model in self.models.items():
            if model is None:
                continue
                
            try:
                if model_name == 'zero_shot':
                    detected = self._detect_with_zero_shot(text, model)
                else:
                    detected = self._detect_with_classifier(text, model, model_name)
                
                detections.append(detected)
                logger.info(f"Modelo {model_name}: {detected}")
                
            except Exception as e:
                logger.error(f"Erro no modelo {model_name}: {e}")
        
        # Se qualquer modelo detectou, considera hate speech
        return any(detections)
    
    def _detect_with_classifier(self, text: str, model, model_name: str) -> bool:
        """Detecção com modelos de classificação"""
        try:
            result = model(text)
            if isinstance(result, list):
                result = result[0]
            
            label = result.get('label', '').upper()
            score = result.get('score', 0)
            
            logger.info(f"{model_name} - Label: {label}, Score: {score}")
            
            # Labels que indicam toxicidade/hate speech
            toxic_labels = ['TOXIC', 'HATE', 'OFFENSIVE', '1', 'POSITIVE', 'LABEL_1']
            
            # Threshold mais baixo para maior sensibilidade
            threshold = 0.3 if model_name == 'toxic_bert' else 0.5
            
            return label in toxic_labels and score > threshold
            
        except Exception as e:
            logger.error(f"Erro na classificação {model_name}: {e}")
            return False
    
    def _detect_with_zero_shot(self, text: str, model) -> bool:
        """Detecção com zero-shot"""
        try:
            result = model(text, self.hate_speech_labels)
            
            top_label = result['labels'][0]
            confidence = result['scores'][0]
            
            logger.info(f"Zero-shot - Top: {top_label}, Score: {confidence}")
            
            # Verifica se é categoria de hate speech com threshold baixo
            is_hate = top_label in self.hate_indicators and confidence > self.hate_threshold
            
            # Log das top 3 classificações
            for i in range(min(3, len(result['labels']))):
                logger.info(f"  {i+1}. {result['labels'][i]}: {result['scores'][i]:.3f}")
            
            return is_hate
            
        except Exception as e:
            logger.error(f"Erro no zero-shot: {e}")
            return False
    
    def analyze_text(self, text: str) -> HateSpeechAnalysis:
        """
        Análise detalhada melhorada
        """
        classifications = []
        detected_categories = []
        confidence_score = 0.0
        fallback_triggered = False
        error_message = None
        
        # Análise de padrões
        pattern_detected = self._detect_dangerous_patterns(text.lower())
        if pattern_detected:
            detected_categories.append(f"Padrão perigoso: {pattern_detected}")
            confidence_score = 0.95
        
        # Análise de palavras-chave
        keyword_detected = self._detect_keywords(text.lower())
        if keyword_detected:
            detected_categories.append(f"Palavra-chave: {keyword_detected}")
            confidence_score = max(confidence_score, 0.8)
        
        # Análise com modelos ML
        try:
            if self.models['zero_shot']:
                result = self.models['zero_shot'](text, self.hate_speech_labels)
                
                for label, score in zip(result['labels'], result['scores']):
                    is_hate = label in self.hate_indicators
                    classifications.append(
                        HateSpeechClassification(
                            category=label,
                            confidence=score,
                            is_hate_speech=is_hate and score > self.hate_threshold
                        )
                    )
                    
                    if is_hate and score > self.hate_threshold:
                        detected_categories.append(f"ML: {label}")
                        confidence_score = max(confidence_score, score)
                        
        except Exception as e:
            logger.error(f"Erro na análise ML: {e}")
            fallback_triggered = True
            error_message = str(e)
        
        # Decisão final
        is_hate_speech = self.detect_hate_speech(text)
        
        return HateSpeechAnalysis(
            text=text,
            is_hate_speech=is_hate_speech,
            confidence_score=confidence_score,
            classifications=classifications,
            detected_categories=detected_categories,
            analysis_timestamp=datetime.now(),
            model_version=self.MODEL_VERSION,
            fallback_triggered=fallback_triggered,
            error_message=error_message
        )
