# Importa a interface do domínio que esta classe implementa, garantindo que essa
# implementação siga o contrato definido para serviços de classificação etária.
from app.domain.services.age_classification_service import AgeClassificationService

# Importa o pipeline da biblioteca Hugging Face Transformers, que facilita o uso
# de modelos pré-treinados para diversas tarefas de NLP (Processamento de Linguagem Natural).
from transformers import pipeline

# Importa PyTorch, uma das principais bibliotecas de machine learning, utilizada
# para treinar e executar modelos de deep learning de forma eficiente.
import torch


class HuggingFaceAgeService(AgeClassificationService):
    def __init__(self):
        """
        Inicializa o serviço de classificação etária usando um modelo pré-treinado da Hugging Face.
        
        Explicações importantes:
        
        - Hugging Face: é uma plataforma e biblioteca open-source que fornece acesso fácil
          a modelos de NLP avançados e pré-treinados, como BERT, GPT, BART, entre outros.
          O pipeline facilita o uso desses modelos para tarefas específicas, sem a necessidade
          de treinar do zero.
        
        - PyTorch: biblioteca de machine learning que oferece suporte à computação acelerada
          via GPU e facilita a criação e execução de redes neurais. É utilizada pela Hugging Face
          como backend para rodar os modelos.
        
        - Zero-shot classification (classificação zero-shot): técnica que permite classificar
          textos em categorias definidas **sem que o modelo tenha sido treinado especificamente
          nessas categorias**. O modelo entende a relação entre o texto e as labels com base
          no conhecimento aprendido durante seu treinamento geral.
        
        - device="mps": usa a GPU integrada dos Macs com Apple Silicon para acelerar o processamento.
        """
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",  # Modelo para zero-shot classification
            device="mps"  # Usa Metal Performance Shaders para aceleração no Apple Mac
        )
        
        # Labels que representam categorias de conteúdo para avaliar a faixa etária recomendada.
        self.age_labels = [
            "conteúdo adequado para todas as idades",
            "conteúdo infantil e educativo",
            "conteúdo com violência leve ou aventura",
            "conteúdo com conflitos e suspense",
            "conteúdo com violência moderada",
            "conteúdo com violência intensa ou temas adultos",
            "conteúdo extremamente violento ou perturbador"
        ]
        
        # Mapeamento das labels para uma idade mínima recomendada.
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
        """
        Classifica um texto para determinar a faixa etária mínima recomendada.
        
        Parâmetro:
        - text (str): texto a ser classificado.
        
        Retorna:
        - int: idade mínima recomendada para o conteúdo do texto.
        
        A classificação é baseada na label com maior score (confiança) dada pelo modelo.
        Ajusta a idade recomendada dependendo da confiança para evitar falsos positivos.
        """
        try:
            # Executa a classificaçãousando o modelo pré-treinado do Hugging Face.
            # O modelo avalia o texto fornecido e calcula a probabilidade de ele
            # se enquadrar em cada uma das categorias (labels) definidas em self.age_labels,
            # mesmo sem ter sido treinado especificamente para essa tarefa.
            # Essa abordagem permite classificar o texto em múltiplas categorias,
            # retornando a confiança para cada uma delas.
            result = self.classifier(text, self.age_labels)
            
            # Obtém a label com maior confiança, ou seja, a categoria que o modelo considera mais provável para o texto.
            top_label = result['labels'][0]

            # Obtém o score (nível de confiança) associado a essa label principal, indicando quão seguro o modelo está dessa classificação.
            confidence = result['scores'][0]

            print(f"Top classification: {top_label}")
            print(f"Confidence: {confidence:.4f}")
            print(f"All scores: {dict(zip(result['labels'][:3], result['scores'][:3]))}")
            print(f"Text: {text[:100]}...")  # Apenas os primeiros 100 caracteres do texto para debug

            # Converte a label principal (categoria mais provável) para a idade mínima recomendada correspondente,
            # usando o dicionário de mapeamento label_to_age.
            # Caso a label não esteja no dicionário, utiliza o valor padrão 10 como faixa etária segura.
            age = self.label_to_age.get(top_label, 10)

            # Ajusta a idade para mais conservador se a confiança for baixa
            if confidence < 0.3:
                age = max(age - 2, 0)
                print(f"Baixa confiança, ajustando idade para: {age}")
            elif confidence > 0.8:
                print(f"Alta confiança na classificação: {age}")

            return age

        except Exception as e:
            print(f"Erro na classificação: {e}")
            # Fallback simples para casos de erro: baseia-se na complexidade do texto
            word_count = len(text.split())
            if word_count > 100:
                return 12  # Textos longos são considerados mais complexos e para faixa etária maior
            else:
                return 10  # Valor padrão conservador
    
    # def get_detailed_analysis(self, text: str) -> dict:
    #     """
    #     Realiza uma análise detalhada do texto, fornecendo as pontuações
    #     para todas as categorias possíveis de faixa etária.
        
    #     Retorna:
    #     - dict: contém o texto original e uma lista com cada categoria,
    #       sua pontuação (confiança) e a idade mínima associada.
    #     """
    #     try:
    #         result = self.classifier(text, self.age_labels)
            
    #         analysis = {
    #             "text": text,
    #             "classifications": []
    #         }

    #         # Preenche a lista de classificações com dados detalhados
    #         for label, score in zip(result['labels'], result['scores']):
    #             analysis["classifications"].append({
    #                 "category": label,
    #                 "confidence": round(score, 4),
    #                 "age_rating": self.label_to_age.get(label, 10)
    #             })
            
    #         return analysis

    #     except Exception as e:
    #         # Retorna o erro no formato de dicionário para facilitar o tratamento externo
    #         return {"error": str(e), "text": text}
