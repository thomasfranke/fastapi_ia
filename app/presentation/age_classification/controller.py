# controller.py
from app.domain.usecases.age_classification_usecase import AgeClassificationUseCase
from app.infrastructure.huggingface_age_service import HuggingFaceAgeService
from .schemas import AgeRatingRequest

def classify_age(request: AgeRatingRequest) -> dict:
    # Criar o service
    service = HuggingFaceAgeService()
    
    # Criar o usecase com o service
    usecase = AgeClassificationUseCase(service)
    
    rating = usecase.execute(request.text)
    return {"rating": f"{rating}+"}