from app.domain.services.age_classification_service import AgeClassificationService
from app.domain.value_objects.text_content import TextContent
from app.domain.entities.age_rating import AgeRating

class AgeClassificationUseCase:
    def __init__(self, classification_service: AgeClassificationService):
        self.classification_service = classification_service
    
    def execute(self, text: str) -> int:
        content = TextContent(text)
        age_value = self.classification_service.classify(content.value)
        rating = AgeRating(age_value)
        return rating.value