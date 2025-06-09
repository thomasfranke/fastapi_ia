from pydantic import BaseModel

class AgeRatingRequest(BaseModel):
    text: str

class AgeRatingResponse(BaseModel):
    rating: str  # Ex: "13+", "18+"
