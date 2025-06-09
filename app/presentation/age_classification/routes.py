from fastapi import APIRouter
from .controller import classify_age
from .schemas import AgeRatingRequest, AgeRatingResponse

router = APIRouter()

@router.post("/age_classification", response_model=AgeRatingResponse)
def age_rating_endpoint(request: AgeRatingRequest):
    return classify_age(request)
