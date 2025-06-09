class AgeRating:
    def __init__(self, value: int):
        valid_ages = [0, 10, 12, 14, 16, 18]
        if value not in valid_ages:
            raise ValueError(f"Classificação inválida: {value}")
        self.value = value