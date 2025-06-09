# app/domain/value_objects/text_content.py
class TextContent:
    def __init__(self, text: str):
        if not text or not text.strip():
            raise ValueError("Texto não pode estar vazio")
        self.value = text.strip()