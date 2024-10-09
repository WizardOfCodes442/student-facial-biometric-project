from pydantic import BaseModel
from typing import List


class RecognitionResponseDTO(BaseModel):
    name: str
    faculty: str
    department: str
    matric_number: str
    level: str
    confidence: str 
    media_url: str