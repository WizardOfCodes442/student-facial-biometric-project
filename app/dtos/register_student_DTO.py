from pydantic import BaseModel

class RegisterStudentDTO(BaseModel):
    name: str
    matric_number: str
    faculty: str
    department: str
    level: str

    