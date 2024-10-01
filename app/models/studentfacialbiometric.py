from sqlalchemy import Column, String, Integer, LargeBinary
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class StudentFacialBiometric(Base):
    __tablename__ = 'student_facial_biometric'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    matric_number = Column(String(100), nullable=False, unique=True)
    faculty = Column(String(255), nullable=False)
    department = Column(String(255), nullable=False)
    level = Column(String(10), nullable=False)
    face_encoding = Column(LargeBinary, nullable=False)
    media_url = Column(String(500), nullable=True)
