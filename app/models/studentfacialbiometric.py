from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
import json

Base = declarative_base()

class StudentFacialBiometric(Base):
    __tablename__ = 'student_facial_biometric'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    matric_number = Column(String(100), nullable=False, unique=True)
    faculty = Column(String(255), nullable=False)
    department = Column(String(255), nullable=False)
    level = Column(String(10), nullable=False)
    avg_face_encoding = Column(Text, nullable=False)
    media_url = Column(Text, nullable=False)  # Store list of URLs as JSON string

    def set_media_url(self, urls: list):
        self.media_url = json.dumps(urls)

    def get_media_url(self):
        return json.loads(self.media_url)
