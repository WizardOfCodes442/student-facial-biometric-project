import cv2
import numpy as np
from fastapi import UploadFile
from sqlalchemy.orm import Session
from sklearn.metrics.pairwise import cosine_similarity
import aiofiles
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from models import StudentFacialBiometric
from utils import CNNModel

CONFIDENCE_THRESHOLD = 0.9

class OptimizedFaceRecognitionSystem:
    def __init__(self):
        self.model = CNNModel.get_instance().model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    async def recognize_face(self, image_file: UploadFile, db: Session) -> dict:
        image = await self._load_image(image_file)
        face = self._detect_face(image)

        if face is None:
            return None

        face_encoding = self._extract_face_encoding(face)

    # Retrieve all students' encodings from the database
        students = db.query(StudentFacialBiometric).all()
    
    # Convert each student's face encoding from bytes to numpy array
        student_encodings = np.array([np.frombuffer(student.avg_face_encoding.encode('latin1'), dtype=np.float32) for student in students])

    # Compare the face encoding with the stored encodings
        similarities = cosine_similarity([face_encoding], student_encodings)[0]
        best_match_index = np.argmax(similarities)
        best_confidence = similarities[best_match_index]
        print(best_confidence)

        if best_confidence > CONFIDENCE_THRESHOLD:
            best_match = students[best_match_index]
            return {
                "name": best_match.name,
                "faculty": best_match.faculty,
                "department": best_match.department,
                "level": best_match.level,
                "matric_number": best_match.matric_number,
                "confidence": str(best_confidence),
                "media_url": best_match.media_url
            }
        return None


    async def _load_image(self, image_file: UploadFile) -> np.ndarray:
        async with aiofiles.tempfile.NamedTemporaryFile("wb", delete=False) as temp_file:
            content = await image_file.read()
            await temp_file.write(content)
            temp_file_path = temp_file.name

        image = cv2.imread(temp_file_path)
        os.unlink(temp_file_path)
        return image

    def _detect_face(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]
        detected_face = image[y:y+h, x:x+w]
        return detected_face

    def _extract_face_encoding(self, face: np.ndarray) -> np.ndarray:
        resized_face = cv2.resize(face, (224, 224))
        processed_face = preprocess_input(resized_face)
        return self.model.predict(np.expand_dims(processed_face, axis=0)).flatten()
