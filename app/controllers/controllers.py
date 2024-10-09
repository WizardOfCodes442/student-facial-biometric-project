import os
from fastapi import FastAPI, APIRouter, UploadFile, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from dtos.register_student_DTO import RegisterStudentDTO
from dtos.recognition_response_DTO import RecognitionResponseDTO
from models.studentfacialbiometric import StudentFacialBiometric  # Correct import
from utils.s3_media_uploader import S3MediaUploader
from face_recognition import OptimizedFaceRecognitionSystem
from db import get_db
import json
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Enable CORS
origins = [
    "*",  # Frontend
    # Frontend URL
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

router = APIRouter()

# Initialize systems
face_recognition_system = OptimizedFaceRecognitionSystem()
s3_uploader = S3MediaUploader(
    bucket_name=os.getenv("S3_BUCKET_NAME"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

@router.post("/register", response_model=dict)
async def register_student(
    data: str = Form(...),
    image1: UploadFile = Form(...),
    image2: UploadFile = Form(...),
    image3: UploadFile = Form(...),
    db: Session = Depends(get_db)
):
    try:
        parsed_data = json.loads(data)
        dto = RegisterStudentDTO(**parsed_data)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")

    # Upload files to S3 and store URLs
    file_urls = []
    for i, image in enumerate([image1, image2, image3], start=1):
        file_url = await s3_uploader.upload_file(image, f"{dto.matric_number}_img{i}.jpg")
        file_urls.append(file_url)

    # Extract face encodings from all three images
    encodings = []
    for i, image in enumerate([image1, image2, image3]):
        face = await face_recognition_system._load_image(image)
        detected_face = face_recognition_system._detect_face(face)

        if detected_face is None:
            raise HTTPException(status_code=400, detail=f"No face detected in image {i+1}")

        face_encoding = face_recognition_system._extract_face_encoding(detected_face)
        encodings.append(face_encoding)

    # Compute the average face encoding
    avg_encoding = np.mean(encodings, axis=0)

    # Create a new student record
    new_student = StudentFacialBiometric(
        name=dto.name,
        matric_number=dto.matric_number,
        faculty=dto.faculty,
        department=dto.department,
        level=dto.level,
        avg_face_encoding=avg_encoding.tobytes(),  # Store avg face encoding
        media_url=json.dumps(file_urls)  # Store URLs as a JSON-encoded string
    )

    db.add(new_student)
    db.commit()

    return {"message": f"Student {dto.name} registered successfully"}

@router.post("/verify", response_model=RecognitionResponseDTO)
async def verify_student(image: UploadFile, db: Session = Depends(get_db)):
    # Use the face recognition system to process the image and recognize the student
    recognition_result = await face_recognition_system.recognize_face(image, db)

    if recognition_result is None:
        raise HTTPException(status_code=404, detail="Student not recognized")

    return recognition_result
