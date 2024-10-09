from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.studentfacialbiometric import Base
import pymysql

# Database setup
DATABASE_URL = "mysql+pymysql://root@localhost:3306/facialrecognitionproject?charset=utf8mb4"
engine = create_engine(
    DATABASE_URL,
    connect_args={
        "ssl": {
            "ssl_verify_identity": False,
        }
    }
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Test the connection before creating tables
def test_connection():
    try:
        connection = engine.connect()
        connection.close()
        print("Database connection successful")
        return True
    except Exception as e:
        print(f"Database connection failed: {str(e)}")
        return False

# Only create tables if connection is successful
if test_connection():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()