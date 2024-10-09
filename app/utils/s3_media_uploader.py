import boto3
from fastapi import UploadFile
from botocore.exceptions import ClientError
import logging
from typing import Optional

class S3MediaUploader:
    def __init__(self, bucket_name: str, aws_access_key_id: str, aws_secret_access_key: str, region_name: str = 'us-east-1'):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        self.bucket_name = bucket_name

    async def upload_file(self, file: UploadFile, destination_name: str) -> Optional[str]:
        try:
            # Read the file content
            file_content = await file.read()
            
            # Upload the file to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=destination_name,
                Body=file_content,
                ContentType=file.content_type
            )
            
            # Generate the URL for the uploaded file
            url = f"https://{self.bucket_name}.s3.amazonaws.com/{destination_name}"
            return url
        except ClientError as e:
            logging.error(f"An error occurred: {e}")
            return None
        finally:
            await file.seek(0)  # Reset file pointer
