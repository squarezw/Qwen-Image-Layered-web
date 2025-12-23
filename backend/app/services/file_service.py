"""
File Service - Handle file upload, storage, and management
"""
import os
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Optional
from PIL import Image
from fastapi import UploadFile
import aiofiles
from ..config import settings


class FileService:
    """Service for handling file operations"""

    def __init__(self):
        self.storage_path = settings.file_storage_path
        self._ensure_storage_directory()

    def _ensure_storage_directory(self):
        """Ensure storage directory exists"""
        os.makedirs(self.storage_path, exist_ok=True)

    async def save_uploaded_file(
        self,
        file: UploadFile,
        task_id: str
    ) -> tuple[str, str]:
        """
        Save uploaded file to storage

        Args:
            file: Uploaded file
            task_id: Task identifier

        Returns:
            Tuple of (file_path, file_hash)
        """
        # Create task directory
        task_dir = os.path.join(self.storage_path, task_id)
        os.makedirs(task_dir, exist_ok=True)

        # Generate file path
        file_ext = os.path.splitext(file.filename or "image.png")[1]
        file_path = os.path.join(task_dir, f"input{file_ext}")

        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        # Calculate hash
        file_hash = hashlib.sha256(content).hexdigest()

        return file_path, file_hash

    async def validate_image_file(self, file: UploadFile) -> bool:
        """
        Validate uploaded image file

        Args:
            file: Uploaded file

        Returns:
            True if valid, raises exception otherwise
        """
        # Check file size
        content = await file.read()
        file_size_mb = len(content) / (1024 * 1024)
        await file.seek(0)  # Reset file pointer

        if file_size_mb > settings.max_file_size_mb:
            raise ValueError(f"File size ({file_size_mb:.2f}MB) exceeds maximum allowed ({settings.max_file_size_mb}MB)")

        # Check file type
        try:
            image = Image.open(file.file)
            image.verify()
            await file.seek(0)  # Reset file pointer
        except Exception as e:
            raise ValueError(f"Invalid image file: {e}")

        return True

    def get_file_url(self, file_path: str) -> str:
        """
        Get URL for accessing a file

        Args:
            file_path: Absolute file path

        Returns:
            Relative URL for file access
        """
        # Convert absolute path to relative URL
        rel_path = os.path.relpath(file_path, self.storage_path)
        return f"/files/{rel_path}"

    def get_absolute_path(self, relative_path: str) -> str:
        """
        Get absolute path from relative path

        Args:
            relative_path: Relative path

        Returns:
            Absolute file path
        """
        return os.path.join(self.storage_path, relative_path)

    async def delete_file(self, file_path: str) -> bool:
        """
        Delete a file

        Args:
            file_path: Path to file

        Returns:
            True if deleted, False if not found
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
        return False

    async def delete_task_files(self, task_id: str) -> bool:
        """
        Delete all files for a task

        Args:
            task_id: Task identifier

        Returns:
            True if deleted
        """
        task_dir = os.path.join(self.storage_path, task_id)
        try:
            if os.path.exists(task_dir):
                import shutil
                shutil.rmtree(task_dir)
                return True
        except Exception as e:
            print(f"Error deleting task files for {task_id}: {e}")
        return False

    async def cleanup_expired_files(self):
        """
        Clean up expired files based on retention period
        """
        retention_days = settings.file_retention_days
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)

        for item in os.listdir(self.storage_path):
            item_path = os.path.join(self.storage_path, item)
            if os.path.isdir(item_path):
                # Check directory modification time
                mtime = datetime.fromtimestamp(os.path.getmtime(item_path))
                if mtime < cutoff_date:
                    try:
                        import shutil
                        shutil.rmtree(item_path)
                        print(f"Cleaned up expired directory: {item}")
                    except Exception as e:
                        print(f"Error cleaning up {item_path}: {e}")


# Global file service instance
file_service = FileService()
