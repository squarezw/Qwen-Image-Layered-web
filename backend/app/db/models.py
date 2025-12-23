"""
SQLAlchemy database models
"""
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, Enum, ForeignKey, Text, JSON, Index
from sqlalchemy.orm import relationship
import enum
from .database import Base


class TaskType(str, enum.Enum):
    """Task type enumeration"""
    DECOMPOSITION = "decomposition"
    EDITING = "editing"


class TaskStatus(str, enum.Enum):
    """Task status enumeration"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class BatchStatus(str, enum.Enum):
    """Batch status enumeration"""
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class FileType(str, enum.Enum):
    """File type enumeration"""
    INPUT = "input"
    LAYER = "layer"
    PPTX = "pptx"
    ZIP = "zip"


class Task(Base):
    """Task model for decomposition and editing operations"""
    __tablename__ = "tasks"

    id = Column(String(36), primary_key=True)  # UUID
    type = Column(Enum(TaskType), nullable=False, index=True)
    status = Column(Enum(TaskStatus), default=TaskStatus.QUEUED, index=True)

    # Input
    input_image_path = Column(String(512))
    input_image_hash = Column(String(64))  # For deduplication

    # Parameters (stored as JSON)
    parameters = Column(JSON)

    # Results (stored as JSON)
    result_data = Column(JSON)
    error_message = Column(Text, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)

    # User tracking (optional for multi-user)
    user_id = Column(String(36), nullable=True, index=True)

    # Batch relationship
    batch_id = Column(String(36), ForeignKey("batches.id"), nullable=True, index=True)
    batch = relationship("Batch", back_populates="tasks")

    # Files relationship
    files = relationship("StoredFile", back_populates="task", cascade="all, delete-orphan")


class Batch(Base):
    """Batch model for batch processing operations"""
    __tablename__ = "batches"

    id = Column(String(36), primary_key=True)  # UUID
    name = Column(String(256))
    status = Column(Enum(BatchStatus), default=BatchStatus.PROCESSING)

    total_count = Column(Integer)
    completed_count = Column(Integer, default=0)
    failed_count = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime, nullable=True)

    # Tasks relationship
    tasks = relationship("Task", back_populates="batch")


class StoredFile(Base):
    """Stored file model for tracking uploaded and generated files"""
    __tablename__ = "stored_files"

    id = Column(String(36), primary_key=True)  # UUID
    file_path = Column(String(512))
    file_type = Column(Enum(FileType))
    task_id = Column(String(36), ForeignKey("tasks.id"), index=True)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    expires_at = Column(DateTime, nullable=True, index=True)  # For cleanup

    # Task relationship
    task = relationship("Task", back_populates="files")


# Create indexes for common queries
Index('idx_tasks_created_at', Task.created_at)
Index('idx_tasks_status_type', Task.status, Task.type)
Index('idx_stored_files_expires_at', StoredFile.expires_at)
