"""
Pydantic schemas for API request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class TaskTypeEnum(str, Enum):
    """Task type enumeration"""
    DECOMPOSITION = "decomposition"
    EDITING = "editing"


class TaskStatusEnum(str, Enum):
    """Task status enumeration"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# Decomposition Schemas
class DecompositionParams(BaseModel):
    """Parameters for image decomposition"""
    prompt: Optional[str] = None
    negative_prompt: str = " "
    seed: int = 777
    randomize_seed: bool = False
    true_guidance_scale: float = Field(4.0, ge=1.0, le=10.0)
    num_inference_steps: int = Field(50, ge=1, le=50)
    layers: int = Field(4, ge=2, le=10)
    cfg_normalize: bool = True
    use_en_prompt: bool = True


class DecompositionSubmitResponse(BaseModel):
    """Response for decomposition submission"""
    task_id: str
    status: str
    position_in_queue: int
    message: str


class LayerInfo(BaseModel):
    """Information about a generated layer"""
    url: str
    index: int
    filename: str


class DecompositionResult(BaseModel):
    """Result of decomposition task"""
    task_id: str
    layers: List[LayerInfo]
    pptx_url: Optional[str] = None
    zip_url: Optional[str] = None
    metadata: dict


# Editing Schemas
class EditingParams(BaseModel):
    """Parameters for layer editing"""
    prompt: str
    seed: int = 42
    randomize_seed: bool = False
    true_guidance_scale: float = Field(4.0, ge=1.0, le=10.0)
    num_inference_steps: int = Field(50, ge=1, le=50)


class EditingSubmitResponse(BaseModel):
    """Response for editing submission"""
    task_id: str
    status: str
    position_in_queue: int
    message: str


class EditingResult(BaseModel):
    """Result of editing task"""
    task_id: str
    image_url: str
    seed_used: int
    metadata: dict


# Task Status Schemas
class TaskStatusResponse(BaseModel):
    """Response for task status query"""
    task_id: str
    status: TaskStatusEnum
    progress: float
    message: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


# Batch Processing Schemas
class BatchSubmitResponse(BaseModel):
    """Response for batch submission"""
    batch_id: str
    task_ids: List[str]
    total_count: int
    message: str


class BatchTaskStatus(BaseModel):
    """Status of a task in a batch"""
    task_id: str
    status: TaskStatusEnum
    progress: float


class BatchStatusResponse(BaseModel):
    """Response for batch status query"""
    batch_id: str
    completed: int
    failed: int
    processing: int
    queued: int
    total: int
    tasks: List[BatchTaskStatus]


# History Schemas
class HistoryItemSummary(BaseModel):
    """Summary of a history item"""
    id: str
    type: TaskTypeEnum
    status: TaskStatusEnum
    created_at: datetime
    input_image_url: str
    result_preview_url: Optional[str] = None
    parameters: dict


class HistoryListResponse(BaseModel):
    """Response for history list"""
    items: List[HistoryItemSummary]
    total: int
    page: int
    pages: int


class HistoryDetailResponse(BaseModel):
    """Detailed history item"""
    id: str
    type: TaskTypeEnum
    status: TaskStatusEnum
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    input_image_url: str
    parameters: dict
    result_data: Optional[dict] = None
    error: Optional[str] = None


# Health Check Schema
class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    gpu_available: bool
    gpu_info: Optional[dict] = None
    models_loaded: bool
    queue_size: int
    timestamp: datetime
