"""
Model Manager - Singleton pattern for loading and managing AI models
"""
import threading
import torch
from typing import Optional
from diffusers import QwenImageLayeredPipeline, QwenImageEditPlusPipeline
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from ..config import settings


class ModelManager:
    """
    Singleton class for managing AI models
    Loads models once at startup and keeps them in GPU memory
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.decomposition_pipeline: Optional[QwenImageLayeredPipeline] = None
            self.editing_pipeline: Optional[QwenImageEditPlusPipeline] = None
            self.rmbg_model: Optional[AutoModelForImageSegmentation] = None
            self.rmbg_transforms: Optional[transforms.Compose] = None

            # Device and dtype configuration
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

            self.initialized = False
            self.loading = False
            self._initialization_error: Optional[Exception] = None

    async def initialize(self):
        """
        Initialize and load all models
        This should be called once during application startup
        """
        if self.initialized:
            return

        if self.loading:
            # Wait for initialization to complete
            while self.loading and not self.initialized:
                await asyncio.sleep(0.1)
            return

        self.loading = True

        try:
            print("Loading Qwen Image Layered models...")

            # Load decomposition model
            print(f"Loading decomposition model: {settings.decomposition_model}")
            self.decomposition_pipeline = QwenImageLayeredPipeline.from_pretrained(
                settings.decomposition_model,
                cache_dir=settings.model_cache_dir if settings.model_cache_dir != "/models" else None
            )
            self.decomposition_pipeline = self.decomposition_pipeline.to(self.device, self.dtype)
            self.decomposition_pipeline.set_progress_bar_config(disable=None)
            print("Decomposition model loaded successfully")

            # Load editing model
            print(f"Loading editing model: {settings.editing_model}")
            self.editing_pipeline = QwenImageEditPlusPipeline.from_pretrained(
                settings.editing_model,
                torch_dtype=self.dtype,
                cache_dir=settings.model_cache_dir if settings.model_cache_dir != "/models" else None
            )
            self.editing_pipeline = self.editing_pipeline.to(self.device)
            print("Editing model loaded successfully")

            # Load RMBG model for background removal
            print(f"Loading RMBG model: {settings.rmbg_model}")
            self.rmbg_model = AutoModelForImageSegmentation.from_pretrained(
                settings.rmbg_model,
                trust_remote_code=True,
                cache_dir=settings.model_cache_dir if settings.model_cache_dir != "/models" else None
            )
            self.rmbg_model = self.rmbg_model.eval().to(self.device)
            print("RMBG model loaded successfully")

            # Initialize RMBG transforms
            self.rmbg_transforms = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            self.initialized = True
            print("All models loaded successfully!")

        except Exception as e:
            self._initialization_error = e
            print(f"Error loading models: {e}")
            raise
        finally:
            self.loading = False

    def get_gpu_info(self) -> dict:
        """Get GPU information"""
        if not torch.cuda.is_available():
            return {
                "available": False,
                "device": "cpu",
                "message": "CUDA not available"
            }

        return {
            "available": True,
            "device": torch.cuda.get_device_name(0),
            "memory_allocated": torch.cuda.memory_allocated(0) / 1024**3,  # GB
            "memory_reserved": torch.cuda.memory_reserved(0) / 1024**3,  # GB
            "memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        }

    @property
    def is_ready(self) -> bool:
        """Check if models are loaded and ready"""
        return self.initialized and not self.loading

    @property
    def initialization_error(self) -> Optional[Exception]:
        """Get initialization error if any"""
        return self._initialization_error


# Import asyncio for async sleep
import asyncio
