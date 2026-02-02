"""
Data models for HIVD Classifier
Maltezos, V. (2026). A framework for efficient image categorisation in social research
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime


class ClassDefinition(BaseModel):
    """Definition of an image class for classification"""
    id: Optional[int] = None
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=1000)
    reference_images: List[str] = Field(default_factory=list, max_items=5)
    parent_id: Optional[int] = Field(None, description="ID of the parent macro-category")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    @validator('reference_images')
    def validate_reference_images(cls, v):
        if len(v) > 5:
            raise ValueError('Maximum 5 reference images allowed per class')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "ProtestSelfies",
                "description": "Selfies taken during protests showing protesters",
                "reference_images": [
                    "image_dataset/protest1.jpg",
                    "image_dataset/protest2.jpg"
                ]
            }
        }


class ClassificationStatus(str, Enum):
    """Status of a classification job"""
    PENDING = "pending"
    RUNNING = "running"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SubcategorizationStrategy(str, Enum):
    """
    POST (Hierarchical): First classify to macro categories, then sub-classify within each macro.
                        Images only move to subcategory folder if confidence >= subcategory_threshold.
                        Otherwise, images stay in the parent macro folder.
    
    PRE (Flat):         All subcategories compete together as equals in a single classification pass.
                        Results are then organized into the parent/child folder structure.
    """
    POST = "post"  # Hierarchical (Macro ? then Sub within each)
    PRE = "pre"    # Flat (All subcategories compete together)


class ClassificationConfig(BaseModel):
    """Configuration for a classification job"""
    job_name: str = Field(..., min_length=1, max_length=100)
    class_ids: List[int] = Field(..., min_items=1)
    
    # Strategy
    strategy: SubcategorizationStrategy = Field(default=SubcategorizationStrategy.POST)
    
    # Model selection - see classifier_engine.CLIP_MODEL_REGISTRY for all options
    # Recommended: "ViT-L-14" with "openai" for balanced performance
    # For best accuracy: "EVA02-L-14" with "merged2b_s4b_b131k"
    # For speed on CPU: "ViT-B-32" with "openai"
    clip_model: str = Field(
        default="ViT-L-14",
        description="CLIP model architecture. Options include: RN101, ViT-B-16, ViT-L-14, ViT-H-14, EVA02-L-14, ViT-SO400M-14-SigLIP-384"
    )
    clip_pretrained: str = Field(
        default="openai",
        description="Pretrained weights. Must be compatible with the chosen model."
    )
    
    # Augmentation parameters
    num_augmentations: int = Field(default=20, ge=1, le=500)
    random_augmentations: int = Field(default=10, ge=1, le=500)
    std_threshold: float = Field(default=3.0, ge=0.5, le=5.0)
    
    # Probability thresholds for macro classification
    high_prob_threshold: float = Field(default=0.6, ge=0.0, le=1.0,
        description="Threshold for high confidence macro classification")
    low_prob_threshold: float = Field(default=0.4, ge=0.0, le=1.0,
        description="Threshold below which classification is considered low confidence")
    
    # Subcategory threshold (for POST/Hierarchical strategy)
    # Images must meet this threshold to be moved from parent to subcategory folder
    subcategory_threshold: float = Field(default=0.8, ge=0.0, le=1.0,
        description="Minimum confidence required to assign image to subcategory (POST strategy). Default 80%.")
    
    # Safety net
    apply_safety_net: bool = Field(default=True)
    
    @validator('low_prob_threshold')
    def validate_thresholds(cls, v, values):
        if 'high_prob_threshold' in values and v >= values['high_prob_threshold']:
            raise ValueError('low_prob_threshold must be less than high_prob_threshold')
        return v


class ClassificationJob(BaseModel):
    """Metadata for a classification job"""
    id: Optional[int] = None
    job_name: str
    status: ClassificationStatus = ClassificationStatus.PENDING
    config: ClassificationConfig
    
    # Progress tracking
    total_images: int = 0
    processed_images: int = 0
    progress_percentage: float = 0.0
    
    # Results
    high_confidence_count: int = 0
    low_confidence_count: int = 0
    
    # Timing
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Error handling
    error_message: Optional[str] = None
    
    class Config:
        use_enum_values = True


class ClassificationResult(BaseModel):
    """Result of classifying a single image"""
    job_id: int
    image_path: str
    predicted_class_id: int
    predicted_class_name: str
    predicted_subclass_id: Optional[int] = None
    predicted_subclass_name: Optional[str] = None
    probability: float
    is_high_confidence: bool
    
    # Safety net info
    safety_net_applied: bool = False
    safety_net_action: Optional[str] = None
    detected_objects: Optional[Dict[str, int]] = None


class ImageInfo(BaseModel):
    """Information about an image in the dataset"""
    path: str
    filename: str
    size_bytes: int
    width: Optional[int] = None
    height: Optional[int] = None


class ProgressUpdate(BaseModel):
    """Real-time progress update for WebSocket"""
    job_id: int
    status: ClassificationStatus
    processed_images: int
    total_images: int
    progress_percentage: float
    current_image: Optional[str] = None
    message: Optional[str] = None
    
    class Config:
        use_enum_values = True


class SystemStatus(BaseModel):
    """Overall system status"""
    active_jobs: int
    total_classes: int
    total_images_dataset: int
    total_images_inference: int
    gpu_available: bool
    clip_model_loaded: bool
    yolo_model_loaded: bool
