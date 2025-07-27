"""Base models and common types for all services."""

from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel as PydanticBaseModel, Field


class BaseModel(PydanticBaseModel):
    """Base model with common configuration."""
    
    class Config:
        # Enable JSON serialization of enums
        use_enum_values = True
        # Allow extra fields for flexibility
        extra = "forbid"
        # Validate assignment
        validate_assignment = True


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class QueryIntent(str, Enum):
    """Query intent classification."""
    SPECIFIC_ITEM = "specific_item"
    STYLE_SEARCH = "style_search"
    OCCASION_BASED = "occasion_based"
    SEASONAL = "seasonal"
    GENERAL = "general"
    UNCLEAR = "unclear"


class Season(str, Enum):
    """Season classification."""
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"
    WINTER = "winter"
    ALL_SEASON = "all_season"


class Occasion(str, Enum):
    """Occasion classification."""
    WORK = "work"
    CASUAL = "casual"
    FORMAL = "formal"
    PARTY = "party"
    ATHLETIC = "athletic"
    WEDDING = "wedding"
    DATE = "date"
    TRAVEL = "travel"
    OTHER = "other"


class ServiceHealth(str, Enum):
    """Service health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    service: Optional[str] = Field(default=None, description="Service that generated the error")
    timestamp: Optional[str] = Field(default=None, description="Error timestamp")


class HealthResponse(BaseModel):
    """Health check response."""
    status: ServiceHealth = Field(default=ServiceHealth.HEALTHY, description="Service status")
    service_name: str = Field(..., description="Name of the service")
    version: str = Field(..., description="Service version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    timestamp: str = Field(..., description="Health check timestamp")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional health details")


class StatusResponse(BaseModel):
    """General status response."""
    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Status message")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Additional status data") 