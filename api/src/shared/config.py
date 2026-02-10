# src/shared/config.py - Fixed version with validate_aws_config

from pydantic_settings import BaseSettings
from typing import Optional, List, Dict
import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    # Supabase Configuration - Will be loaded from .env file or environment variables
    SUPABASE_URL: str
    SUPABASE_ANON_KEY: str
    SUPABASE_SERVICE_ROLE_KEY: str
    DATABASE_URL: str
    
    # Optional settings with defaults
    SUPABASE_STORAGE_BUCKET: str = "videos"
    REDIS_URL: str = "redis://localhost:6379"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True
    
    # AWS SQS Configuration (optional)
    AWS_REGION: str = "eu-west-1"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    
    # SQS Queue Configuration with mapping
    SQS_VIDEO_UPLOADED_QUEUE: str = "videorag-video-uploaded"
    SQS_TRANSCRIPTION_COMPLETED_QUEUE: str = "videorag-transcription-completed"
    SQS_SCENE_DETECTION_COMPLETED_QUEUE: str = "videorag-scene-detection-completed"
    SQS_SEGMENTATION_COMPLETED_QUEUE: str = "videorag-segmentation-completed"
    SQS_EMBEDDINGS_GENERATED_QUEUE: str = "videorag-embeddings-generated"
    SQS_INDEXING_COMPLETED_QUEUE: str = "videorag-indexing-completed"
    SQS_PROCESSING_COMPLETED_QUEUE: str = "videorag-processing-completed"
    SQS_PROCESSING_FAILED_DLQ: str = "videorag-processing-failed-dlq"
    SQS_QUERY_RECEIVED_QUEUE: str = "videorag-query-received"
    SQS_HEALTH_CHECK_QUEUE: str = "videorag-health-check"
    
    # Processing timeouts
    TRANSCRIPTION_TIMEOUT: int = 1800
    SCENE_DETECTION_TIMEOUT: int = 900
    EMBEDDING_TIMEOUT: int = 600
    INDEXING_TIMEOUT: int = 300
    
    # Worker configuration
    TRANSCRIPTION_WORKER_CONCURRENCY: int = 3
    SCENE_DETECTION_WORKER_CONCURRENCY: int = 5
    EMBEDDING_WORKER_CONCURRENCY: int = 4
    INDEXING_WORKER_CONCURRENCY: int = 2
    
    # Optional AWS services
    S3_BUCKET: Optional[str] = None
    OPENSEARCH_HOST: Optional[str] = None
    EVENT_BUS_NAME: Optional[str] = None
    CLOUDWATCH_LOG_GROUP: str = "/aws/videorag"
    
    # Upload configuration
    DIRECT_SUPABASE_UPLOAD: bool = True
    
    # Computed properties
    @property
    def SQS_QUEUE_MAPPING(self) -> Dict[str, str]:
        """Dynamic queue mapping"""
        return {
            "video_uploaded": self.SQS_VIDEO_UPLOADED_QUEUE,
            "transcription_completed": self.SQS_TRANSCRIPTION_COMPLETED_QUEUE,
            "scene_detection_completed": self.SQS_SCENE_DETECTION_COMPLETED_QUEUE,
            "segmentation_completed": self.SQS_SEGMENTATION_COMPLETED_QUEUE,
            "embeddings_generated": self.SQS_EMBEDDINGS_GENERATED_QUEUE,
            "indexing_completed": self.SQS_INDEXING_COMPLETED_QUEUE,
            "processing_completed": self.SQS_PROCESSING_COMPLETED_QUEUE,
            "processing_failed": self.SQS_PROCESSING_FAILED_DLQ,
            "query_received": self.SQS_QUERY_RECEIVED_QUEUE,
            "health_check": self.SQS_HEALTH_CHECK_QUEUE
        }
    
    class Config:
        env_file = ".env"
        extra = "ignore"
        case_sensitive = True

# Create settings instance
try:
    settings = Settings()
    logger.info("✅ Configuration loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load configuration: {e}")
    print(f"\n🔧 Configuration Error: {e}")
    print("\nPlease check that you have:")
    print("1. Created a .env file with your Supabase credentials")
    print("2. Set the required environment variables:")
    print("   - SUPABASE_URL")
    print("   - SUPABASE_ANON_KEY") 
    print("   - SUPABASE_SERVICE_ROLE_KEY")
    print("   - DATABASE_URL")
    raise

# Supabase client configuration
SUPABASE_CONFIG = {
    "url": settings.SUPABASE_URL,
    "key": settings.SUPABASE_SERVICE_ROLE_KEY,
    "auto_refresh_token": True,
    "persist_session": False
}

def validate_supabase_config():
    """Validate Supabase configuration"""
    errors = []
    
    if not settings.SUPABASE_URL:
        errors.append("SUPABASE_URL is required")
    elif "your-project" in settings.SUPABASE_URL or not settings.SUPABASE_URL.startswith("https://"):
        errors.append("SUPABASE_URL appears to be invalid or still contains placeholder values")
    
    if not settings.SUPABASE_ANON_KEY:
        errors.append("SUPABASE_ANON_KEY is required")
    elif "your-anon-key" in settings.SUPABASE_ANON_KEY:
        errors.append("SUPABASE_ANON_KEY still contains placeholder values")
    
    if not settings.SUPABASE_SERVICE_ROLE_KEY:
        errors.append("SUPABASE_SERVICE_ROLE_KEY is required")
    elif "your-service-role-key" in settings.SUPABASE_SERVICE_ROLE_KEY:
        errors.append("SUPABASE_SERVICE_ROLE_KEY still contains placeholder values")
    
    if not settings.DATABASE_URL:
        errors.append("DATABASE_URL is required")
    elif "your-password" in settings.DATABASE_URL or "your-project-ref" in settings.DATABASE_URL:
        errors.append("DATABASE_URL still contains placeholder values")
    
    if errors:
        error_msg = "Supabase Configuration Errors:\n" + "\n".join(f"  - {error}" for error in errors)
        raise ValueError(error_msg)
    
    logger.info("✅ Supabase configuration validated successfully")

def validate_aws_config():
    """Validate AWS configuration for SQS functionality"""
    errors = []
    warnings = []
    
    # Check required AWS configuration
    if not settings.AWS_REGION:
        errors.append("AWS_REGION is required for SQS functionality")
    
    # Check AWS credentials (optional but recommended for production)
    if not settings.AWS_ACCESS_KEY_ID and not os.getenv('AWS_PROFILE'):
        warnings.append("AWS_ACCESS_KEY_ID not set. Make sure AWS credentials are configured via IAM roles, profiles, or environment variables")
    
    if not settings.AWS_SECRET_ACCESS_KEY and not os.getenv('AWS_PROFILE'):
        warnings.append("AWS_SECRET_ACCESS_KEY not set. Make sure AWS credentials are configured via IAM roles, profiles, or environment variables")
    
    # Check SQS queue configuration
    queue_names = [
        settings.SQS_VIDEO_UPLOADED_QUEUE,
        settings.SQS_TRANSCRIPTION_COMPLETED_QUEUE,
        settings.SQS_SCENE_DETECTION_COMPLETED_QUEUE,
        settings.SQS_PROCESSING_FAILED_DLQ
    ]
    
    for queue_name in queue_names:
        if not queue_name or queue_name.strip() == "":
            errors.append(f"SQS queue name cannot be empty")
        elif len(queue_name) > 80:
            errors.append(f"SQS queue name '{queue_name}' is too long (max 80 characters)")
        elif not queue_name.replace('-', '').replace('_', '').isalnum():
            errors.append(f"SQS queue name '{queue_name}' contains invalid characters")
    
    # Log warnings
    for warning in warnings:
        logger.warning(f"⚠️ AWS Config Warning: {warning}")
    
    # Raise errors if any
    if errors:
        error_msg = "AWS Configuration Errors:\n" + "\n".join(f"  - {error}" for error in errors)
        raise ValueError(error_msg)
    
    logger.info("✅ AWS configuration validated successfully")

def validate_all_config():
    """Validate all configuration settings"""
    validate_supabase_config()
    validate_aws_config()
    logger.info("✅ All configuration validated successfully")

# Validate on import (only Supabase for now, AWS is optional)
validate_supabase_config()

# Export
__all__ = [
    "settings", 
    "SUPABASE_CONFIG", 
    "validate_supabase_config", 
    "validate_aws_config",
    "validate_all_config"
]