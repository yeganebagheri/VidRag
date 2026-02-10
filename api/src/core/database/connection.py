# src/core/database/connection.py

import asyncio
import logging
from typing import AsyncGenerator, Optional, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Boolean, JSON, Text, ForeignKey
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
import uuid

from src.shared.config import settings

logger = logging.getLogger(__name__)

# SQLAlchemy Base
Base = declarative_base()

# Processing Status Enum
class ProcessingStatus(str, Enum):
    uploaded = "uploaded"
    queued_for_processing = "queued_for_processing"
    processing = "processing"
    transcribing = "transcribing"
    analyzing_scenes = "analyzing_scenes"
    extracting_features = "extracting_features"
    building_index = "building_index"
    completed = "completed"
    failed = "failed"

# SQLAlchemy Models matching your Supabase tables
class VideoMetadata(Base):
    __tablename__ = "video_metadata"
    
    video_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=False)
    status = Column(String(50), nullable=False, default=ProcessingStatus.uploaded.value)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True, onupdate=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    
    # Video information
    duration = Column(Float, nullable=True)
    fps = Column(Float, nullable=True)
    resolution = Column(JSON, nullable=True)  # {"width": 1920, "height": 1080}
    
    # Processing results
    total_scenes = Column(Integer, nullable=True)
    total_segments = Column(Integer, nullable=True)
    processing_results = Column(JSON, nullable=True)
    knowledge_graph = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # User and access
    user_id = Column(UUID(as_uuid=True), nullable=True)
    is_public = Column(Boolean, nullable=False, default=False)
    
    # Storage information
    storage_path = Column(String(500), nullable=True)
    public_url = Column(String(500), nullable=True)
    
    # Relationships
    segments = relationship("VideoSegment", back_populates="video", cascade="all, delete-orphan")

class VideoSegment(Base):
    __tablename__ = "video_segments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey("video_metadata.video_id"), nullable=False)
    segment_id = Column(String(100), nullable=False)
    scene_id = Column(String(100), nullable=True)
    
    # Timing
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    duration = Column(Float, nullable=False)
    
    # Content
    text = Column(Text, nullable=True)
    ocr_texts = Column(JSON, nullable=True)  # List of OCR text results
    
    # Metadata
    segment_metadata = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    video = relationship("VideoMetadata", back_populates="segments")

# Database Manager Class
class DatabaseManager:
    def __init__(self):
        self.engine = None
        self.async_session_factory = None
        self.supabase_client: Optional[Client] = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize database connections"""
        try:
            logger.info("🔄 Initializing database connections...")
            
            # Initialize PostgreSQL with SQLAlchemy
            await self._initialize_sqlalchemy()
            
            # Initialize Supabase client
            await self._initialize_supabase()
            
            # Create tables if they don't exist
            #await self._create_tables()
            
            self.initialized = True
            logger.info("✅ Database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    async def _initialize_sqlalchemy(self):
        """Initialize SQLAlchemy async engine"""
        try:
            # Create async engine
            self.engine = create_async_engine(
                settings.DATABASE_URL,
                echo=settings.DEBUG,
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True,
                pool_recycle=3600,
            )
            
            # Create session factory
            self.async_session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            logger.info("✅ SQLAlchemy async engine initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize SQLAlchemy: {e}")
            raise
    
    async def _initialize_supabase(self):
        """Initialize Supabase client"""
        try:
            logger.info(f"🔄 Initializing Supabase with URL: {settings.SUPABASE_URL[:50]}...")
            logger.info(f"🔄 Using service role key: {settings.SUPABASE_SERVICE_ROLE_KEY[:50]}...")
            
            # Create Supabase client with proper options
            options = ClientOptions(
                postgrest_client_timeout=60,
                storage_client_timeout=60,
                auto_refresh_token=True,
                persist_session=False
            )
            
            logger.info("🔄 Creating Supabase client...")
            self.supabase_client = create_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_SERVICE_ROLE_KEY,
                options=options
            )
            
            logger.info("🔄 Testing Supabase connection...")
            # Test connection
            test_response = self.supabase_client.table('video_metadata').select('video_id').limit(1).execute()
            
            logger.info("✅ Supabase client initialized and connected")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase: {e}")
            logger.exception(e)  # ADD THIS - prints full traceback
            # Don't raise here - we can still work with just PostgreSQL
            logger.warning("⚠️ Continuing without Supabase client features")
    
    async def _create_tables(self):
        """Create tables if they don't exist"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("✅ Database tables created/verified")
            
        except Exception as e:
            logger.error(f"❌ Failed to create tables: {e}")
            raise
    
    async def get_session(self) -> AsyncSession:
        """Get async database session"""
        if not self.initialized:
            await self.initialize()
        
        return self.async_session_factory()
    
    def get_supabase_client(self) -> Optional[Client]:
        """Get Supabase client"""
        return self.supabase_client
    
    async def health_check(self) -> Dict[str, Any]:
        """Check database health"""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "sqlalchemy_connected": False,
            "supabase_connected": False,
            "pool_stats": {}
        }
        
        try:
            # Check SQLAlchemy connection
            if self.engine:
                async with self.engine.begin() as conn:
                    result = await conn.execute("SELECT 1")
                    if result.scalar() == 1:
                        health_status["sqlalchemy_connected"] = True
                        
                        # Get pool stats
                        pool = self.engine.pool
                        health_status["pool_stats"] = {
                            "pool_size": pool.size(),
                            "checked_in": pool.checkedin(),
                            "checked_out": pool.checkedout(),
                            "overflow": pool.overflow(),
                            "invalid": pool.invalid()
                        }
            
            # Check Supabase connection
            if self.supabase_client:
                test_response = self.supabase_client.table('video_metadata').select('video_id').limit(1).execute()
                health_status["supabase_connected"] = True
                health_status["supabase_status"] = "connected"
            
            # Overall status
            if not health_status["sqlalchemy_connected"]:
                health_status["status"] = "degraded"
                
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
            logger.error(f"Database health check failed: {e}")
        
        return health_status
    
    async def close(self):
        """Close database connections"""
        try:
            if self.engine:
                await self.engine.dispose()
                logger.info("✅ Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")

# Global database manager instance
db_manager = DatabaseManager()

# Context manager for database sessions
@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session with automatic cleanup"""
    if not db_manager.initialized:
        await db_manager.initialize()
    
    session = await db_manager.get_session()
    try:
        yield session
    except Exception as e:
        await session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        await session.close()

# Storage Manager for Supabase Storage
class SupabaseStorageManager:
    def __init__(self):
        self.supabase_client: Optional[Client] = None
        self.bucket_name = settings.SUPABASE_STORAGE_BUCKET
        
    async def initialize(self):
        """Initialize storage manager"""
        self.supabase_client = await get_supabase_client()
        if not self.supabase_client:
            raise Exception("Supabase client not available")
    
    async def upload_video(self, video_id: str, file_path_or_data, filename: str) -> Dict[str, Any]:
        """Upload video to Supabase Storage"""
        try:
            if not self.supabase_client:
                await self.initialize()
            
            storage_path = f"videos/{video_id}/{filename}"
            
            # Handle both file path and binary data
            if isinstance(file_path_or_data, str):
                # File path
                with open(file_path_or_data, 'rb') as file:
                    file_data = file.read()
            else:
                # Binary data
                file_data = file_path_or_data
            
            # Upload to storage
            response = self.supabase_client.storage.from_(self.bucket_name).upload(
                path=storage_path,
                file=file_data,
                file_options={
                    "content-type": "video/mp4",  # Adjust based on file type
                    "upsert": True
                }
            )
            
            if response.status_code == 200:
                # Get public URL
                public_url = self.supabase_client.storage.from_(self.bucket_name).get_public_url(storage_path)
                
                return {
                    "success": True,
                    "storage_path": storage_path,
                    "public_url": public_url,
                    "bucket": self.bucket_name
                }
            else:
                return {
                    "success": False,
                    "error": f"Upload failed with status {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Failed to upload video to storage: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_video_url(self, video_id: str, filename: str, expires_in: int = 3600) -> Optional[str]:
        """Get signed URL for video download"""
        try:
            if not self.supabase_client:
                await self.initialize()
            
            storage_path = f"videos/{video_id}/{filename}"
            
            # Create signed URL
            signed_url = self.supabase_client.storage.from_(self.bucket_name).create_signed_url(
                path=storage_path,
                expires_in=expires_in
            )
            
            return signed_url.get('signedURL')
            
        except Exception as e:
            logger.error(f"Failed to get video URL: {e}")
            return None
    
    async def delete_video(self, video_id: str, filename: str) -> bool:
        """Delete video from storage"""
        try:
            if not self.supabase_client:
                await self.initialize()
            
            storage_path = f"videos/{video_id}/{filename}"
            
            response = self.supabase_client.storage.from_(self.bucket_name).remove([storage_path])
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Failed to delete video from storage: {e}")
            return False

# Dependency injection functions
async def get_supabase_client() -> Optional[Client]:
    """Get Supabase client instance"""
    if not db_manager.initialized:
        await db_manager.initialize()
    return db_manager.get_supabase_client()

async def get_storage_manager() -> SupabaseStorageManager:
    """Get storage manager instance"""
    storage_manager = SupabaseStorageManager()
    await storage_manager.initialize()
    return storage_manager

# Initialization functions
async def initialize_database():
    """Initialize database connections"""
    await db_manager.initialize()

async def cleanup_database():
    """Cleanup database connections"""
    await db_manager.close()

# Health check function
async def check_database_health() -> Dict[str, Any]:
    """Check database health"""
    return await db_manager.health_check()

# Export commonly used items
__all__ = [
    "Base",
    "VideoMetadata", 
    "VideoSegment",
    "ProcessingStatus",
    "get_db_session",
    "get_supabase_client",
    "get_storage_manager",
    "initialize_database",
    "cleanup_database",
    "check_database_health",
    "db_manager"
]