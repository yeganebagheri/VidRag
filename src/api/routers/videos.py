# src/api/routers/videos.py

import os
import aiofiles
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from src.core.services.video_processor import get_enhanced_video_processor, EnhancedVideoProcessor
from src.core.search.vector_search import get_enhanced_vector_search, HierarchicalVectorSearch
from src.core.models.video import Video, VideoCreate
from src.core.database.repositories.video_repository import get_video_repository, SupabaseVideoRepository
from src.core.database.connection import get_storage_manager, SupabaseStorageManager
import uuid
import tempfile
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer(auto_error=False)

# Ensure upload directory exists
UPLOAD_DIR = Path(tempfile.gettempdir()) / "videorag_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[str]:
    """Extract user ID from JWT token (implement based on your auth system)"""
    if not credentials:
        return None
    
    # TODO: Implement JWT validation with Supabase Auth
    # For now, return a mock user ID
    try:
        # You would decode the JWT token here
        # token = credentials.credentials
        # decoded = jwt.decode(token, supabase_jwt_secret, algorithms=["HS256"])
        # return decoded.get("sub")
        return "mock-user-id"  # Replace with actual JWT validation
    except Exception as e:
        logger.warning(f"Invalid token: {e}")
        return None

async def process_video_background(
    video_id: str, 
    file_path: str,
    video_processor: EnhancedVideoProcessor,
    vector_search: HierarchicalVectorSearch,
    video_repo: SupabaseVideoRepository,
    storage_manager: SupabaseStorageManager,
    filename: str
):
    """Enhanced background task with Supabase Storage integration"""
    try:
        logger.info(f"Starting background processing for video {video_id}")
        
        # Update status to processing
        await video_repo.update_status(video_id, "processing")
        
        # Upload to Supabase Storage
        logger.info(f"Uploading video {video_id} to Supabase Storage")
        upload_result = await storage_manager.upload_video(video_id, file_path, filename)
        
        if upload_result["success"]:
            # Update storage info in database
            await video_repo.update_storage_info(
                video_id, 
                upload_result["storage_path"], 
                upload_result["public_url"]
            )
            logger.info(f"Video {video_id} uploaded to storage: {upload_result['storage_path']}")
        else:
            logger.error(f"Failed to upload video {video_id}: {upload_result['error']}")
            await video_repo.update_status(video_id, "failed", upload_result['error'])
            return
        
        # Process video
        results = await video_processor.process_video(file_path, video_id)
        
        # Index embeddings for search
        if results.get("embeddings") or results.get("hierarchical_segments"):
            await vector_search.index_video(video_id, results)
        
        # Save results to repository
        await video_repo.save_processing_results(video_id, results)
        
        # Update status to completed
        await video_repo.update_status(video_id, "completed")
        
        logger.info(f"Video processing completed for {video_id}")
        
    except Exception as e:
        logger.error(f"Video processing failed for {video_id}: {e}")
        await video_repo.update_status(video_id, "failed", str(e))
    
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file: {e}")


@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    is_public: bool = False,
    video_repo: SupabaseVideoRepository = Depends(get_video_repository),
    storage_manager: SupabaseStorageManager = Depends(get_storage_manager),
    current_user: Optional[str] = Depends(get_current_user)
):
    """Enhanced video upload with SQS event-driven processing"""

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file type
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"File type {file_ext} not supported. Allowed: {allowed_extensions}"
        )
    
    # Check file size (e.g., max 1GB)
    max_file_size = 1024 * 1024 * 1024  # 1GB
    if hasattr(file, 'size') and file.size > max_file_size:
        raise HTTPException(
            status_code=400,
            detail=f"File size too large. Maximum size: {max_file_size / (1024**3):.1f}GB"
        )
    
    try:
        # Create video record
        video_id = str(uuid.uuid4())
        
        # Get file size
        content = await file.read()
        file_size = len(content)
        
        # Create video with enhanced data
        video_create = VideoCreate(
            id=video_id,
            filename=file.filename,
            file_size=file_size,
            is_public=is_public
        )
        
        video = await video_repo.create(video_create, user_id=current_user)
        
        # Save uploaded file temporarily or directly to Supabase Storage
        file_path = None
        if settings.DIRECT_SUPABASE_UPLOAD:
            # Upload directly to Supabase Storage
            upload_result = await storage_manager.upload_video(video_id, content, file.filename)
            if upload_result["success"]:
                await video_repo.update_storage_info(
                    video_id, 
                    upload_result["storage_path"], 
                    upload_result["public_url"]
                )
                file_path = upload_result["storage_path"]
        else:
            # Save temporarily for worker processing
            temp_path = UPLOAD_DIR / f"{video_id}_{file.filename}"
            async with aiofiles.open(temp_path, 'wb') as f:
                await f.write(content)
            file_path = str(temp_path)
        
        # ===== SQS EVENT-DRIVEN PROCESSING =====
        # Instead of background_tasks, publish to SQS
        sqs_publisher = await get_sqs_event_publisher()
        
        correlation_id = str(uuid.uuid4())  # Track this upload through the pipeline
        
        # Publish video uploaded event to SQS
        success = await sqs_publisher.publish_video_processing_event(
            video_id=video_id,
            stage="uploaded",
            progress=0.0,
            metadata={
                "filename": file.filename,
                "file_size": file_size,
                "file_path": file_path,
                "user_id": current_user,
                "is_public": is_public,
                "upload_timestamp": datetime.utcnow().isoformat()
            },
            correlation_id=correlation_id
        )
        
        if not success:
            logger.error(f"Failed to publish upload event for video {video_id}")
            # Could fallback to background task or return error
            raise HTTPException(status_code=500, detail="Failed to initiate processing")
        
        # Update video status to indicate processing started
        await video_repo.update_status(video_id, "queued_for_processing")
        
        return {
            "video_id": video_id,
            "filename": file.filename,
            "file_size": file_size,
            "status": "queued_for_processing",
            "is_public": is_public,
            "user_id": current_user,
            "correlation_id": correlation_id,
            "message": "Video uploaded successfully. Processing queued via SQS.",
            "estimated_processing_time": "5-15 minutes",
            "processing_pipeline": [
                "transcription",
                "scene_detection", 
                "segmentation",
                "embedding_generation",
                "indexing"
            ]
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/{video_id}")
async def get_video_status(
    video_id: str,
    video_repo: SupabaseVideoRepository = Depends(get_video_repository),
    current_user: Optional[str] = Depends(get_current_user)
):
    """Get video processing status with user authorization"""
    video = await video_repo.get_by_id(video_id, user_id=current_user)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found or access denied")
    
    return video

@router.get("/")
async def list_videos(
    limit: int = 50,
    offset: int = 0,
    is_public: Optional[bool] = None,
    video_repo: SupabaseVideoRepository = Depends(get_video_repository),
    current_user: Optional[str] = Depends(get_current_user)
):
    """List videos with filtering and pagination"""
    videos = await video_repo.list_videos(
        user_id=current_user,
        is_public=is_public,
        limit=limit,
        offset=offset
    )
    
    return {
        "videos": videos,
        "total": len(videos),
        "limit": limit,
        "offset": offset
    }

@router.get("/user/stats")
async def get_user_stats(
    video_repo: SupabaseVideoRepository = Depends(get_video_repository),
    current_user: Optional[str] = Depends(get_current_user)
):
    """Get user video statistics"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    stats = await video_repo.get_user_videos_stats(current_user)
    return stats

@router.delete("/{video_id}")
async def delete_video(
    video_id: str,
    video_repo: SupabaseVideoRepository = Depends(get_video_repository),
    storage_manager: SupabaseStorageManager = Depends(get_storage_manager),
    current_user: Optional[str] = Depends(get_current_user)
):
    """Delete video with proper authorization"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Get video info before deletion
    video = await video_repo.get_by_id(video_id, user_id=current_user)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found or access denied")
    
    # Delete from storage if exists
    if video.storage_path:
        try:
            await storage_manager.delete_video(video_id, video.filename)
        except Exception as e:
            logger.warning(f"Failed to delete from storage: {e}")
    
    # Delete from database
    success = await video_repo.delete_video(video_id, user_id=current_user)
    
    if success:
        return {"message": "Video deleted successfully", "video_id": video_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to delete video")

@router.post("/search")
async def search_videos(
    query: str,
    limit: int = 20,
    video_repo: SupabaseVideoRepository = Depends(get_video_repository),
    current_user: Optional[str] = Depends(get_current_user)
):
    """Search videos using Supabase full-text search"""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    videos = await video_repo.search_videos(query, user_id=current_user, limit=limit)
    
    return {
        "query": query,
        "videos": videos,
        "total": len(videos)
    }

@router.get("/{video_id}/download_url")
async def get_download_url(
    video_id: str,
    video_repo: SupabaseVideoRepository = Depends(get_video_repository),
    storage_manager: SupabaseStorageManager = Depends(get_storage_manager),
    current_user: Optional[str] = Depends(get_current_user)
):
    """Get signed download URL for video"""
    video = await video_repo.get_by_id(video_id, user_id=current_user)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found or access denied")
    
    if not video.storage_path:
        raise HTTPException(status_code=404, detail="Video not found in storage")
    
    signed_url = await storage_manager.get_video_url(video_id, video.filename)
    
    if signed_url:
        return {
            "video_id": video_id,
            "download_url": signed_url,
            "expires_in": 3600  # 1 hour
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to generate download URL")

@router.put("/{video_id}/visibility")
async def update_video_visibility(
    video_id: str,
    is_public: bool,
    video_repo: SupabaseVideoRepository = Depends(get_video_repository),
    current_user: Optional[str] = Depends(get_current_user)
):
    """Update video visibility (public/private)"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Check ownership
    video = await video_repo.get_by_id(video_id, user_id=current_user)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found or access denied")
    
    # Update visibility
    try:
        async with video_repo.get_db_session() as session:
            query = (
                update(VideoMetadata)
                .where(VideoMetadata.video_id == uuid.UUID(video_id))
                .values(is_public=is_public, updated_at=datetime.utcnow())
            )
            await session.execute(query)
            await session.commit()
        
        return {
            "video_id": video_id,
            "is_public": is_public,
            "message": f"Video visibility updated to {'public' if is_public else 'private'}"
        }
        
    except Exception as e:
        logger.error(f"Failed to update visibility: {e}")
        raise HTTPException(status_code=500, detail="Failed to update video visibility")