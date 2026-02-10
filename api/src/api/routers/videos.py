

import os
import uuid 
import aiofiles
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
#from src.core.services.enhanced_video_processor import  EnhancedVideoProcessor
#from src.core.search.vector_search import  HierarchicalVectorSearch
from src.core.models.video import Video, VideoCreate
from src.core.database.repositories.video_repository import get_video_repository, SupabaseVideoRepository
from src.core.database.connection import get_storage_manager, SupabaseStorageManager
import uuid
import tempfile
import logging
from typing import Optional, List
import boto3
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer(auto_error=False)

# Ensure upload directory exists
UPLOAD_DIR = Path(tempfile.gettempdir()) / "videorag_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[str]:
    """Extract user ID from JWT token (implement based on your auth system)"""
    
    # TEMPORARY FIX: Bypass authentication for local testing
    logger.info("Using test user (authentication bypassed for local dev)")
    return "00000000-0000-0000-0000-000000000001"

async def process_video_background(
    video_id: str, 
    file_path: str,
    #video_processor: EnhancedVideoProcessor,
    #vector_search: HierarchicalVectorSearch,
    video_repo: SupabaseVideoRepository,
    storage_manager: SupabaseStorageManager,
    filename: str
):
    #Enhanced background task with Supabase Storage integration
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


# COMPLETE FIXED upload_video function
# Replace lines 7179-7281 in api/src/api/routers/videos.py

@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    video_repo: SupabaseVideoRepository = Depends(get_video_repository),
    current_user: Optional[str] = Depends(get_current_user)
):
    """Upload video and queue for ML processing"""
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"File type {file_ext} not supported"
        )
    
    try:
        logger.info(f"=== Starting upload: {file.filename} ===")
        
        # Generate video ID
        video_id = str(uuid.uuid4())
        logger.info(f"Generated video_id: {video_id}")
        
        # Read file content
        content = await file.read()
        file_size = len(content)
        logger.info(f"File size: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
        
        # Upload to S3
        logger.info("Uploading to S3...")
        s3 = boto3.client('s3', region_name=os.getenv('AWS_REGION', 'us-east-1'))
        s3_bucket = os.getenv('S3_BUCKET', 'videorag-uploads')
        s3_key = f"videos/{video_id}/{file.filename}"
        
        s3.put_object(
            Bucket=s3_bucket,
            Key=s3_key,
            Body=content
        )
        logger.info(f"S3 upload complete: s3://{s3_bucket}/{s3_key}")
        
        # Create VideoCreate object (correct fields)
        video_create = VideoCreate(
            filename=file.filename,
            title=file.filename.rsplit('.', 1)[0] if '.' in file.filename else file.filename,
            user_id=current_user,  # This is already set to "test-user-id" from get_current_user
            file_size=file_size
        )
        
        logger.info("Creating database record...")
        
        # Create database record
        video = await video_repo.create(video_create, video_id=video_id)
        
        logger.info(f"Database record created: {video.video_id}")
        
        # Update with S3 info
        logger.info("Updating database with S3 info...")
        video = await video_repo.update_storage_info(video_id, s3_key, None)
        
        # Send to SQS queue
        queue_url = os.getenv('SQS_QUEUE_URL')
        
        if queue_url:
            logger.info(f"Sending to SQS: {queue_url}")
            sqs = boto3.client('sqs', region_name=os.getenv('AWS_REGION', 'us-east-1'))
            
            message = {
                "video_id": video_id,
                "s3_key": s3_key,
                "s3_bucket": s3_bucket,
                "filename": file.filename,
                "file_size": file_size,
                "user_id": current_user
            }
            
            sqs.send_message(
                QueueUrl=queue_url,
                MessageBody=json.dumps(message)
            )
            
            logger.info("SQS message sent")
            
            # Update status to queued
            await video_repo.update_status(video_id, "queued_for_processing")
        else:
            logger.warning("SQS_QUEUE_URL not set - skipping queue")
        
        logger.info(f"=== Upload complete: {video_id} ===")
        
        return {
            "video_id": video_id,
            "filename": file.filename,
            "file_size": file_size,
            "status": "queued_for_processing" if queue_url else "uploaded",
            "s3_key": s3_key,
            "message": "Video uploaded and queued for ML processing"
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
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
''' 
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
'''


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