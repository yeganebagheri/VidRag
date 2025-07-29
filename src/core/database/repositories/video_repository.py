# src/core/database/repositories/video_repository.py

from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_, func, text
from supabase import Client
from src.core.models.video import Video, VideoCreate, ProcessingStatus
from src.core.database.connection import VideoMetadata, VideoSegment, get_db_session, get_supabase_client
from src.core.cache.cache_manager import get_cache_manager, cache_result
from datetime import datetime, timedelta
import uuid
import json
import pickle
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SupabaseVideoRepository:
    """Enhanced video repository with Supabase integration"""
    
    def __init__(self, session: AsyncSession = None):
        self.session = session
        self.cache_manager = None
        self.supabase_client: Optional[Client] = None
        self._cache_ttl = {
            'video': 3600,      # 1 hour
            'segments': 7200,   # 2 hours
            'analytics': 1800   # 30 minutes
        }
    
    async def _get_cache_manager(self):
        """Get cache manager instance"""
        if self.cache_manager is None:
            self.cache_manager = await get_cache_manager()
        return self.cache_manager
    
    async def _get_supabase_client(self) -> Client:
        """Get Supabase client"""
        if self.supabase_client is None:
            self.supabase_client = await get_supabase_client()
        return self.supabase_client
    
    async def create(self, video_data: VideoCreate, user_id: Optional[str] = None) -> Video:
        """Create new video record with Supabase features"""
        async with get_db_session() as session:
            try:
                video_uuid = uuid.UUID(video_data.id) if video_data.id else uuid.uuid4()
                
                # Create video metadata record
                video_metadata = VideoMetadata(
                    video_id=video_uuid,
                    filename=video_data.filename,
                    file_size=video_data.file_size,
                    status=ProcessingStatus.uploaded.value,
                    user_id=uuid.UUID(user_id) if user_id else None,
                    is_public=getattr(video_data, 'is_public', False)
                )
                
                session.add(video_metadata)
                await session.commit()
                await session.refresh(video_metadata)
                
                # Also create in Supabase for real-time features
                supabase = await self._get_supabase_client()
                supabase_data = {
                    "video_id": str(video_metadata.video_id),
                    "filename": video_metadata.filename,
                    "file_size": video_metadata.file_size,
                    "status": video_metadata.status,
                    "user_id": str(video_metadata.user_id) if video_metadata.user_id else None,
                    "is_public": video_metadata.is_public,
                    "created_at": video_metadata.created_at.isoformat()
                }
                
                try:
                    supabase.table('video_metadata').insert(supabase_data).execute()
                except Exception as e:
                    logger.warning(f"Failed to sync to Supabase table: {e}")
                
                # Convert to Video model
                video = Video(
                    id=str(video_metadata.video_id),
                    filename=video_metadata.filename,
                    file_size=video_metadata.file_size,
                    status=ProcessingStatus.uploaded,
                    created_at=video_metadata.created_at,
                    user_id=str(video_metadata.user_id) if video_metadata.user_id else None
                )
                
                # Cache the video
                cache_manager = await self._get_cache_manager()
                await cache_manager.set(
                    'video_metadata', 
                    str(video.id), 
                    video.dict(), 
                    ttl=self._cache_ttl['video']
                )
                
                logger.info(f"✅ Created video record: {video.id}")
                return video
                
            except Exception as e:
                await session.rollback()
                logger.error(f"❌ Failed to create video: {e}")
                raise
    
    async def get_by_id(self, video_id: str, user_id: Optional[str] = None) -> Optional[Video]:
        """Get video by ID with Supabase RLS consideration"""
        # Try cache first
        cache_manager = await self._get_cache_manager()
        cached_video = await cache_manager.get('video_metadata', video_id)
        
        if cached_video:
            return Video(**cached_video)
        
        # Query database with RLS consideration
        async with get_db_session() as session:
            query = select(VideoMetadata).where(VideoMetadata.video_id == uuid.UUID(video_id))
            
            # Add user filter if RLS is needed
            if user_id:
                query = query.where(
                    or_(
                        VideoMetadata.user_id == uuid.UUID(user_id),
                        VideoMetadata.is_public == True
                    )
                )
            
            result = await session.execute(query)
            video_metadata = result.scalar_one_or_none()
            
            if not video_metadata:
                return None
            
            video = Video(
                id=str(video_metadata.video_id),
                filename=video_metadata.filename,
                file_size=video_metadata.file_size,
                status=ProcessingStatus(video_metadata.status),
                created_at=video_metadata.created_at,
                updated_at=video_metadata.updated_at,
                duration=video_metadata.duration,
                fps=video_metadata.fps,
                resolution=video_metadata.resolution,
                total_scenes=video_metadata.total_scenes,
                total_segments=video_metadata.total_segments,
                processing_results=video_metadata.processing_results,
                error_message=video_metadata.error_message,
                user_id=str(video_metadata.user_id) if video_metadata.user_id else None,
                storage_path=video_metadata.storage_path,
                public_url=video_metadata.public_url
            )
            
            # Cache for future use
            await cache_manager.set(
                'video_metadata', 
                video_id, 
                video.dict(), 
                ttl=self._cache_ttl['video']
            )
            
            return video
    
    async def list_videos(self, user_id: Optional[str] = None, 
                         is_public: Optional[bool] = None,
                         limit: int = 50, offset: int = 0) -> List[Video]:
        """List videos with Supabase filtering"""
        async with get_db_session() as session:
            query = select(VideoMetadata)
            
            # Apply filters
            if user_id:
                if is_public is not None:
                    if is_public:
                        query = query.where(VideoMetadata.is_public == True)
                    else:
                        query = query.where(VideoMetadata.user_id == uuid.UUID(user_id))
                else:
                    # Show user's videos and public videos
                    query = query.where(
                        or_(
                            VideoMetadata.user_id == uuid.UUID(user_id),
                            VideoMetadata.is_public == True
                        )
                    )
            elif is_public is not None:
                query = query.where(VideoMetadata.is_public == is_public)
            
            # Add pagination
            query = query.offset(offset).limit(limit).order_by(VideoMetadata.created_at.desc())
            
            result = await session.execute(query)
            video_metadata_list = result.scalars().all()
            
            videos = []
            for vm in video_metadata_list:
                video = Video(
                    id=str(vm.video_id),
                    filename=vm.filename,
                    file_size=vm.file_size,
                    status=ProcessingStatus(vm.status),
                    created_at=vm.created_at,
                    updated_at=vm.updated_at,
                    user_id=str(vm.user_id) if vm.user_id else None,
                    storage_path=vm.storage_path,
                    public_url=vm.public_url
                )
                videos.append(video)
            
            return videos
    
    async def update_status(self, video_id: str, status: str, error_message: str = None):
        """Update video processing status in both PostgreSQL and Supabase"""
        async with get_db_session() as session:
            try:
                # Update PostgreSQL
                query = (
                    update(VideoMetadata)
                    .where(VideoMetadata.video_id == uuid.UUID(video_id))
                    .values(
                        status=status,
                        error_message=error_message,
                        updated_at=datetime.utcnow()
                    )
                )
                
                await session.execute(query)
                await session.commit()
                
                # Update Supabase for real-time updates
                try:
                    supabase = await self._get_supabase_client()
                    update_data = {
                        "status": status,
                        "updated_at": datetime.utcnow().isoformat()
                    }
                    if error_message:
                        update_data["error_message"] = error_message
                    
                    supabase.table('video_metadata').update(update_data).eq(
                        'video_id', video_id
                    ).execute()
                except Exception as e:
                    logger.warning(f"Failed to update Supabase: {e}")
                
                # Invalidate cache
                cache_manager = await self._get_cache_manager()
                await cache_manager.delete('video_metadata', video_id)
                
                logger.info(f"✅ Updated video {video_id} status to: {status}")
                
            except Exception as e:
                await session.rollback()
                logger.error(f"❌ Failed to update video status: {e}")
                raise
    
    async def save_processing_results(self, video_id: str, results: Dict[str, Any]):
        """Save comprehensive processing results with Supabase storage info"""
        async with get_db_session() as session:
            try:
                # Update video metadata
                video_update = {
                    'processing_results': results,
                    'duration': results.get('video_info', {}).get('duration'),
                    'fps': results.get('video_info', {}).get('fps'),
                    'resolution': {
                        'width': results.get('video_info', {}).get('width'),
                        'height': results.get('video_info', {}).get('height')
                    },
                    'total_scenes': len(results.get('scene_boundaries', [])),
                    'total_segments': len(results.get('hierarchical_segments', [])),
                    'knowledge_graph': results.get('knowledge_graph', {}),
                    'processed_at': datetime.utcnow(),
                    'updated_at': datetime.utcnow()
                }
                
                query = (
                    update(VideoMetadata)
                    .where(VideoMetadata.video_id == uuid.UUID(video_id))
                    .values(**video_update)
                )
                
                await session.execute(query)
                
                # Save hierarchical segments
                await self._save_video_segments(session, video_id, results)
                
                await session.commit()
                
                # Update Supabase
                try:
                    supabase = await self._get_supabase_client()
                    supabase_update = {
                        "total_scenes": video_update['total_scenes'],
                        "total_segments": video_update['total_segments'],
                        "processed_at": video_update['processed_at'].isoformat(),
                        "updated_at": video_update['updated_at'].isoformat()
                    }
                    supabase.table('video_metadata').update(supabase_update).eq(
                        'video_id', video_id
                    ).execute()
                except Exception as e:
                    logger.warning(f"Failed to update Supabase: {e}")
                
                # Invalidate relevant caches
                cache_manager = await self._get_cache_manager()
                await cache_manager.delete('video_metadata', video_id)
                await cache_manager.delete('video_segments', video_id)
                
                logger.info(f"✅ Saved processing results for video: {video_id}")
                
            except Exception as e:
                await session.rollback()
                logger.error(f"❌ Failed to save processing results: {e}")
                raise
    
    async def _save_video_segments(self, session: AsyncSession, video_id: str, results: Dict[str, Any]):
        """Save video segments to database"""
        try:
            # Delete existing segments
            delete_query = delete(VideoSegment).where(VideoSegment.video_id == uuid.UUID(video_id))
            await session.execute(delete_query)
            
            # Save hierarchical segments
            hierarchical_segments = results.get('hierarchical_segments', [])
            for segment_data in hierarchical_segments:
                segment = VideoSegment(
                    video_id=uuid.UUID(video_id),
                    segment_id=segment_data.get('segment_id', str(uuid.uuid4())),
                    scene_id=segment_data.get('scene_id'),
                    start_time=segment_data.get('start'),
                    end_time=segment_data.get('end'),
                    duration=segment_data.get('duration'),
                    text=segment_data.get('text'),
                    ocr_texts=segment_data.get('ocr_texts'),
                    segment_metadata=segment_data.get('metadata', {})
                )
                session.add(segment)
                
        except Exception as e:
            logger.error(f"❌ Failed to save video segments: {e}")
            raise
    
    async def get_user_videos_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user video statistics using Supabase"""
        try:
            supabase = await self._get_supabase_client()
            
            # Get video counts by status
            result = supabase.table('video_metadata').select(
                'status', 
                count='exact'
            ).eq('user_id', user_id).execute()
            
            stats = {
                'total_videos': 0,
                'by_status': {},
                'total_duration': 0.0,
                'total_storage_used': 0
            }
            
            for row in result.data:
                stats['by_status'][row['status']] = row.get('count', 0)
                stats['total_videos'] += row.get('count', 0)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get user stats: {e}")
            return {}

    async def update_storage_info(self, video_id: str, storage_path: str, public_url: str):
        """Update video storage information after upload to Supabase Storage"""
        async with get_db_session() as session:
            try:
                query = (
                    update(VideoMetadata)
                    .where(VideoMetadata.video_id == uuid.UUID(video_id))
                    .values(
                        storage_path=storage_path,
                        public_url=public_url,
                        updated_at=datetime.utcnow()
                    )
                )
                
                await session.execute(query)
                await session.commit()
                
                # Update Supabase
                try:
                    supabase = await self._get_supabase_client()
                    supabase.table('video_metadata').update({
                        'storage_path': storage_path,
                        'public_url': public_url,
                        'updated_at': datetime.utcnow().isoformat()
                    }).eq('video_id', video_id).execute()
                except Exception as e:
                    logger.warning(f"Failed to update Supabase storage info: {e}")
                
                # Invalidate cache
                cache_manager = await self._get_cache_manager()
                await cache_manager.delete('video_metadata', video_id)
                
                logger.info(f"✅ Updated storage info for video {video_id}")
                
            except Exception as e:
                await session.rollback()
                logger.error(f"❌ Failed to update storage info: {e}")
                raise

    async def delete_video(self, video_id: str, user_id: Optional[str] = None) -> bool:
        """Delete video with proper authorization"""
        async with get_db_session() as session:
            try:
                # Check ownership if user_id provided
                if user_id:
                    ownership_query = select(VideoMetadata).where(
                        and_(
                            VideoMetadata.video_id == uuid.UUID(video_id),
                            VideoMetadata.user_id == uuid.UUID(user_id)
                        )
                    )
                    result = await session.execute(ownership_query)
                    if not result.scalar_one_or_none():
                        logger.warning(f"User {user_id} attempted to delete video {video_id} without permission")
                        return False
                
                # Delete segments first
                await session.execute(
                    delete(VideoSegment).where(VideoSegment.video_id == uuid.UUID(video_id))
                )
                
                # Delete metadata
                await session.execute(
                    delete(VideoMetadata).where(VideoMetadata.video_id == uuid.UUID(video_id))
                )
                
                await session.commit()
                
                # Delete from Supabase
                try:
                    supabase = await self._get_supabase_client()
                    supabase.table('video_metadata').delete().eq('video_id', video_id).execute()
                except Exception as e:
                    logger.warning(f"Failed to delete from Supabase: {e}")
                
                # Invalidate cache
                cache_manager = await self._get_cache_manager()
                await cache_manager.delete('video_metadata', video_id)
                await cache_manager.delete('video_segments', video_id)
                
                logger.info(f"✅ Deleted video {video_id}")
                return True
                
            except Exception as e:
                await session.rollback()
                logger.error(f"❌ Failed to delete video: {e}")
                return False

    async def search_videos(self, query: str, user_id: Optional[str] = None, 
                           limit: int = 20) -> List[Video]:
        """Search videos using Supabase full-text search"""
        try:
            supabase = await self._get_supabase_client()
            
            # Build search query
            search_query = supabase.table('video_metadata').select('*')
            
            # Add text search
            search_query = search_query.text_search('filename', query)
            
            # Add user filter
            if user_id:
                search_query = search_query.or_(f'user_id.eq.{user_id},is_public.eq.true')
            else:
                search_query = search_query.eq('is_public', True)
            
            # Execute search
            result = search_query.limit(limit).execute()
            
            videos = []
            for row in result.data:
                video = Video(
                    id=row['video_id'],
                    filename=row['filename'],
                    file_size=row['file_size'],
                    status=ProcessingStatus(row['status']),
                    created_at=datetime.fromisoformat(row['created_at']),
                    user_id=row.get('user_id'),
                    storage_path=row.get('storage_path'),
                    public_url=row.get('public_url')
                )
                videos.append(video)
            
            return videos
            
        except Exception as e:
            logger.error(f"Video search failed: {e}")
            return []

# Dependency injection
_video_repository = None

async def get_video_repository() -> SupabaseVideoRepository:
    """Get video repository instance"""
    global _video_repository
    if _video_repository is None:
        _video_repository = SupabaseVideoRepository()
    return _video_repository

# Alias for backward compatibility
VideoRepository = SupabaseVideoRepository