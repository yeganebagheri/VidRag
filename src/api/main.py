# src/api/main.py - Fixed imports and dependencies

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from src.api.routers import videos, queries
import logging
import asyncio
from contextlib import asynccontextmanager
from src.shared.config import settings, validate_aws_config
from src.core.database.connection import initialize_database, cleanup_database, db_manager
from src.core.cache.cache_manager import get_cache_manager

# Import processors and search (with proper error handling)
try:
    from src.core.services.video_processor import enhanced_video_processor
except ImportError:
    enhanced_video_processor = None
    logging.warning("Video processor not available")

try:
    from src.core.search.vector_search import enhanced_vector_search
except ImportError:
    enhanced_vector_search = None
    logging.warning("Vector search not available")

try:
    from src.core.events.sqs_publisher import get_sqs_event_publisher
except ImportError:
    get_sqs_event_publisher = None
    logging.warning("SQS publisher not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced application lifespan management with graceful fallbacks"""
    logger.info("🚀 Initializing VideoRAG API...")
    
    try:
        # Initialize Supabase database (required)
        await initialize_database()
        logger.info("✅ Database initialized")
        
        # Initialize cache manager (optional)
        try:
            cache_manager = await get_cache_manager()
            logger.info("✅ Cache manager initialized")
        except Exception as e:
            logger.warning(f"⚠️ Cache manager initialization failed: {e}")
        
        # Initialize SQS event publisher (optional)
        try:
            if get_sqs_event_publisher:
                validate_aws_config()
                sqs_publisher = await get_sqs_event_publisher()
                logger.info("✅ SQS event publisher initialized")
            else:
                logger.info("ℹ️ SQS functionality not available")
        except Exception as e:
            logger.warning(f"⚠️ SQS initialization failed: {e}")
        
        # Initialize enhanced video processor (optional)
        try:
            if enhanced_video_processor:
                await enhanced_video_processor.initialize()
                logger.info("✅ Video processor initialized")
            else:
                logger.info("ℹ️ Video processor not available")
        except Exception as e:
            logger.warning(f"⚠️ Video processor initialization failed: {e}")
        
        # Initialize enhanced vector search (optional)
        try:
            if enhanced_vector_search:
                await enhanced_vector_search.initialize()
                logger.info("✅ Vector search initialized")
            else:
                logger.info("ℹ️ Vector search not available")
        except Exception as e:
            logger.warning(f"⚠️ Vector search initialization failed: {e}")
        
        logger.info("✅ VideoRAG API ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize application: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("🔄 Shutting down VideoRAG API...")
    try:
        await cleanup_database()
        
        # Close cache manager
        try:
            cache_manager = await get_cache_manager()
            await cache_manager.close()
        except:
            pass
        
        # Close SQS publisher
        try:
            if get_sqs_event_publisher:
                sqs_publisher = await get_sqs_event_publisher()
                await sqs_publisher.close()
        except:
            pass
        
        logger.info("✅ Cleanup completed")
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")

app = FastAPI(
    title="VideoRAG API", 
    version="2.1.0",
    description="Enhanced Video Retrieval Augmented Generation API with Supabase Integration",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"] if settings.DEBUG else ["yourdomain.com", "*.yourdomain.com"]
)

# CORS middleware with enhanced configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:8000",  # API server
        "https://yourdomain.com",  # Production frontend
        settings.SUPABASE_URL,  # Supabase dashboard
    ] if not settings.DEBUG else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Include routers
app.include_router(videos.router, prefix="/api/v1/videos", tags=["videos"])
app.include_router(queries.router, prefix="/api/v1/queries", tags=["queries"])

@app.get("/")
async def root():
    """Root endpoint with feature list"""
    features = [
        "video_upload_to_supabase_storage", 
        "supabase_database_integration",
        "row_level_security",
        "real_time_updates"
    ]
    
    # Add optional features if available
    if enhanced_video_processor:
        features.extend([
            "enhanced_transcription", 
            "scene_detection",
            "hierarchical_ocr"
        ])
    
    if enhanced_vector_search:
        features.extend([
            "multimodal_vector_search",
            "hierarchical_search",
            "adaptive_fusion"
        ])
    
    if get_sqs_event_publisher:
        features.extend([
            "sqs_event_driven_processing",
            "distributed_workers",
            "auto_scaling_queues"
        ])
    
    return {
        "message": "VideoRAG API with Supabase Integration",
        "version": "2.1.0",
        "features": features,
        "integrations": {
            "database": "PostgreSQL with pgvector (Supabase)",
            "storage": "Supabase Storage for videos",
            "auth": "Supabase Auth ready",
            "realtime": "Supabase Realtime for live updates",
            "event_system": "AWS SQS for distributed processing" if get_sqs_event_publisher else "Local events",
            "caching": "Redis for performance",
            "monitoring": "CloudWatch ready" if get_sqs_event_publisher else "Basic logging"
        },
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Enhanced health check with component status"""
    try:
        # Check database health (required)
        db_health = await db_manager.health_check()
        
        # Check cache health (optional)
        cache_stats = {"redis_connected": False}
        try:
            cache_manager = await get_cache_manager()
            cache_stats = await cache_manager.get_stats()
        except:
            pass
        
        # Check SQS health (optional)
        sqs_stats = {"initialized": False}
        try:
            if get_sqs_event_publisher:
                sqs_publisher = await get_sqs_event_publisher()
                sqs_stats = await sqs_publisher.get_metrics()
        except:
            pass
        
        # Check models status (optional)
        models_status = {
            "video_processor": enhanced_video_processor is not None,
            "vector_search": enhanced_vector_search is not None
        }
        
        if enhanced_video_processor:
            try:
                models_status.update({
                    "whisper": enhanced_video_processor.whisper_model is not None,
                    "ocr": enhanced_video_processor.ocr_reader is not None,
                    "clip": enhanced_video_processor.clip_model is not None,
                    "embeddings": enhanced_video_processor.embedding_model is not None,
                    "nlp": enhanced_video_processor.nlp is not None
                })
            except:
                pass
        
        if enhanced_vector_search:
            try:
                models_status["search_embeddings"] = enhanced_vector_search.embedding_model is not None
            except:
                pass
        
        overall_status = "healthy" if db_health["status"] == "healthy" else "degraded"
        
        return {
            "status": overall_status,
            "timestamp": asyncio.get_event_loop().time(),
            "database": db_health,
            "cache": cache_stats,
            "sqs": sqs_stats,
            "models_loaded": models_status,
            "capabilities": {
                "supabase_storage": True,
                "row_level_security": True,
                "real_time_updates": True,
                "hierarchical_segmentation": enhanced_video_processor is not None,
                "scene_detection": enhanced_video_processor is not None,
                "multimodal_fusion": enhanced_vector_search is not None,
                "sqs_event_processing": get_sqs_event_publisher is not None,
                "distributed_workers": get_sqs_event_publisher is not None
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": asyncio.get_event_loop().time()
        }

@app.get("/api/v1/stats")
async def get_system_stats():
    """Get comprehensive system statistics"""
    try:
        basic_stats = {
            "api_version": "2.1.0",
            "components_available": {
                "video_processor": enhanced_video_processor is not None,
                "vector_search": enhanced_vector_search is not None,
                "sqs_publisher": get_sqs_event_publisher is not None
            }
        }
        
        # Add processing stats if available
        if enhanced_vector_search:
            try:
                basic_stats.update({
                    "processed_videos": len(enhanced_vector_search.video_embeddings),
                    "total_segments": sum(
                        len(segments) for segments in enhanced_vector_search.video_embeddings.values()
                    ),
                    "total_scenes": sum(
                        len(scenes) for scenes in enhanced_vector_search.scene_embeddings.values()
                    )
                })
            except:
                pass
        
        # Get database stats
        db_health = await db_manager.health_check()
        
        # Get cache stats
        cache_stats = {}
        try:
            cache_manager = await get_cache_manager()
            cache_stats = await cache_manager.get_stats()
        except:
            pass
        
        # Get SQS stats
        sqs_stats = {}
        try:
            if get_sqs_event_publisher:
                sqs_publisher = await get_sqs_event_publisher()
                sqs_stats = await sqs_publisher.get_metrics()
        except:
            pass
        
        return {
            **basic_stats,
            "database_stats": {
                "status": db_health["status"],
                "pool_stats": db_health.get("pool_stats", {}),
                "supabase_status": db_health.get("supabase_status", "unknown")
            },
            "cache_stats": cache_stats,
            "sqs_stats": sqs_stats,
            "integrations": {
                "supabase": {
                    "storage_enabled": True,
                    "auth_enabled": bool(settings.SUPABASE_ANON_KEY),
                    "realtime_available": True,
                    "rls_enabled": True
                },
                "aws": {
                    "sqs_enabled": get_sqs_event_publisher is not None,
                    "region": settings.AWS_REGION,
                    "s3_enabled": bool(settings.S3_BUCKET),
                    "opensearch_enabled": bool(settings.OPENSEARCH_HOST)
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system stats: {str(e)}")

@app.get("/api/v1/config")
async def get_api_config():
    """Get API configuration (non-sensitive)"""
    return {
        "version": "2.1.0",
        "debug": settings.DEBUG,
        "features": {
            "supabase_integration": True,
            "sqs_event_processing": get_sqs_event_publisher is not None,
            "vector_search": enhanced_vector_search is not None,
            "file_upload": True,
            "real_time_updates": True,
            "user_authentication": bool(settings.SUPABASE_ANON_KEY),
            "distributed_workers": get_sqs_event_publisher is not None
        },
        "limits": {
            "max_file_size": "1GB",
            "supported_formats": [".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"],
            "max_concurrent_uploads": 5,
            "transcription_timeout": settings.TRANSCRIPTION_TIMEOUT,
            "processing_timeout": settings.SCENE_DETECTION_TIMEOUT
        },
        "storage": {
            "provider": "supabase",
            "bucket": settings.SUPABASE_STORAGE_BUCKET,
            "signed_urls": True
        }
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Not Found",
        "message": "The requested resource was not found",
        "path": str(request.url.path)
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return {
        "error": "Internal Server Error", 
        "message": "An internal server error occurred",
        "debug": str(exc) if settings.DEBUG else "Contact support for assistance"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug"
    )