from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from src.api.routers import videos, queries
from src.core.database.connection import initialize_database, cleanup_database, db_manager
from src.core.cache.cache_manager import get_cache_manager
from src.core.events.sqs_publisher import get_sqs_event_publisher
from dotenv import load_dotenv
import logging
import asyncio
from contextlib import asynccontextmanager
from src.shared.config import settings, validate_aws_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Simplified API lifespan - ML processing handled by separate worker"""
    logger.info("🚀 Initializing VideoRAG API...")
    
    try:
        # Validate AWS configuration
        validate_aws_config()
        logger.info("✅ AWS configuration validated")
        
        # Initialize Supabase database
        await initialize_database()
        
        # Initialize cache manager
        cache_manager = await get_cache_manager()
        logger.info("✅ Cache manager initialized")
        
        # Initialize SQS event publisher
        #sqs_publisher = await get_sqs_event_publisher()
        #logger.info("✅ SQS event publisher initialized")
        
        logger.info("🎯 VideoRAG API ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize application: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("🔄 Shutting down VideoRAG API...")
    try:
        await cleanup_database()
        
        cache_manager = await get_cache_manager()
        await cache_manager.close()
        
        sqs_publisher = await get_sqs_event_publisher()
        await sqs_publisher.close()
        
        logger.info("✅ Cleanup completed")
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")

app = FastAPI(
    title="VideoRAG API", 
    version="2.0.0",
    description="Video Retrieval API - ML processing handled by separate worker service",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"] if settings.DEBUG else ["yourdomain.com", "*.yourdomain.com"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else [
        "http://localhost:3000",
        settings.SUPABASE_URL,
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(videos.router, prefix="/api/v1/videos", tags=["videos"])
app.include_router(queries.router, prefix="/api/v1/queries", tags=["queries"])

@app.get("/")
async def root():
    return {
        "message": "VideoRAG API - Upload videos for ML processing",
        "version": "2.0.0",
        "architecture": "Separated API + ML Worker",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    try:
        db_health = await db_manager.health_check()
        cache_manager = await get_cache_manager()
        cache_stats = await cache_manager.get_stats()
        sqs_publisher = await get_sqs_event_publisher()
        sqs_stats = await sqs_publisher.get_metrics()
        
        return {
            "status": "healthy" if db_health["status"] == "healthy" else "degraded",
            "database": db_health,
            "cache": cache_stats,
            "sqs": sqs_stats,
            "note": "ML processing handled by separate worker service"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )