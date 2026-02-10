import asyncio
import json
import boto3
import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import tempfile
from supabase import create_client, Client


# Your existing processor
from processors.video_processor import EnhancedVideoProcessor
from utils.database import DatabaseManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLWorker:
    def __init__(self):
        self.sqs_client = None
        self.s3_client = None
        self.video_processor = None
        self.db_manager = None
        
        # Configuration from environment
        self.queue_url = os.getenv('SQS_QUEUE_URL')
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')
        self.s3_bucket = os.getenv('S3_BUCKET', 'videorag-uploads')
        
        # Worker configuration
        self.max_messages = 1
        self.wait_time = 20
        self.visibility_timeout = 1800  # 30 minutes
        
        # Temp directory
        self.temp_dir = Path(tempfile.gettempdir()) / "ml_worker"
        self.temp_dir.mkdir(exist_ok=True)
        

    
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing ML Worker...")
        
        # AWS clients
        self.sqs_client = boto3.client('sqs', region_name=self.aws_region)
        self.s3_client = boto3.client('s3', region_name=self.aws_region)


        # Initialize Supabase client HERE
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
        
        self.supabase: Client = create_client(supabase_url, supabase_key)
        
        # Database
        self.db_manager = DatabaseManager()
        await self.db_manager.initialize()
        
        # ML models
        logger.info("Loading ML models...")
        self.video_processor = EnhancedVideoProcessor()
        await self.video_processor.initialize()
        
        logger.info("ML Worker initialized successfully")
    
    async def run(self):
        """Main worker loop"""
        await self.initialize()
        
        logger.info(f"Worker running, polling queue: {self.queue_url}")
        
        while True:
            try:
                # Receive messages
                response = self.sqs_client.receive_message(
                    QueueUrl=self.queue_url,
                    MaxNumberOfMessages=self.max_messages,
                    WaitTimeSeconds=self.wait_time,
                    VisibilityTimeout=self.visibility_timeout,
                    MessageAttributeNames=['All']
                )
                
                messages = response.get('Messages', [])
                
                if messages:
                    for message in messages:
                        await self.process_message(message)
                
            except KeyboardInterrupt:
                logger.info("Shutting down worker...")
                break
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)
                await asyncio.sleep(5)
    
    async def process_message(self, message: Dict[str, Any]):
        """Process a single SQS message"""
        receipt_handle = message['ReceiptHandle']
        message_id = message['MessageId']
        
        try:
            # Parse message
            body = json.loads(message['Body'])
            video_id = body.get('video_id')
            s3_key = body.get('s3_key')
            
            logger.info(f"Processing video {video_id} (message: {message_id})")
            
            # Download video from S3
            video_path = await self.download_video(s3_key, video_id)
            
            # Process video
            results = await self.video_processor.process_video(
                video_path, 
                video_id
            )
            
            # Save results to database
            await self.save_results(video_id, results)
            
            # Update video status
            await self.update_video_status(video_id, 'completed')
            
            # Delete message (success)
            self.sqs_client.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle
            )
            
            logger.info(f"Successfully processed video {video_id}")
            
            # Cleanup
            if os.path.exists(video_path):
                os.remove(video_path)
            
        except Exception as e:
            logger.error(f"Failed to process message {message_id}: {e}", exc_info=True)
            
            # Update video status to failed
            try:
                video_id = json.loads(message['Body']).get('video_id')
                if video_id:
                    await self.update_video_status(video_id, 'failed', str(e))
            except:
                pass
    
    async def download_video(self, s3_key: str, video_id: str) -> str:
        """Download video from S3"""
        local_path = self.temp_dir / f"{video_id}.mp4"
        
        logger.info(f"Downloading {s3_key} from S3...")
        
        self.s3_client.download_file(
            self.s3_bucket,
            s3_key,
            str(local_path)
        )
        
        return str(local_path)
    
    async def save_results(self, video_id: str, results: Dict[str, Any]):
        """Save processing results to database using Supabase"""
        try:
            import os
            from supabase import create_client
            
            # Get Supabase credentials from environment
            supabase_url = os.getenv('SUPABASE_URL')
            supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
            
            if not supabase_url or not supabase_key:
                logger.warning("Supabase credentials not set - skipping database save")
                return
            
            # Create Supabase client
            supabase = create_client(supabase_url, supabase_key)
            
            # Prepare update data
            update_data = {
                'status': 'completed',
                'total_scenes': results.get('total_scenes', 0),
                'total_segments': len(results.get('hierarchical_segments', [])),
                'processing_results': results,
                'updated_at': datetime.utcnow().isoformat()
            }
            
            # Update video record
            response = supabase.table('video_metadata').update(update_data).eq('video_id', video_id).execute()
            
            if response.data:
                logger.info(f"✅ Results saved to database for video {video_id}")
            else:
                logger.warning(f"⚠️ No data returned when saving results for {video_id}")
                
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    async def update_video_status(self, video_id: str, status: str, error: str = None):
        """Update video status using Supabase REST API"""
        try:
            response = self.supabase.table('video_metadata').update({
                'status': status,
                'processed_at': 'now()' if status == 'completed' else None,
                'error_message': None if status == 'completed' else 'Processing failed'
            }).eq('video_id', video_id).execute()
            
            logger.info(f"Updated video {video_id} status to {status}")
        except Exception as e:
            logger.error(f"Failed to update status: {e}")
            raise
        # async with self.db_manager.get_session() as session:
        #     from sqlalchemy import update
        #     from utils.models import VideoMetadata
            
        #     values = {
        #         'status': status,
        #         'updated_at': datetime.utcnow()
        #     }
            
        #     if error:
        #         values['error_message'] = error
            
        #     if status == 'completed':
        #         values['processed_at'] = datetime.utcnow()
            
        #     query = update(VideoMetadata).where(
        #         VideoMetadata.video_id == video_id
        #     ).values(**values)
            
        #     await session.execute(query)
        #     await session.commit()

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    worker = MLWorker()
    asyncio.run(worker.run())