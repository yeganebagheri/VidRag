# def lambda_handler(event, context):
#     print("Received event:", event)
#     # Process uploaded S3 object here
#     return {"statusCode": 200}

# step 1- read the event information- all video metadata that I recive from s3

# step 2- extract the video meta data and object key from the event

# step 3- create a record in supabase

# step 4- generate a presignet url for the video

# step 5- create aN EVENT IN SQS
import json
import boto3
import os
import logging
from datetime import datetime
import uuid
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
sqs = boto3.client('sqs')
s3 = boto3.client('s3')

# Environment variables
SQS_QUEUE_URL = os.environ.get('SQS_QUEUE_URL')
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_SERVICE_KEY = os.environ.get('SUPABASE_SERVICE_KEY')

# Import Supabase client
try:
    from supabase import create_client, Client
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
except ImportError:
    logger.error("Supabase client not available")
    supabase = None

def lambda_handler(event, context):
    """
    AWS Lambda handler for S3 events
    
    Steps:
    1. Read S3 event information and extract video metadata
    2. Extract video metadata and object key from the event
    3. Create a record in Supabase
    4. Generate a presigned URL for the video
    5. Create an event in SQS
    """
    
    logger.info(f"Received S3 event: {json.dumps(event)}")
    
    try:
        # Step 1: Read the event information - all video metadata received from S3
        s3_records = event.get('Records', [])
        
        if not s3_records:
            logger.error("No S3 records found in event")
            return {"statusCode": 400, "body": "No S3 records found"}
        
        results = []
        
        for record in s3_records:
            try:
                # Step 2: Extract video metadata and object key from the event
                s3_data = record['s3']
                bucket_name = s3_data['bucket']['name']
                object_key = s3_data['object']['key']
                file_size = s3_data['object']['size']
                
                # Extract additional metadata
                event_name = record['eventName']
                event_time = record['eventTime']
                
                logger.info(f"Processing S3 object: {bucket_name}/{object_key}")
                
                # Validate it's a video file
                if not is_video_file(object_key):
                    logger.warning(f"Skipping non-video file: {object_key}")
                    continue
                
                # Generate video ID
                video_id = str(uuid.uuid4())
                filename = object_key.split('/')[-1]  # Extract filename from key
                
                # Get additional file metadata from S3
                file_metadata = get_s3_object_metadata(bucket_name, object_key)
                
                # Step 3: Create a record in Supabase
                supabase_record = create_supabase_record(
                    video_id=video_id,
                    filename=filename,
                    s3_bucket=bucket_name,
                    s3_key=object_key,
                    file_size=file_size,
                    metadata=file_metadata
                )
                
                if not supabase_record:
                    logger.error(f"Failed to create Supabase record for {object_key}")
                    continue
                
                # Step 4: Generate a presigned URL for the video
                presigned_url = generate_presigned_url(bucket_name, object_key)
                
                # Step 5: Create an event in SQS
                sqs_event = create_sqs_event(
                    video_id=video_id,
                    filename=filename,
                    s3_bucket=bucket_name,
                    s3_key=object_key,
                    file_size=file_size,
                    presigned_url=presigned_url,
                    event_time=event_time,
                    metadata=file_metadata
                )
                
                if sqs_event:
                    logger.info(f"Successfully processed S3 object: {object_key}")
                    results.append({
                        "video_id": video_id,
                        "object_key": object_key,
                        "status": "success"
                    })
                else:
                    logger.error(f"Failed to send SQS event for {object_key}")
                    results.append({
                        "video_id": video_id,
                        "object_key": object_key,
                        "status": "sqs_failed"
                    })
                
            except Exception as e:
                logger.error(f"Error processing S3 record: {str(e)}")
                results.append({
                    "object_key": record.get('s3', {}).get('object', {}).get('key', 'unknown'),
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "S3 events processed",
                "results": results,
                "processed_count": len(results)
            })
        }
        
    except Exception as e:
        logger.error(f"Lambda handler error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": "Internal server error",
                "message": str(e)
            })
        }

def is_video_file(object_key: str) -> bool:
    """Check if the file is a video based on extension"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.flv', '.wmv'}
    file_extension = os.path.splitext(object_key.lower())[1]
    return file_extension in video_extensions

def get_s3_object_metadata(bucket_name: str, object_key: str) -> Dict[str, Any]:
    """Get additional metadata from S3 object"""
    try:
        response = s3.head_object(Bucket=bucket_name, Key=object_key)
        
        metadata = {
            "content_type": response.get('ContentType'),
            "last_modified": response.get('LastModified').isoformat() if response.get('LastModified') else None,
            "etag": response.get('ETag'),
            "content_length": response.get('ContentLength'),
            "storage_class": response.get('StorageClass', 'STANDARD')
        }
        
        # Add custom metadata if present
        if 'Metadata' in response:
            metadata.update(response['Metadata'])
        
        return metadata
        
    except Exception as e:
        logger.warning(f"Failed to get S3 metadata for {object_key}: {str(e)}")
        return {}

def create_supabase_record(video_id: str, filename: str, s3_bucket: str, 
                          s3_key: str, file_size: int, metadata: Dict[str, Any]) -> Optional[Dict]:
    """Create a record in Supabase video_metadata table"""
    
    if not supabase:
        logger.error("Supabase client not available")
        return None
    
    try:
        # Prepare the record data matching your schema
        record_data = {
            "video_id": video_id,
            "filename": filename,
            "file_size": file_size,
            "status": "uploaded",  # Initial status
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            
            # Storage information
            "storage_path": f"s3://{s3_bucket}/{s3_key}",
            "public_url": None,  # Will be set when generating presigned URL
            
            # Additional metadata
            "processing_results": {
                "s3_metadata": metadata,
                "upload_source": "s3_event",
                "bucket": s3_bucket,
                "object_key": s3_key
            },
            
            # Set defaults for other fields
            "user_id": None,  # Set if you have user context
            "is_public": False,
            "duration": None,  # Will be populated during processing
            "fps": None,
            "resolution": None,
            "total_scenes": None,
            "total_segments": None,
            "knowledge_graph": None,
            "error_message": None,
            "processed_at": None
        }
        
        # Insert into Supabase
        result = supabase.table('video_metadata').insert(record_data).execute()
        
        if result.data:
            logger.info(f"Created Supabase record for video_id: {video_id}")
            return result.data[0]
        else:
            logger.error(f"Failed to create Supabase record: {result}")
            return None
            
    except Exception as e:
        logger.error(f"Error creating Supabase record: {str(e)}")
        return None

def generate_presigned_url(bucket_name: str, object_key: str, expiration: int = 3600) -> Optional[str]:
    """Generate a presigned URL for the S3 object"""
    try:
        presigned_url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': object_key},
            ExpiresIn=expiration
        )
        
        logger.info(f"Generated presigned URL for {object_key}")
        return presigned_url
        
    except Exception as e:
        logger.error(f"Error generating presigned URL: {str(e)}")
        return None

def create_sqs_event(video_id: str, filename: str, s3_bucket: str, s3_key: str, 
                    file_size: int, presigned_url: Optional[str], event_time: str,
                    metadata: Dict[str, Any]) -> bool:
    """Create and send an event to SQS queue"""
    
    if not SQS_QUEUE_URL:
        logger.error("SQS_QUEUE_URL environment variable not set")
        return False
    
    try:
        # Create the SQS message payload
        message_body = {
            "event_type": "video.uploaded",
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "correlation_id": video_id,
            "payload": {
                "video_id": video_id,
                "filename": filename,
                "s3_bucket": s3_bucket,
                "s3_key": s3_key,
                "file_size": file_size,
                "presigned_url": presigned_url,
                "upload_time": event_time,
                "metadata": metadata,
                "next_stage": "transcription",
                "processing_pipeline": [
                    "transcription",
                    "scene_detection", 
                    "segmentation",
                    "embedding_generation",
                    "indexing"
                ]
            }
        }
        
        # Send message to SQS
        response = sqs.send_message(
            QueueUrl=SQS_QUEUE_URL,
            MessageBody=json.dumps(message_body),
            MessageAttributes={
                'EventType': {
                    'StringValue': 'video.uploaded',
                    'DataType': 'String'
                },
                'VideoId': {
                    'StringValue': video_id,
                    'DataType': 'String'
                },
                'CorrelationId': {
                    'StringValue': video_id,
                    'DataType': 'String'
                },
                'Source': {
                    'StringValue': 's3_upload_handler',
                    'DataType': 'String'
                }
            }
        )
        
        if response.get('MessageId'):
            logger.info(f"Sent SQS message for video_id: {video_id}, MessageId: {response['MessageId']}")
            return True
        else:
            logger.error(f"Failed to send SQS message: {response}")
            return False
            
    except Exception as e:
        logger.error(f"Error sending SQS message: {str(e)}")
        return False

def update_supabase_with_presigned_url(video_id: str, presigned_url: str) -> bool:
    """Update the Supabase record with the presigned URL"""
    
    if not supabase or not presigned_url:
        return False
    
    try:
        result = supabase.table('video_metadata').update({
            "public_url": presigned_url,
            "updated_at": datetime.utcnow().isoformat()
        }).eq('video_id', video_id).execute()
        
        if result.data:
            logger.info(f"Updated Supabase record with presigned URL for video_id: {video_id}")
            return True
        else:
            logger.error(f"Failed to update Supabase record with presigned URL")
            return False
            
    except Exception as e:
        logger.error(f"Error updating Supabase record: {str(e)}")
        return False