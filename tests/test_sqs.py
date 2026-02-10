# test_sqs.py
import boto3
import json

sqs = boto3.client('sqs', region_name='eu-west-1')
queue_url = 'https://sqs.eu-west-1.amazonaws.com/921436147967/video-processing-queue'

message = {
    'video_id': '9304f5a7-b8e0-473a-b5fb-6a9fc3c1cea5',
    's3_key': 'videos/9304f5a7-b8e0-473a-b5fb-6a9fc3c1cea5/test_video.mp4',
    's3_bucket': 'videorag-uploads',
    'filename': 'test_video.mp4'
}

response = sqs.send_message(
    QueueUrl=queue_url,
    MessageBody=json.dumps(message)
)

print(f"Message sent! MessageId: {response['MessageId']}")