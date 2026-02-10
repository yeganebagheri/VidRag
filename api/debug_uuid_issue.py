# debug_uuid_issue.py
# Run this to identify where the UUID error is happening

import sys
import os

# Add your API path
sys.path.insert(0, '/Users/yeganebagheri/Desktop/VR-VOD/api')

print("🔍 Debugging UUID Issue")
print("=" * 60)

# Test 1: UUID generation
print("\n1. Testing UUID generation...")
try:
    import uuid
    video_id = str(uuid.uuid4())
    print(f"   ✅ Generated: {video_id}")
    print(f"   ✅ Type: {type(video_id)}")
    print(f"   ✅ Length: {len(video_id)}")
    
    # Validate
    uuid.UUID(video_id)
    print(f"   ✅ Validation: PASSED")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Check if issue is in Pydantic models
print("\n2. Testing Pydantic Video models...")
try:
    from src.core.models.video import Video, VideoCreate
    from datetime import datetime
    
    # Create VideoCreate instance
    video_create = VideoCreate(
        filename="test.mp4",
        title="test",
        user_id="test-user-id"
    )
    print(f"   ✅ VideoCreate: {video_create}")
    
    # Create Video instance
    video = Video(
        video_id=str(uuid.uuid4()),
        filename="test.mp4",
        title="test",
        status="uploaded",
        user_id="test-user-id",
        created_at=datetime.utcnow()
    )
    print(f"   ✅ Video: {video}")
    print(f"   ✅ video.video_id: {video.video_id}")
    print(f"   ✅ video.video_id type: {type(video.video_id)}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Check database connection
print("\n3. Testing database UUID handling...")
try:
    from dotenv import load_dotenv
    load_dotenv()
    
    from supabase import create_client
    
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    
    if not supabase_url or not supabase_key:
        print("   ⚠️  Supabase credentials not found in .env")
    else:
        print(f"   ✅ Supabase URL: {supabase_url[:30]}...")
        
        supabase = create_client(supabase_url, supabase_key)
        
        # Try to insert a test record with UUID
        test_video_id = str(uuid.uuid4())
        print(f"   📝 Testing insert with video_id: {test_video_id}")
        
        test_data = {
            'video_id': test_video_id,
            'filename': 'test_debug.mp4',
            'title': 'Debug Test',
            'user_id': 'test-user-id',
            'status': 'uploaded',
            'created_at': datetime.utcnow().isoformat()
        }
        
        print(f"   📝 Insert data: {test_data}")
        
        # Try insert
        response = supabase.table('video_metadata').insert(test_data).execute()
        
        if response.data:
            print(f"   ✅ Insert successful!")
            print(f"   ✅ Returned data: {response.data[0]}")
            
            # Clean up - delete test record
            supabase.table('video_metadata').delete().eq('video_id', test_video_id).execute()
            print(f"   ✅ Cleanup: Test record deleted")
        else:
            print(f"   ❌ Insert failed: No data returned")
            
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check video_metadata table schema
print("\n4. Checking video_metadata table schema...")
try:
    if supabase:
        # Get a sample record to see schema
        response = supabase.table('video_metadata').select('*').limit(1).execute()
        
        if response.data and len(response.data) > 0:
            print(f"   ✅ Sample record found:")
            sample = response.data[0]
            for key, value in sample.items():
                print(f"      • {key}: {type(value).__name__} = {str(value)[:50]}")
        else:
            print(f"   ⚠️  No records in video_metadata table")
            print(f"      This is OK if it's a new database")
            
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("Debug complete!")
print("\nIf you see any ❌ errors above, that's where the issue is.")
print("Share the output with me for further diagnosis.")