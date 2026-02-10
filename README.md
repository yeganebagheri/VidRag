# VidRag

How to run:

1. we need two terminal:
a) api
b) ml-worker



Terminal 1 - Start the API
bashcd /Users/yeganebagheri/Desktop/VR-VOD
source venv/bin/activate
cd api
uvicorn src.api.main:app --reload

Terminal 2 - Start the ML Worker
cd /Users/yeganebagheri/Desktop/VR-VOD
source venv/bin/activate
cd ml-worker
python src/worker.py

Terminal 3 - Test the API
Open a third terminal window for testing:
bash# Test health check
curl http://127.0.0.1:8000/health

# Or open in browser
open http://127.0.0.1:8000/docs



#curl -X POST "http://127.0.0.1:8000/api/v1/videos/upload" \ -F "file=/Users/yeganebagheri/Desktop/VR-VOD/test_video.mp4"
 this one should work-> curl -X POST "http://127.0.0.1:8000/api/v1/videos/upload" -F "file=@/Users/yeganebagheri/Desktop/VR-VOD/test_video.mp4"

for installing in venv:
 ./venv/bin/pip install openai-whisper

 # Explicitly tell pip to install in the venv
pip install --ignore-installed --no-user faster-whisper

Terminal 4 - MECD-Benchmark
cd ~/Desktop/VR-VOD/MECD-Benchmark
source mecd_env/bin/activate