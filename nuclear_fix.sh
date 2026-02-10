#!/bin/bash
# nuclear_fix.sh - Complete environment isolation fix

echo "Nuclear environment fix - this will completely isolate your environment"

# Step 1: Completely remove virtual environment
rm -rf venv

# Step 2: Clear Python cache
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Step 3: Find and use the best Python version
find_python() {
    for version in python3.11 python3.10 python3.9; do
        if command -v $version &> /dev/null; then
            echo $version
            return 0
        fi
    done
    echo "python3"
}

PYTHON_CMD=$(find_python)
echo "Using Python: $($PYTHON_CMD --version)"

# Step 4: Create completely isolated environment
$PYTHON_CMD -m venv venv --clear
source venv/bin/activate

# Step 5: Force pip to use only venv
export PYTHONPATH=""
export PIP_REQUIRE_VIRTUALENV=1

# Step 6: Verify isolation
echo "Python location: $(which python)"
echo "Pip location: $(which pip)"

# Step 7: Install core packages with isolation
pip install --no-deps --upgrade pip setuptools wheel

# Step 8: Install packages step by step with verification
echo "Installing NumPy 1.x..."
pip install --no-deps "numpy>=1.24.4,<2.0"

echo "Installing core dependencies..."
pip install --no-deps typing-extensions pydantic-core annotated-types

echo "Installing Pydantic..."
pip install "pydantic>=2.5.0,<3.0"

echo "Installing FastAPI stack..."
pip install "fastapi>=0.104.1"
pip install "uvicorn[standard]>=0.24.0"
pip install "pydantic-settings>=2.1.0"

echo "Installing database packages..."
pip install "sqlalchemy>=2.0.23"
pip install "asyncpg>=0.29.0"
pip install "psycopg2-binary>=2.9.9"

echo "Installing Supabase..."
pip install "httpx>=0.24.0,<0.29.0"
pip install "supabase>=2.0.2"

echo "Installing utilities..."
pip install "aiofiles>=23.2.1"
pip install "python-multipart>=0.0.6"
pip install "redis>=5.0.1"
pip install "boto3>=1.34.0"

# Step 9: Test core functionality
echo "Testing core imports..."
python -c "
import sys
print(f'Python executable: {sys.executable}')
print(f'Python path: {sys.path[0]}')

try:
    import numpy
    print(f'NumPy: {numpy.__version__} from {numpy.__file__}')
    assert not numpy.__version__.startswith('2.'), 'NumPy 2.x detected!'
    
    import fastapi
    print(f'FastAPI: OK from {fastapi.__file__}')
    
    import supabase
    print(f'Supabase: OK from {supabase.__file__}')
    
    print('Core packages working in isolated environment!')
    
except Exception as e:
    print(f'Error: {e}')
    exit(1)
"

# Step 10: Create minimal working main.py that bypasses problematic imports
cat > minimal_main.py << 'EOF'
#!/usr/bin/env python3
"""Minimal VideoRAG API that works without problematic ML packages"""

import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Create app without importing problematic modules
app = FastAPI(
    title="VideoRAG API (Minimal Mode)",
    version="1.0.0",
    description="Basic VideoRAG API without heavy ML dependencies"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "VideoRAG API (Minimal Mode)",
        "status": "running",
        "features": ["basic_api", "database_ready"],
        "note": "ML features disabled to prevent dependency conflicts"
    }

@app.get("/health")
async def health_check():
    try:
        import numpy
        numpy_version = numpy.__version__
        numpy_ok = not numpy_version.startswith('2.')
        
        return {
            "status": "healthy" if numpy_ok else "degraded",
            "python_version": sys.version,
            "numpy_version": numpy_version,
            "numpy_compatible": numpy_ok,
            "environment": "isolated"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/config")
async def get_config():
    """Test config loading"""
    try:
        from src.shared.config import settings
        return {
            "config_loaded": True,
            "debug": settings.DEBUG,
            "api_host": settings.API_HOST,
            "api_port": settings.API_PORT
        }
    except Exception as e:
        return {
            "config_loaded": False,
            "error": str(e)
        }

# Basic video endpoints without ML dependencies
@app.post("/api/v1/videos/upload")
async def upload_video():
    return {"message": "Upload endpoint available - implement based on your needs"}

@app.get("/api/v1/videos")
async def list_videos():
    return {"videos": [], "message": "Video listing endpoint ready"}

@app.post("/api/v1/queries/search")
async def search_videos():
    return {"results": [], "message": "Search endpoint ready - add ML components when environment is stable"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
EOF

# Step 11: Test the minimal app
echo "Testing minimal app..."
python -c "
try:
    from minimal_main import app
    print('Minimal app imports successfully!')
except Exception as e:
    print(f'Minimal app import failed: {e}')
"

echo ""
echo "Nuclear fix completed!"
echo ""
echo "Start options:"
echo "1. Minimal app (guaranteed to work): python minimal_main.py"
echo "2. Full app (if env is fixed): python -m uvicorn src.api.main:app --reload"
echo ""
echo "The minimal app provides:"
echo "- Basic FastAPI server"
echo "- Health checks"
echo "- Configuration testing"
echo "- Stub endpoints for videos and search"
echo ""
echo "Once this works, you can gradually add back ML components"