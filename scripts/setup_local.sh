# scripts/setup_local.sh
#!/bin/bash

echo "🚀 Setting up VideoRAG local development environment..."

# Check if Python 3.11+ is installed
python3 --version || { echo "❌ Python 3.11+ required"; exit 1; }

# Check Docker installation and start if needed
echo "🐳 Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first:"
    echo "   - macOS: https://docs.docker.com/desktop/install/mac-install/"
    echo "   - Windows: https://docs.docker.com/desktop/install/windows-install/"
    echo "   - Linux: https://docs.docker.com/engine/install/"
    exit 1
fi

if ! docker info > /dev/null 2>&1; then
    echo "⚠️  Docker is not running. Attempting to start Docker..."
    
    # Try to start Docker on different platforms
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        echo "Starting Docker Desktop on macOS..."
        open -a Docker
        echo "⏳ Waiting for Docker to start (this may take a minute)..."
        while ! docker info > /dev/null 2>&1; do
            sleep 2
            echo -n "."
        done
        echo ""
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        echo "Starting Docker service on Linux..."
        sudo systemctl start docker
        sleep 5
    else
        echo "❌ Please start Docker manually and run this script again."
        echo "   - On Windows: Start Docker Desktop"
        echo "   - On macOS: Start Docker Desktop"
        echo "   - On Linux: sudo systemctl start docker"
        exit 1
    fi
fi

echo "✅ Docker is running!"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies with better error handling
echo "📥 Installing Python dependencies..."
pip install --upgrade pip

# Install PyTorch first (CPU version for local development)
echo "🔥 Installing PyTorch (CPU version for local dev)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other ML dependencies
echo "🤖 Installing ML dependencies..."
pip install transformers sentence-transformers

# Install remaining dependencies
echo "📦 Installing remaining dependencies..."
pip install -r requirements.txt

# Start infrastructure services
echo "🐳 Starting infrastructure services..."
cd docker
docker-compose up -d redis postgres opensearch localstack

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
echo "Checking Redis..."
timeout 30 bash -c 'until docker-compose exec -T redis redis-cli ping | grep -q PONG; do sleep 1; done' || echo "⚠️  Redis might need more time"

echo "Checking PostgreSQL..."
timeout 30 bash -c 'until docker-compose exec -T postgres pg_isready -U videorag_user; do sleep 1; done' || echo "⚠️  PostgreSQL might need more time"

echo "Checking OpenSearch..."
timeout 60 bash -c 'until curl -s http://localhost:9200/_cluster/health | grep -q "yellow\|green"; do sleep 2; done' || echo "⚠️  OpenSearch might need more time"

# Setup LocalStack S3 bucket
echo "🪣 Setting up LocalStack S3 bucket..."
sleep 5  # Give LocalStack time to start
aws --endpoint-url=http://localhost:4566 s3 mb s3://videorag-videos-local 2>/dev/null || echo "⚠️  S3 bucket might already exist"

# Run database migrations (you'll need to implement these)
echo "🗄️ Database setup..."
cd ..
# python -m alembic upgrade head  # Uncomment when you have migrations

echo "✅ Local development environment is ready!"
echo ""
echo "🎯 Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Start the API server: uvicorn src.api.main:app --reload"
echo "3. Start workers: python -m src.workers.transcription_worker"
echo "4. Open http://localhost:8000/docs for API documentation"
echo "5. Open http://localhost:5601 for OpenSearch Dashboards"
echo ""
echo "🔧 Troubleshooting:"
echo "- If services fail to start, try: cd docker && docker-compose down && docker-compose up -d"
echo "- Check service status: cd docker && docker-compose ps"
echo "- View logs: cd docker && docker-compose logs [service-name]"