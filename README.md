# VidRag

# VideoRAG - Advanced Video Retrieval Augmented Generation

> **Note**: This is a simplified version of the VideoRAG system developed during my collaboration with Bons Company. This public version is shared with the full consent and approval of Bons Company for educational and demonstration purposes.

A sophisticated video processing and retrieval system that combines multimodal AI techniques with distributed event-driven architecture for intelligent video content analysis and search.

## 🚀 Overview

VideoRAG is an enterprise-grade system that transforms video content into searchable, queryable knowledge through advanced AI processing. It provides hierarchical video segmentation, multimodal embeddings, knowledge graph extraction, and intelligent retrieval capabilities.

### Key Features

- **🎥 Advanced Video Processing**: Scene detection, hierarchical segmentation, and multimodal content extraction
- **🧠 AI-Powered Analysis**: Whisper transcription, CLIP visual encoding, OCR text recognition, and spaCy NLP
- **📊 Knowledge Graph Construction**: Automatic entity extraction and relationship mapping from video content
- **🔍 Intelligent Search**: Hierarchical vector search with adaptive modality weighting and contextual ranking
- **⚡ Event-Driven Architecture**: SQS-based distributed processing with auto-scaling workers
- **🔒 Enterprise Ready**: Supabase integration with Row-Level Security, real-time updates, and comprehensive monitoring

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI API   │────│  Supabase DB    │────│  Redis Cache    │
│   Server        │    │  + Storage      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │──────────────│   AWS SQS       │──────────────│
         │              │   Event Bus     │              │
         │              └─────────────────┘              │
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Transcription  │    │ Scene Detection │    │   Embedding     │
│    Workers      │    │    Workers      │    │   Workers       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Processing Pipeline

1. **Video Upload** → Supabase Storage + PostgreSQL metadata
2. **Event Publishing** → SQS queues trigger distributed workers
3. **Transcription** → Whisper model extracts speech with timestamps
4. **Scene Detection** → CLIP-based visual similarity analysis
5. **Hierarchical Segmentation** → Semantic grouping of content
6. **Multimodal Embedding** → Text, visual, and OCR embeddings
7. **Knowledge Graph** → Entity extraction and relationship mapping
8. **Vector Indexing** → FAISS indices for fast similarity search

## 🛠️ Technology Stack

### Core AI Models
- **Whisper (OpenAI)**: Speech-to-text transcription with word-level timestamps
- **CLIP (OpenAI)**: Visual feature extraction and scene boundary detection
- **SentenceTransformers**: Text embedding generation for semantic search
- **EasyOCR**: Optical character recognition for text-in-video extraction
- **spaCy**: Named entity recognition and relationship extraction

### Infrastructure
- **FastAPI**: High-performance async API framework
- **Supabase**: PostgreSQL database with real-time capabilities and object storage
- **AWS SQS**: Message queues for distributed event processing
- **Redis**: Caching layer with category-based TTL management
- **FAISS**: Vector similarity search with multiple indices
- **Docker**: Containerized deployment and scaling

### Database Schema
- **PostgreSQL + pgvector**: Vector storage and similarity search
- **Row-Level Security**: Multi-tenant data isolation
- **Real-time subscriptions**: Live processing status updates

## 📋 Prerequisites

- Python 3.8+
- FFmpeg (for video/audio processing)
- PostgreSQL with pgvector extension
- Redis server
- AWS account (for SQS)
- Supabase project

## 🔧 Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/videorag.git
cd videorag
```

### 2. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file with your credentials:

```env
# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
DATABASE_URL=postgresql://postgres:your-password@db.your-project.supabase.co:5432/postgres

# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# Optional Services
REDIS_URL=redis://localhost:6379
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true
```

### 4. Database Setup
```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create tables (handled automatically by the application)
-- See src/core/database/connection.py for schema definitions
```

### 5. SQS Queue Setup
```bash
# The application automatically creates required SQS queues:
# - videorag-video-uploaded
# - videorag-transcription-completed
# - videorag-scene-detection-completed
# - videorag-segmentation-completed
# - videorag-embeddings-generated
# - videorag-indexing-completed
# - videorag-processing-completed
# - videorag-processing-failed-dlq
```

## 🚀 Usage

### Start the API Server
```bash
python -m src.api.main
```

### Start Workers (in separate terminals)
```bash
# Transcription worker
python -m src.workers.sqs_transcription_worker

# Additional workers can be started as needed
# python -m src.workers.sqs_scene_detection_worker
# python -m src.workers.sqs_embedding_worker
```

### API Endpoints

#### Video Management
```bash
# Upload video
curl -X POST "http://localhost:8000/api/v1/videos/upload" \
  -F "file=@video.mp4" \
  -F "is_public=true"

# Get video status
curl "http://localhost:8000/api/v1/videos/{video_id}"

# List videos
curl "http://localhost:8000/api/v1/videos/"
```

#### Search & Query
```bash
# Basic search
curl -X POST "http://localhost:8000/api/v1/queries/search" \
  -H "Content-Type: application/json" \
  -d '{"text": "machine learning concepts", "limit": 10}'

# Hierarchical search
curl -X POST "http://localhost:8000/api/v1/queries/search/hierarchical" \
  -H "Content-Type: application/json" \
  -d '{"text": "AI explanation", "search_level": "both", "limit": 15}'

# Multimodal search with custom weights
curl -X POST "http://localhost:8000/api/v1/queries/search/multimodal" \
  -H "Content-Type: application/json" \
  -d '{"query": "data visualization", "modality_weights": {"text": 0.5, "visual": 0.3, "ocr": 0.2}}'
```

## 🔍 Core Features Deep Dive

### 1. Hierarchical Video Processing

The system processes videos at multiple granular levels:

- **Scene Level**: Detected using CLIP visual similarity analysis
- **Segment Level**: Semantically coherent chunks within scenes
- **Transcript Level**: Individual speech segments with word-level timing

```python
# Example hierarchical structure
{
  "video_id": "uuid",
  "scenes": [
    {
      "scene_id": "scene_0",
      "start": 0.0,
      "end": 45.2,
      "segments": [
        {
          "segment_id": "scene_0_0",
          "start": 0.0,
          "end": 15.3,
          "text": "Welcome to our machine learning tutorial...",
          "visual_features": [...],
          "ocr_results": [...]
        }
      ]
    }
  ]
}
```

### 2. Multimodal Embeddings

Each video segment generates embeddings across three modalities:

- **Text Embeddings**: SentenceTransformers for semantic content
- **Visual Embeddings**: CLIP features for visual content
- **OCR Embeddings**: Text-in-video content embeddings

### 3. Knowledge Graph Construction

Automatic extraction of:
- **Entities**: People, organizations, concepts (spaCy NER)
- **Relationships**: Co-occurrence and semantic relationships
- **Topics**: Theme identification across video content
- **Temporal Mapping**: Entity presence over time

```python
# Example knowledge graph
{
  "entities": {
    "PERSON": ["John Doe", "Jane Smith"],
    "ORG": ["OpenAI", "Google"],
    "CONCEPT": ["machine learning", "neural networks"]
  },
  "relationships": [
    {
      "subject": "John Doe",
      "predicate": "discusses",
      "object": "machine learning",
      "timestamp": 125.4
    }
  ]
}
```

### 4. Adaptive Search Ranking

The search system employs multiple ranking strategies:

- **Modality Weighting**: Automatic adjustment based on query type
- **Temporal Coherence**: Bonus for adjacent relevant segments
- **Contextual Scoring**: Scene-level context consideration
- **Semantic Clustering**: Related content grouping

### 5. Event-Driven Processing

SQS-based architecture enables:
- **Horizontal Scaling**: Auto-scaling worker instances
- **Fault Tolerance**: Dead letter queues and retry logic
- **Progress Tracking**: Real-time processing status updates
- **Load Distribution**: Intelligent task routing

## 📊 Monitoring & Analytics

### Health Checks
```bash
curl "http://localhost:8000/health"
```

### System Statistics
```bash
curl "http://localhost:8000/api/v1/stats"
```

### Worker Status
```bash
curl "http://localhost:8000/api/v1/workers/status"
```

## 🔒 Security Features

- **Row-Level Security**: Multi-tenant data isolation
- **JWT Authentication**: Token-based API access
- **CORS Configuration**: Cross-origin request management
- **Input Validation**: Comprehensive request sanitization
- **Rate Limiting**: API abuse prevention

## 🚀 Deployment

### Docker Deployment
```bash
# Build image
docker build -t videorag:latest .

# Run with docker-compose
docker-compose up -d
```

### AWS ECS/EKS
The system is designed for cloud-native deployment with:
- Auto-scaling worker services
- Load-balanced API endpoints
- Managed database and caching services

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run integration tests
pytest tests/integration/

# Load testing
pytest tests/load/
```

## 📈 Performance

### Benchmarks
- **Video Processing**: ~2-5x real-time (depending on complexity)
- **Search Latency**: <100ms for typical queries
- **Throughput**: 1000+ concurrent requests
- **Storage Efficiency**: ~10-20MB metadata per hour of video

### Optimization Features
- **Caching**: Multi-layer caching with Redis
- **Indexing**: FAISS indices for sub-linear search
- **Batching**: Efficient batch processing for embeddings
- **Compression**: Optimized vector storage

## 🛣️ Roadmap

- [ ] **Advanced NLP**: Integration with larger language models
- [ ] **Real-time Processing**: Live video stream processing
- [ ] **Advanced Analytics**: Video content insights and trends
- [ ] **Mobile SDK**: Client libraries for mobile applications
- [ ] **Federated Search**: Cross-video collection querying

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Bons Company** for the collaboration opportunity and consent to share this simplified version
- **OpenAI** for Whisper and CLIP models
- **Hugging Face** for SentenceTransformers
- **Supabase** for the excellent PostgreSQL platform
- **FastAPI** for the high-performance web framework

## ⚖️ Legal Notice

This project represents a simplified version of work conducted in collaboration with Bons Company. The original, full-featured system remains proprietary to Bons Company. This public version is shared with explicit consent for educational, portfolio, and open-source community purposes.

**Important**: This simplified version may not include all production-level features, optimizations, or proprietary algorithms present in the original system developed for Bons Company.

## 📞 Support

For questions and support:
- 📧 Email: support@videorag.com
- 💬 Discord: [VideoRAG Community](https://discord.gg/videorag)
- 📖 Documentation: [docs.videorag.com](https://docs.videorag.com)

---

**VideoRAG** - Transforming video content into intelligent, searchable knowledge.
