# Amazon Fashion Search Engine

> **Semantic fashion product search using LLMs and vector similarity**

A production-ready microservice that combines OpenAI embeddings with intelligent query processing to deliver relevant fashion product search results. Built with FastAPI, Next.js, and Docker for scalable deployment.

## 🚀 **Quick Start (5 minutes)**

### **Option 1: Docker Compose (Recommended)**
```bash
# 1. Clone and navigate
git clone <repository-url>
cd amazon-fashion-search-engine

# 2. Configure environment
cp .env.template .env
# Edit .env and add your OPENAI_API_KEY

# 3. Start the application
docker-compose up

# 4. Access the application
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs
```

### **Option 2: Manual Setup**
```bash
# Method A: Using setup.py (Recommended)
pip install -e .                    # Install with dependencies
cp .env.template .env                # Copy environment template
# Edit .env and add your OPENAI_API_KEY

# Method B: Using setup script  
chmod +x scripts/setup.sh
./scripts/setup.sh

# Start services
python services/search-api/main.py &
cd frontend && npm run dev

# Visit http://localhost:3000
```

### **Option 3: Development Setup**
```bash
# Install with development dependencies
pip install -e ".[dev]"             # Includes testing and linting tools
cp .env.template .env
# Edit .env and add your OPENAI_API_KEY

# Install frontend dependencies
cd frontend && npm install

# Verify installation
fashion-search-pipeline --validate-only    # Test CLI commands
python -c "from services.search_api.src.search import SearchEngine; print('✅ Modules imported successfully')"

# Start development servers  
python services/search-api/main.py --reload &
cd frontend && npm run dev
```

---

## 🏗️ **Architecture Overview**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Search API    │    │ Data Pipeline   │
│   (Next.js)     │◄───┤   (FastAPI)     │    │   (Python)      │
│   Port: 3000    │    │   Port: 8000    │    │   (CLI)         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐             │
         └──────────────►│ Vector Database │◄────────────┘
                        │ (FAISS + Data)  │
                        └─────────────────┘
```

### **Key Components:**
- **🔍 Search API**: FastAPI service with semantic search and LLM-powered filtering
- **⚛️ Frontend**: Next.js application with modern UI and search interface  
- **📊 Data Pipeline**: Python service for data processing and embedding generation
- **🗄️ Vector Database**: FAISS index with consolidated product embeddings

---

## 🔧 **Data Processing Options**

### **Default: Use Preloaded Data (Fastest)**
```bash
# Just start the services - data is already processed
python services/data-pipeline/main.py  # Validates existing data
python services/search-api/main.py     # Starts API with preloaded data
```

### **Rebuild with Sample Data**
```bash
# Quick rebuild with 50k products (~30-60 min)
python services/data-pipeline/main.py --rebuild

# Custom sample size
python services/data-pipeline/main.py --rebuild --sample-size 1000  # 2-5 min
```

### **Full Dataset Processing**
```bash
# Process complete dataset (~800k products, 2-4 hours)
python services/data-pipeline/main.py --rebuild --full
```

---

## 🧪 **Sample Queries**

Try these queries to see the semantic search capabilities:

### **Natural Language Queries**
```
"comfortable summer dresses under $50"
"elegant wedding guest outfit" 
"professional work attire for women"
"casual weekend clothing"
"formal business suits"
```

### **Style & Occasion Based**
```
"vintage style leather jackets"
"bohemian summer accessories"  
"minimalist office wear"
"athletic wear for running"
"cozy winter sweaters"
```

### **Advanced Filtering**
```
"red evening gowns under $100"
"designer handbags with good ratings"
"waterproof hiking boots size 9"
"organic cotton t-shirts"
"sustainable fashion brands"
```

---

## 💻 **API Usage**

### **Search Endpoint**
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "comfortable summer dresses",
    "top_k": 10,
    "price_max": 50.0,
    "min_rating": 4.0
  }'
```

### **Health Check**
```bash
curl http://localhost:8000/health
```

### **API Documentation**
Visit `http://localhost:8000/docs` for interactive API documentation.

---

## 🔧 **CLI Commands**

After installing with `pip install -e .`, you get convenient CLI commands:

### **Data Pipeline Commands**
```bash
# Using CLI command (after pip install)
fashion-search-pipeline --rebuild                    # Rebuild with 50k sample
fashion-search-pipeline --rebuild --sample-size 1000 # Custom sample size
fashion-search-pipeline --rebuild --full             # Process full dataset
fashion-search-pipeline --validate-only              # Just validate data

# Using Python module directly
python services/data-pipeline/main.py --rebuild
python -m services.data_pipeline.main --rebuild
```

### **Search API Commands**
```bash
# Using CLI command (after pip install)
fashion-search-api                                   # Start API server
fashion-search-api --reload                          # Development mode
fashion-search-api --host 0.0.0.0 --port 8080       # Custom host/port

# Using Python module directly  
python services/search-api/main.py
python -m services.search_api.main --reload
```

---

## 🐳 **Docker Deployment Options**

### **Quick Start (Default)**
```bash
docker-compose up
# Starts: Search API + Frontend with preloaded data
```

### **Rebuild Data First**
```bash
docker-compose --profile rebuild up
# Rebuilds data, then starts all services
```

### **Development Mode**
```bash
docker-compose --profile dev up
# Auto-reload enabled for development
```

### **API Only**
```bash
docker-compose up search-api
# Just the search API service
```

---

## ⚙️ **Configuration**

### **Environment Variables**

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *Required* | OpenAI API key for embeddings |
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8000` | API server port |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Frontend API URL |
| `STRATIFIED_SAMPLE_SIZE` | `50000` | Sample size for data rebuild |
| `LOG_LEVEL` | `INFO` | Logging level |

### **Sample Sizes & Performance**

| Sample Size | Processing Time | Use Case |
|-------------|-----------------|----------|
| 1,000 | 2-5 minutes | Quick testing |
| 10,000 | 10-20 minutes | Development |
| 50,000 | 30-60 minutes | **Default (Recommended)** |
| Full (~800k) | 2-4 hours | Production |

---

## 🎯 **Key Features**

### **🔍 Semantic Search**
- **OpenAI Embeddings**: Uses `text-embedding-3-small` for semantic understanding
- **Vector Similarity**: FAISS index for efficient similarity search
- **Query Enhancement**: LLM-powered query expansion and context

### **🤖 Intelligent Filtering**
- **LLM Filter Extraction**: Automatically extracts filters from natural language
- **Graceful Degradation**: Falls back when filters yield too few results
- **Unified Schema**: Standardized filter values for consistent matching

### **📊 Hybrid Ranking**
- **Semantic Similarity**: Primary ranking by embedding similarity
- **Business Signals**: Rating, popularity, and completeness boosting
- **Reranking Options**: Heuristic (fast) or LLM-powered (better quality)

### **🏭 Production Ready**
- **Microservice Architecture**: Independent, scalable services
- **Docker Support**: One-click deployment with Docker Compose
- **Health Checks**: Comprehensive monitoring and status endpoints
- **Error Handling**: Graceful error handling and user feedback

---

## 🔍 **System Monitoring**

### **Health Endpoints**
```bash
# API Health
curl http://localhost:8000/health

# Frontend Health  
curl http://localhost:3000/api/health

# API Statistics
curl http://localhost:8000/stats
```

### **Log Monitoring**
```bash
# View API logs
docker-compose logs search-api

# View frontend logs
docker-compose logs frontend

# Follow logs in real-time
docker-compose logs -f
```

---

## 🛠️ **Development**

### **Local Development Setup**
```bash
# 1. Install dependencies
uv install                    # Python dependencies
cd frontend && npm install    # Frontend dependencies

# 2. Set up environment
cp .env.template .env
# Edit .env and add your OPENAI_API_KEY

# 3. Start development servers
python services/search-api/main.py --reload &
cd frontend && npm run dev

# 4. Access development servers
# Frontend: http://localhost:3000
# API: http://localhost:8000
```

### **Working with Modular Components**

The search engine uses a modular architecture where each component has a focused responsibility:

```python
# SearchEngine orchestrates all components
from services.search_api.src.search import SearchEngine

search_engine = SearchEngine(settings)

# Individual components can be used independently
from services.search_api.src.search import VectorSearchManager, LLMProcessor

vector_search = VectorSearchManager(settings)
llm_processor = LLMProcessor(settings)

# Test components separately
similarities, indices = vector_search.search(query_embedding, k=10)
enhanced_query, filters = await llm_processor.process_search_query("blue dress")
```

### **Component Development**

Each component can be developed and tested independently:

```bash
# Test individual components
python -c "
from services.search_api.src.search.vector_search import VectorSearchManager
from shared.models import Settings
import numpy as np

settings = Settings()
vs = VectorSearchManager(settings)
print('VectorSearchManager imported successfully')
"

# Test data pipeline components  
python -c "
from services.data_pipeline.src.processors.data_loader import DataLoader
from shared.models import Settings

settings = Settings()
loader = DataLoader(settings)
print('DataLoader imported successfully')
"
```

### **Code Structure**
```
amazon-fashion-search-engine/
├── services/
│   ├── data-pipeline/          # Data processing service
│   │   ├── main.py            # Enhanced CLI interface
│   │   └── src/
│   │       ├── pipeline.py    # Main pipeline orchestrator
│   │       ├── data_processor.py     # Data processing logic
│   │       ├── embedding_generator.py # Embedding generation
│   │       └── processors/    # Modular processing components
│   │           ├── data_loader.py    # Data loading and source detection
│   │           └── __init__.py
│   └── search-api/            # Search API service
│       ├── main.py            # Enhanced CLI interface
│       └── src/
│           ├── api.py         # FastAPI application
│           └── search/        # Modular search components
│               ├── engine.py          # Main search orchestrator
│               ├── vector_search.py   # FAISS operations
│               ├── llm_integration.py # OpenAI API integration
│               ├── filtering.py       # Product filtering logic
│               ├── ranking.py         # Ranking algorithms
│               └── __init__.py
├── frontend/                  # Next.js application
│   ├── app/                   # App router pages
│   ├── components/            # React components
│   └── services/              # API integration
├── shared/                    # Shared utilities
│   ├── models/                # Pydantic models
│   └── utils/                 # Common functions
└── data/                      # Data storage
    ├── processed/             # Processed products
    └── embeddings/            # Vector embeddings
```

---

## 💡 **Design Decisions**

### **Why These Technologies?**
- **FastAPI**: Modern, fast, with automatic API documentation
- **Next.js**: React framework with SSR and optimized performance
- **FAISS**: Facebook's vector database - fast and scalable
- **OpenAI Embeddings**: State-of-the-art semantic understanding
- **Docker**: Consistent deployment across environments

### **Data Strategy**
- **Stratified Sampling**: Ensures diverse, representative product mix
- **Consolidated Storage**: Single FAISS index for optimal performance
- **Preloaded Data**: 50k products ready for immediate testing
- **Scalable Processing**: Batch processing with rate limit handling

### **Modular Architecture**
- **Component-Based Design**: Each module has a single responsibility
- **SearchEngine**: Orchestrates VectorSearchManager, LLMProcessor, FilterManager, RankingManager
- **DataPipeline**: Orchestrates DataLoader, DataProcessor, EmbeddingGenerator
- **Independent Testing**: Each component can be tested and optimized separately
- **Graceful Degradation**: Component failures don't break the entire system

### **Search Quality**
- **Multi-stage Pipeline**: Vector search → Filtering → Ranking
- **LLM Integration**: Query enhancement and filter extraction
- **Dual Ranking**: Heuristic (fast) and LLM-based (high quality) ranking options
- **Graceful Degradation**: Maintains results when filters are too restrictive
- **Business Logic**: Incorporates ratings, popularity, and completeness

---

## 🚨 **Troubleshooting**

### **Common Issues**

**Missing OpenAI API Key**
```bash
# Error: Please set a valid OPENAI_API_KEY
cp .env.template .env
# Edit .env and add your API key
```

**Data Files Missing**
```bash
# Error: Processed data not found
fashion-search-pipeline --rebuild
# OR
python services/data-pipeline/main.py --rebuild
```

**Module Import Errors**
```bash
# Error: ModuleNotFoundError: No module named 'services'
pip install -e .                     # Install in development mode
# OR
pip install -e ".[dev]"              # Install with dev dependencies
```

**CLI Commands Not Found**
```bash
# Error: fashion-search-api: command not found
pip install -e .                     # Reinstall to get CLI commands
# OR use Python module directly
python services/search-api/main.py
```

**Port Already in Use**
```bash
# Change ports in .env file
API_PORT=8001
FRONTEND_PORT=3001
```

**Docker Build Issues**
```bash
# Clean rebuild
docker-compose down
docker system prune -a
docker-compose up --build
```

### **Performance Issues**

**Slow Search Response**
- Check if FAISS index is built (logs will show "Building FAISS index...")
- Consider smaller sample size for faster startup
- Ensure sufficient RAM (4GB+ recommended)

**Rate Limiting**
- Use `--sequential` flag for data pipeline
- Reduce `LLM_CONCURRENT_LIMIT` in environment

---

## 📈 **Performance Metrics**

### **Expected Performance**
- **Search Latency**: 200-800ms (including LLM processing)
- **Throughput**: 50-100 requests/second (single instance)
- **Memory Usage**: 2-4GB (depends on dataset size)
- **Startup Time**: 30-60 seconds (with preloaded data)

### **Optimization Options**
- **Heuristic Ranking**: 2-3x faster than LLM reranking
- **Smaller Samples**: Faster startup and lower memory usage
- **Batch Processing**: Efficient for data pipeline operations

---

## 🤝 **Contributing & Next Steps**

### **Potential Enhancements**
- **Advanced Filters**: Size, color, material detection
- **Personalization**: User preference learning
- **A/B Testing**: Ranking algorithm comparison
- **Caching**: Redis integration for faster responses
- **Monitoring**: Prometheus metrics and Grafana dashboards

### **Scalability Considerations**
- **Horizontal Scaling**: Multiple API instances behind load balancer
- **Database Scaling**: Sharded FAISS indices for large datasets
- **CDN Integration**: Image and static asset optimization
- **Async Processing**: Background task queue for heavy operations

---

## 📄 **License**

This project is for educational and demonstration purposes.

---

## 🆘 **Support**

For questions about this implementation:
1. Check the **Troubleshooting** section above
2. Review API documentation at `http://localhost:8000/docs`
3. Check service health at `http://localhost:8000/health`
4. Review logs with `docker-compose logs`

**Happy searching! 🔍✨**
