# System Architecture

## 🏗️ **Microservices Architecture Overview**

The Amazon Fashion Search Engine follows a **microservices architecture** with clear separation of concerns, enabling independent development, deployment, and scaling of each component.

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                          │
├─────────────────────────────────────────────────────────────────┤
│                     Frontend Service                           │
│                    (Next.js + React)                          │
└─────────────────────┬───────────────────────────────────────────┘
                      │ HTTP/REST API
┌─────────────────────▼───────────────────────────────────────────┐
│                    Search API Service                          │
│                     (FastAPI)                                 │
│  ┌─────────────────┬──────────────────┬─────────────────────┐  │
│  │   Search API    │   Vector Store   │   Health/Stats     │  │
│  │   Endpoints     │   (FAISS)        │   Monitoring       │  │
│  └─────────────────┴──────────────────┴─────────────────────┘  │
└─────────────────────┬───────────────────────────────────────────┘
                      │ File System / Shared Storage
┌─────────────────────▼───────────────────────────────────────────┐
│                  Data Pipeline Service                         │
│                     (Python)                                  │
│  ┌─────────────────┬──────────────────┬─────────────────────┐  │
│  │ Data Ingestion  │ Quality Sampling │ Embedding Generation│  │
│  │ & Validation    │ & Processing     │ (OpenAI API)        │  │
│  └─────────────────┴──────────────────┴─────────────────────┘  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                    External Services                           │
│  ┌─────────────────┬──────────────────┬─────────────────────┐  │
│  │   OpenAI API    │   Raw Dataset    │   Processed Data   │  │
│  │  (Embeddings)   │    (1.3GB)       │   & Embeddings     │  │
│  └─────────────────┴──────────────────┴─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 🎯 **Service Responsibilities**

### 1. **Data Pipeline Service** (`services/data-pipeline/`)
**Purpose**: Batch processing for data ingestion and preparation

**Responsibilities**:
- Raw dataset analysis and validation
- Quality-based stratified sampling
- Text preparation and feature extraction
- OpenAI embedding generation with batching
- FAISS index creation and optimization
- Data caching and persistence

**Technology Stack**:
- Python 3.11+
- pandas, numpy for data processing
- OpenAI API client for embeddings
- FAISS for vector index creation
- Pydantic for data validation

**Run Mode**: Batch job (one-time or scheduled)

### 2. **Search API Service** (`services/search-api/`)
**Purpose**: High-performance search API with sub-500ms response times

**Responsibilities**:
- REST API endpoints for search operations
- Vector similarity search with FAISS
- Query enhancement using LLM
- Result ranking and filtering
- Health monitoring and statistics
- API documentation and validation

**Technology Stack**:
- FastAPI for async API framework
- FAISS for vector similarity search
- OpenAI API for query enhancement
- Pydantic for request/response models
- uvicorn for ASGI server

**Run Mode**: Long-running service

### 3. **Frontend Service** (`services/frontend/`)
**Purpose**: Modern web interface for search and product discovery

**Responsibilities**:
- User interface for search and browsing
- Real-time search with debouncing
- Product visualization and details
- Responsive design for all devices
- Error handling and fallback modes

**Technology Stack**:
- Next.js 15 with App Router
- React 19 with TypeScript
- Tailwind CSS for styling
- Lucide React for icons

**Run Mode**: Development server / Static deployment

## 🔧 **Shared Components**

### **Shared Library** (`shared/`)
**Purpose**: Common code, models, and utilities across services

**Contains**:
- Pydantic data models (Product, SearchRequest, etc.)
- Configuration management
- Common utilities and helpers
- Data validation schemas
- API client interfaces

### **Infrastructure** (`infrastructure/`)
**Purpose**: Deployment, monitoring, and operational tools

**Contains**:
- Docker configurations
- Kubernetes manifests
- CI/CD pipeline configurations
- Monitoring and logging setup
- Environment configurations

## 📊 **Data Flow Architecture**

### **Data Processing Flow**
```
Raw Dataset (1.3GB) 
    ↓
Data Pipeline Service
    ├─ Dataset Analysis & Quality Assessment
    ├─ Stratified Sampling (50K products)
    ├─ Text Preparation & Feature Extraction
    ├─ Batch Embedding Generation (OpenAI API)
    ├─ FAISS Index Creation
    └─ Cache to Shared Storage
    ↓
Processed Data Available for Search API
```

### **Search Request Flow**
```
User Query (Frontend)
    ↓ HTTP Request
Search API Service
    ├─ Query Validation & Enhancement (OpenAI)
    ├─ Vector Similarity Search (FAISS)
    ├─ Business Logic Filtering
    ├─ Result Ranking & Pagination
    └─ Response Formatting
    ↓ HTTP Response
Frontend (Product Display)
```

## 🔗 **Service Communication**

### **Inter-Service Communication**
- **Frontend ↔ Search API**: HTTP/REST API calls
- **Search API ↔ Data Pipeline**: Shared file system / object storage
- **Services ↔ External APIs**: Direct HTTP calls (OpenAI)

### **Data Storage**
- **Raw Data**: File system (`data/raw/`)
- **Processed Data**: Parquet files (`data/processed/`)
- **Vector Embeddings**: NumPy arrays with metadata
- **FAISS Index**: Serialized index files
- **Configuration**: Environment variables / config files

## 🚀 **Deployment Strategies**

### **Development Mode**
```bash
# Start all services locally
./scripts/start-dev.sh

# Or individually:
cd services/data-pipeline && python pipeline.py
cd services/search-api && uvicorn main:app --reload
cd services/frontend && npm run dev
```

### **Production Mode**
- **Containerized**: Each service in separate Docker containers
- **Orchestrated**: Kubernetes for service management
- **Scaled**: Horizontal scaling based on load
- **Monitored**: Health checks, metrics, and logging

## 🔒 **Security & Configuration**

### **Environment Variables**
```bash
# Shared Configuration
OPENAI_API_KEY=your_openai_key
LOG_LEVEL=INFO

# Data Pipeline Service
DATA_PIPELINE_SAMPLE_SIZE=50000
DATA_PIPELINE_BATCH_SIZE=100

# Search API Service  
SEARCH_API_HOST=0.0.0.0
SEARCH_API_PORT=8000
SEARCH_API_WORKERS=4

# Frontend Service
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### **Security Considerations**
- API key management and rotation
- Rate limiting and request validation
- CORS configuration for frontend
- Health check endpoints for monitoring
- Error handling without data leakage

## 📈 **Scalability & Performance**

### **Horizontal Scaling**
- **Data Pipeline**: Can be parallelized for larger datasets
- **Search API**: Stateless service, easily horizontally scaled
- **Frontend**: Static deployment with CDN distribution

### **Performance Optimizations**
- **Caching**: Processed data and frequent queries
- **Indexing**: Optimized FAISS configurations
- **Batching**: Efficient OpenAI API usage
- **Async**: Non-blocking I/O operations

### **Resource Requirements**
- **Data Pipeline**: CPU/memory intensive (8GB+ RAM)
- **Search API**: Low latency requirements (SSD storage)
- **Frontend**: Minimal resources (static files)

## 🔄 **Development Workflow**

### **Service Independence**
- Each service can be developed independently
- Clear API contracts between services
- Independent testing and deployment
- Separate dependency management

### **Local Development**
- Mock data for faster development iterations
- Service-specific development scripts
- Hot-reload capabilities for all services
- Comprehensive logging and debugging

This architecture provides a solid foundation for production deployment while maintaining development simplicity and operational excellence. 