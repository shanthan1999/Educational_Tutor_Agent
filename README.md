# Educational Tutor Agent API

A production-ready AI-powered educational assistant with science knowledge base and web search capabilities.

## 🚀 Features

- **Hybrid Knowledge System**: Local ScienceQA dataset + Web search fallback
- **Production-Ready API**: FastAPI with comprehensive error handling
- **Performance Optimized**: Caching, rate limiting, and monitoring
- **Containerized Deployment**: Docker and Docker Compose support
- **Health Monitoring**: Built-in health checks and metrics
- **Comprehensive Testing**: Full test suite included

## 📋 Prerequisites

- Python 3.10+
- Docker (optional)
- Exa API key (optional, for web search)

## 🛠️ Installation

### Option 1: Direct Installation

```bash
# Clone the repository
git clone <repository-url>
cd educational-tutor-agent

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env
# Edit .env with your settings

# Run the application
python api.py
```

### Option 2: Docker Deployment

```bash
# Copy environment configuration
cp .env.example .env
# Edit .env with your settings

# Build and run with Docker Compose
docker-compose up -d

# Or build and run manually
docker build -t tutor-agent .
docker run -p 8000:8000 --env-file .env tutor-agent
```

## ⚙️ Configuration

Key environment variables:

```bash
# API Configuration
EXA_API_KEY=your_exa_api_key_here          # Optional: For web search

# Performance Settings
FORCE_CPU=false                            # Set to true to force CPU usage
MAX_CONCURRENT_REQUESTS=10                 # Max concurrent requests
REQUEST_TIMEOUT=30                         # Request timeout in seconds

# Caching Settings
ENABLE_CACHING=true                        # Enable response caching
CACHE_TTL=3600                            # Cache TTL in seconds

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60                  # Requests per minute per IP

# Logging
LOG_LEVEL=INFO                            # Logging level
LOG_FILE=logs/app.log                     # Log file path
```

## 🔧 API Usage

### Base URL
```
http://localhost:8000
```

### Endpoints

#### Query Educational Content
```http
POST /query
Content-Type: application/json

{
  "question": "What is photosynthesis?",
  "include_sources": true,
  "use_web_search": true
}
```

**Response:**
```json
{
  "answer": "Photosynthesis is the process by which plants...",
  "sources": [
    {
      "content": "Process plants use to make food...",
      "metadata": {
        "subject": "Biology",
        "source": "ScienceQA"
      }
    }
  ],
  "confidence": null,
  "response_time_ms": 1250,
  "used_web_search": false,
  "cached": false
}
```

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "uptime_seconds": 3600,
  "total_requests": 150,
  "error_rate_percent": 2.1,
  "cache_stats": {
    "total_entries": 45,
    "expired_entries": 3
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

#### API Statistics
```http
GET /stats
```

### Python Client Example

```python
import requests

# Ask a question
response = requests.post(
    "http://localhost:8000/query",
    json={
        "question": "What causes climate change?",
        "include_sources": True
    }
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Response time: {result['response_time_ms']}ms")
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "What is DNA?",
       "include_sources": true
     }'
```

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_api.py::TestQueryEndpoint -v
pytest tests/test_api.py::TestPerformance -v
```

## 📊 Monitoring

### Built-in Monitoring

- **Health Endpoint**: `/health` - System health status
- **Stats Endpoint**: `/stats` - Detailed API statistics
- **Structured Logging**: Comprehensive request/response logging

### Optional External Monitoring

The Docker Compose setup includes optional Prometheus and Grafana:

```bash
# Start with monitoring
docker-compose up -d

# Access monitoring dashboards
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

## 🔒 Security Considerations

### For Production Deployment:

1. **Environment Variables**: Never commit `.env` files
2. **API Keys**: Use secure key management systems
3. **CORS**: Configure allowed origins properly
4. **Rate Limiting**: Adjust limits based on your needs
5. **HTTPS**: Use reverse proxy with SSL/TLS
6. **Network Security**: Implement proper firewall rules

### Example Nginx Configuration:

```nginx
server {
    listen 443 ssl;
    server_name yourdomain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 🚀 Deployment Options

### Option 1: Cloud Platforms

**Google Cloud Run:**
```bash
gcloud run deploy tutor-agent \
  --image gcr.io/PROJECT-ID/tutor-agent \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

**AWS ECS/Fargate:**
```bash
aws ecs create-service \
  --cluster tutor-cluster \
  --service-name tutor-agent \
  --task-definition tutor-agent:1 \
  --desired-count 2
```

### Option 2: Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tutor-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tutor-agent
  template:
    metadata:
      labels:
        app: tutor-agent
    spec:
      containers:
      - name: tutor-agent
        image: tutor-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: EXA_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: exa-api-key
```

## 📈 Performance Optimization

### Recommendations:

1. **Hardware**: Use GPU for better performance
2. **Caching**: Enable Redis for distributed caching
3. **Load Balancing**: Use multiple instances behind a load balancer
4. **Model Optimization**: Consider model quantization for faster inference
5. **Database**: Use persistent storage for vector embeddings

### Scaling Configuration:

```yaml
# docker-compose.override.yml
version: '3.8'
services:
  tutor-agent:
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
    environment:
      - MAX_CONCURRENT_REQUESTS=20
```

## 🐛 Troubleshooting

### Common Issues:

1. **Out of Memory**: Reduce `MAX_DATASET_SIZE` or enable `FORCE_CPU=true`
2. **Slow Responses**: Enable caching and check hardware resources
3. **Web Search Not Working**: Verify `EXA_API_KEY` is set correctly
4. **High Error Rate**: Check logs and monitor health endpoint

### Debug Mode:

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python api.py
```

### Check System Status:

```bash
# Check health
curl http://localhost:8000/health

# Check detailed stats
curl http://localhost:8000/stats
```

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:

1. Check the troubleshooting section
2. Review the logs
3. Create an issue with detailed information
4. Include system specs and error messages 