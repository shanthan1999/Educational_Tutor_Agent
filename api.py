"""
Production-ready FastAPI interface for Educational Tutor Agent
"""

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
import time
import asyncio
from contextlib import asynccontextmanager
import logging

from config import config
from utils import (
    setup_production_logging, 
    health_checker, 
    cache,
    validate_query,
    monitor_performance,
    retry_with_backoff
)
from tutor_agent import setup_educational_agent

# Setup logging
logger = setup_production_logging(config.LOG_LEVEL, config.LOG_FILE)

# Rate limiting storage
rate_limit_storage = {}

# Pydantic models
class QueryRequest(BaseModel):
    """Request model for queries."""
    question: str = Field(..., min_length=3, max_length=1000, description="The question to ask")
    include_sources: bool = Field(default=True, description="Whether to include source information")
    use_web_search: bool = Field(default=True, description="Whether to use web search if needed")
    
    @validator('question')
    def validate_question(cls, v):
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()

class QueryResponse(BaseModel):
    """Response model for queries."""
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None
    confidence: Optional[float] = None
    response_time_ms: int
    used_web_search: bool = False
    cached: bool = False

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    uptime_seconds: int
    total_requests: int
    error_rate_percent: float
    cache_stats: Dict[str, Any]
    timestamp: str

# Rate limiting decorator
def rate_limit(max_requests: int = None):
    """Rate limiting decorator."""
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            client_ip = request.client.host
            current_time = time.time()
            max_reqs = max_requests or config.RATE_LIMIT_PER_MINUTE
            
            # Clean old entries
            if client_ip in rate_limit_storage:
                rate_limit_storage[client_ip] = [
                    req_time for req_time in rate_limit_storage[client_ip]
                    if current_time - req_time < 60  # 1 minute window
                ]
            else:
                rate_limit_storage[client_ip] = []
            
            # Check rate limit
            if len(rate_limit_storage[client_ip]) >= max_reqs:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Maximum {max_reqs} requests per minute."
                )
            
            # Record request
            rate_limit_storage[client_ip].append(current_time)
            
            return await func(request, *args, **kwargs)
        
        return wrapper
    return decorator

# Global agent instance
educational_agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("🚀 Starting Educational Tutor Agent API...")
    
    global educational_agent
    try:
        educational_agent = setup_educational_agent()
        logger.info("✅ Educational agent initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize educational agent: {e}")
        raise
    
    # Background task to clean cache
    async def cleanup_cache():
        while True:
            try:
                expired_count = cache.clear_expired()
                if expired_count > 0:
                    logger.info(f"🧹 Cleaned {expired_count} expired cache entries")
                await asyncio.sleep(300)  # Clean every 5 minutes
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(300)
    
    # Start background tasks
    cleanup_task = asyncio.create_task(cleanup_cache())
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Educational Tutor Agent API...")
    cleanup_task.cancel()

# Create FastAPI app
app = FastAPI(
    title="Educational Tutor Agent API",
    description="AI-powered educational assistant with science knowledge base and web search capabilities",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Middleware for request logging and monitoring
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log requests and monitor performance."""
    start_time = time.time()
    
    # Record request
    health_checker.record_request()
    
    try:
        response = await call_next(request)
        success = True
    except Exception as e:
        health_checker.record_error()
        success = False
        logger.error(f"Request failed: {e}")
        raise
    finally:
        process_time = time.time() - start_time
        logger.info(
            f"Request: {request.method} {request.url.path} - "
            f"Time: {process_time:.3f}s - "
            f"Success: {success}"
        )
    
    return response

# API Routes
@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Educational Tutor Agent API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "query": "/query",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        health_status = health_checker.get_health_status()
        return HealthResponse(**health_status)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/query", response_model=QueryResponse)
@rate_limit()
async def query_agent(request: Request, query_req: QueryRequest, background_tasks: BackgroundTasks):
    """Main query endpoint for the educational agent."""
    
    if not educational_agent:
        raise HTTPException(status_code=503, detail="Educational agent not available")
    
    start_time = time.time()
    
    try:
        # Validate input
        validation_result = validate_query(query_req.question)
        if validation_result is not True:
            raise HTTPException(status_code=400, detail=validation_result)
        
        # Check cache first
        cache_key = f"query:{hash(query_req.question)}"
        cached_result = cache.get(cache_key) if config.ENABLE_CACHING else None
        
        if cached_result:
            logger.info(f"🎯 Cache hit for query: {query_req.question[:50]}...")
            response_time = int((time.time() - start_time) * 1000)
            
            return QueryResponse(
                answer=cached_result['answer'],
                sources=cached_result.get('sources'),
                confidence=cached_result.get('confidence'),
                response_time_ms=response_time,
                used_web_search=cached_result.get('used_web_search', False),
                cached=True
            )
        
        # Process query with the agent
        logger.info(f"🤔 Processing query: {query_req.question[:50]}...")
        
        # Call the agent
        result = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: educational_agent({"question": query_req.question})
        )
        
        # Extract information
        answer = result.get('answer', 'I apologize, but I could not generate an answer.')
        sources = []
        
        if query_req.include_sources and 'source_documents' in result:
            sources = [
                {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result['source_documents'][:3]
            ]
        
        # Calculate response time
        response_time = int((time.time() - start_time) * 1000)
        
        # Prepare response
        response_data = {
            'answer': answer,
            'sources': sources if query_req.include_sources else None,
            'confidence': None,  # Could implement confidence scoring
            'response_time_ms': response_time,
            'used_web_search': False,  # Could detect this from the agent
            'cached': False
        }
        
        # Cache the result
        if config.ENABLE_CACHING:
            background_tasks.add_task(
                cache.set, 
                cache_key, 
                response_data, 
                config.CACHE_TTL
            )
        
        logger.info(f"✅ Query processed successfully in {response_time}ms")
        
        return QueryResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Query processing failed: {e}")
        health_checker.record_error()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/stats", response_model=dict)
async def get_stats():
    """Get API statistics."""
    try:
        health_status = health_checker.get_health_status()
        return {
            "api_stats": health_status,
            "cache_stats": cache.stats(),
            "rate_limit_stats": {
                "active_clients": len(rate_limit_storage),
                "total_tracked_requests": sum(len(reqs) for reqs in rate_limit_storage.values())
            }
        }
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Stats retrieval failed")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "timestamp": time.time()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    health_checker.record_error()
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": time.time()}
    )

if __name__ == "__main__":
    import uvicorn
    
    # Run the API
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to False in production
        log_level="info",
        workers=1  # Single worker for shared memory
    ) 