"""
Production-ready configuration management for Educational Tutor Agent
"""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ProductionConfig:
    """Production configuration with validation."""
    
    # Model settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL: str = "google/flan-t5-base"
    
    # Processing settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RETRIEVAL: int = 4
    MAX_DATASET_SIZE: int = 1000  # Increased for production
    
    # Generation settings
    MAX_NEW_TOKENS: int = 512  # Increased for detailed answers
    MIN_LENGTH: int = 50
    TEMPERATURE: float = 0.1
    DO_SAMPLE: bool = True
    NUM_BEAMS: int = 2
    
    # API settings
    TAVILY_API_KEY: Optional[str] = os.getenv("TAVILY_API_KEY")
    
    # Performance settings
    FORCE_CPU: bool = os.getenv("FORCE_CPU", "false").lower() == "true"
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    
    # Caching settings
    ENABLE_CACHING: bool = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
    
    # Logging settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE", "logs/app.log")
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        errors = []
        
        if self.CHUNK_SIZE <= 0:
            errors.append("CHUNK_SIZE must be positive")
        
        if self.MAX_NEW_TOKENS <= 0:
            errors.append("MAX_NEW_TOKENS must be positive")
        
        if not 0 <= self.TEMPERATURE <= 1:
            errors.append("TEMPERATURE must be between 0 and 1")
        
        if self.REQUEST_TIMEOUT <= 0:
            errors.append("REQUEST_TIMEOUT must be positive")
        
        if errors:
            raise ValueError(f"Configuration errors: {'; '.join(errors)}")
        
        return True

# Create validated config instance
config = ProductionConfig()
config.validate() 