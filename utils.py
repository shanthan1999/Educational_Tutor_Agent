"""
Utility functions for production deployment
"""

import time
import hashlib
import logging
from functools import wraps
from typing import Any, Dict, Optional, Callable
from datetime import datetime, timedelta
import json

# Setup structured logging
def setup_production_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup production-ready logging with structured format."""
    
    # Create custom formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        # Ensure directory exists
        import os
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Simple in-memory cache
class SimpleCache:
    """Thread-safe in-memory cache with TTL support."""
    
    def __init__(self, default_ttl: int = 3600):
        self.cache: Dict[str, Dict] = {}
        self.default_ttl = default_ttl
    
    def _is_expired(self, entry: Dict) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() > entry['expires_at']
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self.cache:
            entry = self.cache[key]
            if not self._is_expired(entry):
                return entry['value']
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL."""
        ttl = ttl or self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)
        
        self.cache[key] = {
            'value': value,
            'expires_at': expires_at,
            'created_at': datetime.now()
        }
    
    def clear_expired(self) -> int:
        """Clear expired entries and return count."""
        expired_keys = [
            key for key, entry in self.cache.items() 
            if self._is_expired(entry)
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'total_entries': len(self.cache),
            'expired_entries': len([
                k for k, v in self.cache.items() 
                if self._is_expired(v)
            ])
        }

# Global cache instance
cache = SimpleCache()

def cached(ttl: int = 3600, key_func: Optional[Callable] = None):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
                cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
    """Decorator for retrying functions with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    
                    logging.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    time.sleep(delay)
            
            # All retries failed
            logging.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            raise last_exception
        
        return wrapper
    return decorator

def monitor_performance(func):
    """Decorator to monitor function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            execution_time = time.time() - start_time
            
            # Log performance metrics
            logging.info(
                f"Performance: {func.__name__} - "
                f"Time: {execution_time:.3f}s - "
                f"Success: {success}" +
                (f" - Error: {error}" if error else "")
            )
        
        return result
    
    return wrapper

def validate_input(validator_func: Callable):
    """Decorator for input validation."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate input
            validation_result = validator_func(*args, **kwargs)
            if validation_result is not True:
                raise ValueError(f"Input validation failed: {validation_result}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

# Input validators
def validate_query(query: str, *args, **kwargs) -> bool:
    """Validate query input."""
    if not query or not isinstance(query, str):
        return "Query must be a non-empty string"
    
    if len(query.strip()) < 3:
        return "Query must be at least 3 characters long"
    
    if len(query) > 1000:
        return "Query must be less than 1000 characters"
    
    return True

# Health check utilities
class HealthChecker:
    """System health monitoring."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.request_count = 0
        self.error_count = 0
    
    def record_request(self):
        """Record a new request."""
        self.request_count += 1
    
    def record_error(self):
        """Record an error."""
        self.error_count += 1
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        uptime = datetime.now() - self.start_time
        error_rate = (self.error_count / max(self.request_count, 1)) * 100
        
        return {
            'status': 'healthy' if error_rate < 10 else 'unhealthy',
            'uptime_seconds': int(uptime.total_seconds()),
            'total_requests': self.request_count,
            'total_errors': self.error_count,
            'error_rate_percent': round(error_rate, 2),
            'cache_stats': cache.stats(),
            'timestamp': datetime.now().isoformat()
        }

# Global health checker
health_checker = HealthChecker() 