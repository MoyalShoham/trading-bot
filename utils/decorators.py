"""
Utility decorators for rate limiting, retries, and other common patterns.
"""

import asyncio
import functools
import time
from typing import Callable, Any, Optional, Dict
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter using token bucket algorithm."""
    
    def __init__(self, calls: int, period: float):
        """
        Initialize rate limiter.
        
        Args:
            calls: Number of calls allowed in the period
            period: Time period in seconds
        """
        self.calls = calls
        self.period = period
        self.tokens = calls
        self.last_update = time.time()
        
    async def acquire(self) -> None:
        """Wait until a token is available."""
        current_time = time.time()
        time_passed = current_time - self.last_update
        
        # Add tokens based on time passed
        self.tokens = min(self.calls, self.tokens + time_passed * (self.calls / self.period))
        self.last_update = current_time
        
        if self.tokens < 1:
            sleep_time = (1 - self.tokens) * (self.period / self.calls)
            await asyncio.sleep(sleep_time)
            self.tokens = 0
        else:
            self.tokens -= 1


# Global rate limiters
_rate_limiters: Dict[str, RateLimiter] = {}


def rate_limit(calls: int, period: float, key: Optional[str] = None):
    """
    Rate limiting decorator.
    
    Args:
        calls: Number of calls allowed in the period
        period: Time period in seconds  
        key: Optional key to group rate limits (default: function name)
    """
    def decorator(func: Callable) -> Callable:
        limiter_key = key or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if limiter_key not in _rate_limiters:
                _rate_limiters[limiter_key] = RateLimiter(calls, period)
            
            await _rate_limiters[limiter_key].acquire()
            return await func(*args, **kwargs)
            
        return wrapper
    return decorator


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry on
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                        
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise
                    
                    logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    
                    if attempt < max_retries:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff_factor
            
            # Should never reach here, but just in case
            raise last_exception
            
        return wrapper
    return decorator


def timeout(seconds: float):
    """
    Timeout decorator for async functions.
    
    Args:
        seconds: Timeout in seconds
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logger.error(f"Function {func.__name__} timed out after {seconds} seconds")
                raise
                
        return wrapper
    return decorator


def cache_result(ttl: float = 300):
    """
    Cache decorator with TTL (time to live).
    
    Args:
        ttl: Cache time to live in seconds
    """
    cache = {}
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            current_time = time.time()
            
            # Check if cached result exists and is still valid
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if current_time - timestamp < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return result
                else:
                    # Remove expired cache entry
                    del cache[cache_key]
            
            # Execute function and cache result
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            cache[cache_key] = (result, current_time)
            logger.debug(f"Cache miss for {func.__name__}, result cached")
            return result
            
        return wrapper
    return decorator


def measure_execution_time(log_level: str = "DEBUG"):
    """
    Decorator to measure and log function execution time.
    
    Args:
        log_level: Logging level for the execution time message
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                    
                execution_time = time.time() - start_time
                log_func = getattr(logger, log_level.lower(), logger.debug)
                log_func(f"Function {func.__name__} executed in {execution_time:.4f} seconds")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Function {func.__name__} failed after {execution_time:.4f} seconds: {e}")
                raise
                
        return wrapper
    return decorator


def validate_inputs(**validators):
    """
    Input validation decorator.
    
    Args:
        **validators: Dictionary of parameter_name: validation_function pairs
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get function signature to map args to parameter names
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate inputs
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValueError(f"Invalid value for parameter '{param_name}': {value}")
            
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
                
        return wrapper
    return decorator


def singleton(cls):
    """
    Singleton decorator for classes.
    """
    instances = {}
    
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting to close circuit
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def can_execute(self) -> bool:
        """Check if function can be executed."""
        if self.state == 'CLOSED':
            return True
        elif self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'HALF_OPEN'
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def on_success(self):
        """Handle successful execution."""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'


def circuit_breaker(failure_threshold: int = 5, recovery_timeout: float = 60):
    """
    Circuit breaker decorator.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time to wait before attempting to close circuit
    """
    breakers = {}
    
    def decorator(func: Callable) -> Callable:
        breaker_key = f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if breaker_key not in breakers:
                breakers[breaker_key] = CircuitBreaker(failure_threshold, recovery_timeout)
            
            breaker = breakers[breaker_key]
            
            if not breaker.can_execute():
                raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                breaker.on_success()
                return result
                
            except Exception as e:
                breaker.on_failure()
                logger.error(f"Circuit breaker recorded failure for {func.__name__}: {e}")
                raise
                
        return wrapper
    return decorator
