import redis
import json
import pickle
import hashlib
import numpy as np
from typing import Any, Optional, Dict, List, Union
import asyncio
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class EnhancedCacheManager:
    """Enhanced cache manager with Redis backend and hierarchical caching"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 default_ttl: int = 3600):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.redis_client = None
        self.local_cache = {}  # Fallback local cache
        self.initialized = False
        
        # Cache categories with different TTLs
        self.cache_categories = {
            'embeddings': 86400,      # 24 hours for embeddings
            'processed_video': 7200,   # 2 hours for processed video data
            'search_results': 1800,    # 30 minutes for search results
            'llm_responses': 3600,     # 1 hour for LLM responses
            'video_metadata': 86400,   # 24 hours for video metadata
            'scene_data': 7200,        # 2 hours for scene detection data
            'knowledge_graph': 86400,  # 24 hours for knowledge graphs
        }
    
    async def initialize(self):
        """Initialize cache manager with Redis connection"""
        try:
            import redis.asyncio as redis_async
            self.redis_client = redis_async.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            await self.redis_client.ping()
            logger.info("✅ Redis cache manager initialized successfully")
            self.initialized = True
        except Exception as e:
            logger.warning(f"Redis unavailable, using local cache: {e}")
            self.redis_client = None
            self.initialized = True
    
    def _generate_cache_key(self, category: str, key: str, **kwargs) -> str:
        """Generate hierarchical cache key"""
        # Create deterministic key with category prefix
        key_data = f"{category}:{key}"
        if kwargs:
            # Sort kwargs for consistent key generation
            sorted_kwargs = sorted(kwargs.items())
            params_str = json.dumps(sorted_kwargs, sort_keys=True)
            key_data += f":{hashlib.md5(params_str.encode()).hexdigest()}"
        return key_data
    
    async def get(self, category: str, key: str, **kwargs) -> Optional[Any]:
        """Get cached value with category-based TTL"""
        cache_key = self._generate_cache_key(category, key, **kwargs)
        
        try:
            if self.redis_client:
                # Try Redis first
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    if category == 'embeddings':
                        # Special handling for numpy arrays
                        return self._deserialize_embeddings(cached_data)
                    return json.loads(cached_data)
            
            # Fallback to local cache
            if cache_key in self.local_cache:
                item = self.local_cache[cache_key]
                if datetime.now() < item['expires']:
                    return item['data']
                else:
                    del self.local_cache[cache_key]
            
            return None
            
        except Exception as e:
            logger.error(f"Cache get error for {cache_key}: {e}")
            return None
    
    async def set(self, category: str, key: str, value: Any, ttl: Optional[int] = None, **kwargs):
        """Set cached value with category-based TTL"""
        cache_key = self._generate_cache_key(category, key, **kwargs)
        ttl = ttl or self.cache_categories.get(category, self.default_ttl)
        
        try:
            if self.redis_client:
                # Serialize based on category
                if category == 'embeddings':
                    serialized_value = self._serialize_embeddings(value)
                else:
                    serialized_value = json.dumps(value, default=str)
                
                await self.redis_client.setex(cache_key, ttl, serialized_value)
            
            # Also store in local cache as backup
            self.local_cache[cache_key] = {
                'data': value,
                'expires': datetime.now() + timedelta(seconds=ttl)
            }
            
        except Exception as e:
            logger.error(f"Cache set error for {cache_key}: {e}")
            # Store in local cache as fallback
            self.local_cache[cache_key] = {
                'data': value,
                'expires': datetime.now() + timedelta(seconds=ttl)
            }
    
    def _serialize_embeddings(self, embeddings: Union[np.ndarray, List]) -> str:
        """Serialize numpy arrays for caching"""
        if isinstance(embeddings, np.ndarray):
            return json.dumps({
                'type': 'numpy',
                'data': embeddings.tolist(),
                'shape': embeddings.shape,
                'dtype': str(embeddings.dtype)
            })
        elif isinstance(embeddings, list):
            return json.dumps({
                'type': 'list',
                'data': embeddings
            })
        return json.dumps(embeddings)
    
    def _deserialize_embeddings(self, data: str) -> Union[np.ndarray, List]:
        """Deserialize numpy arrays from cache"""
        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict) and parsed.get('type') == 'numpy':
                return np.array(parsed['data'], dtype=parsed['dtype']).reshape(parsed['shape'])
            elif isinstance(parsed, dict) and parsed.get('type') == 'list':
                return parsed['data']
            return parsed
        except:
            return json.loads(data)
    
    async def delete(self, category: str, key: str, **kwargs):
        """Delete cached value"""
        cache_key = self._generate_cache_key(category, key, **kwargs)
        
        try:
            if self.redis_client:
                await self.redis_client.delete(cache_key)
            
            if cache_key in self.local_cache:
                del self.local_cache[cache_key]
                
        except Exception as e:
            logger.error(f"Cache delete error for {cache_key}: {e}")
    
    async def clear_category(self, category: str):
        """Clear all cache entries for a specific category"""
        try:
            if self.redis_client:
                pattern = f"{category}:*"
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
            
            # Clear from local cache too
            keys_to_delete = [k for k in self.local_cache.keys() if k.startswith(f"{category}:")]
            for key in keys_to_delete:
                del self.local_cache[key]
                
        except Exception as e:
            logger.error(f"Cache clear category error for {category}: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            'local_cache_size': len(self.local_cache),
            'categories': list(self.cache_categories.keys()),
            'redis_connected': self.redis_client is not None
        }
        
        if self.redis_client:
            try:
                info = await self.redis_client.info()
                stats.update({
                    'redis_memory_used': info.get('used_memory_human'),
                    'redis_keys': info.get('db0', {}).get('keys', 0)
                })
            except:
                pass
        
        return stats
    
    async def close(self):
        """Close cache manager connections"""
        if self.redis_client:
            await self.redis_client.close()
        logger.info("Cache manager closed")

# Dependency injection functions
_cache_manager_instance = None

async def get_cache_manager() -> EnhancedCacheManager:
    """Get singleton cache manager instance"""
    global _cache_manager_instance
    if _cache_manager_instance is None:
        _cache_manager_instance = EnhancedCacheManager()
        await _cache_manager_instance.initialize()
    return _cache_manager_instance

# Context managers for specific cache operations
class CacheContext:
    """Context manager for cache operations"""
    
    def __init__(self, cache_manager: EnhancedCacheManager, category: str):
        self.cache_manager = cache_manager
        self.category = category
    
    async def get_or_compute(self, key: str, compute_func, ttl: Optional[int] = None, **kwargs):
        """Get from cache or compute and cache the result"""
        cached_result = await self.cache_manager.get(self.category, key, **kwargs)
        if cached_result is not None:
            return cached_result
        
        # Compute result
        if asyncio.iscoroutinefunction(compute_func):
            result = await compute_func()
        else:
            result = compute_func()
        
        # Cache the result
        await self.cache_manager.set(self.category, key, result, ttl, **kwargs)
        return result

# Decorators for automatic caching
def cache_result(category: str, key_func=None, ttl: Optional[int] = None):
    """Decorator to automatically cache function results"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache_manager = await get_cache_manager()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = await cache_manager.get(category, cache_key)
            if cached_result is not None:
                return cached_result
            
            # Compute result
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Cache result
            await cache_manager.set(category, cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator