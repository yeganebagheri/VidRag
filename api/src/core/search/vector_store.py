from typing import List, Optional, Dict, Any
from src.core.models.query import QueryResult

class VectorStore:
    def __init__(self):
        self.initialized = False
        # Mock data
        self.mock_results = [
            QueryResult(
                video_id="video_1",
                segment_id="seg_1",
                text="This is a sample video segment about machine learning",
                score=0.85,
                timestamp=10.5
            ),
            QueryResult(
                video_id="video_2", 
                segment_id="seg_2",
                text="Another segment discussing artificial intelligence",
                score=0.75,
                timestamp=25.2
            )
        ]
    
    async def initialize(self):
        """Initialize vector store"""
        self.initialized = True
    
    async def search(self, query: str, k: int = 10, filters: Optional[Dict] = None) -> List[QueryResult]:
        """Search for similar vectors - mock implementation"""
        # Mock search - return filtered results based on query
        results = []
        for result in self.mock_results:
            if any(word.lower() in result.text.lower() for word in query.split()):
                results.append(result)
        
        return results[:k]
    
    async def close(self):
        """Close vector store connection"""
        self.initialized = False

# Dependency injection
async def get_vector_store() -> VectorStore:
    store = VectorStore()
    await store.initialize()
    return store