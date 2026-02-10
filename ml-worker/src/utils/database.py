from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from contextlib import asynccontextmanager
import os

class DatabaseManager:
    def __init__(self):
        self.engine = None
        self.session_factory = None
    
    async def initialize(self):
        database_url = os.getenv('DATABASE_URL')
        
        self.engine = create_async_engine(
            database_url,
            echo=False,
            pool_size=5,
            max_overflow=10
        )
        
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    @asynccontextmanager
    async def get_session(self):
        session = self.session_factory()
        try:
            yield session
        finally:
            await session.close()