from typing import Optional
from langchain_community.vectorstores import Redis
from langchain_core.vectorstores import VectorStoreRetriever
from vector_db.db_provider import DBProvider
import os

class RedisProvider(DBProvider):
    type = "Redis"
    url: Optional[str] = None
    index: Optional[str] = None
    schema: Optional[str] = None
    retriever: Optional[any] = None
    db: Optional[any] = None
    retriever: Optional[VectorStoreRetriever] = None

    def __init__(self):
        super().__init__()
        self.url = os.getenv('REDIS_URL')
        self.index = os.getenv('REDIS_INDEX')
        self.schema =  os.getenv('REDIS_SCHEMA') if os.getenv('REDIS_SCHEMA') else "redis_schema.yaml"
        if self.url is None:
            raise ValueError("REDIS_URL is not specified")
        if self.index is None:
            raise ValueError("REDIS_INDEX is not specified")

        pass
  
    @classmethod
    def _get_type(cls) -> str:
        """Returns type of the db provider"""
        return cls.type
    
    def get_retriever(self) -> VectorStoreRetriever:
        if self.retriever is None:
            self.db = Redis.from_existing_index(
                self.get_embeddings(),
                redis_url=self.url,
                index_name=self.index,
                schema=self.schema
            )
            self.retriever = self.db.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 4, "distance_threshold": 0.5})
         
        return self.retriever
