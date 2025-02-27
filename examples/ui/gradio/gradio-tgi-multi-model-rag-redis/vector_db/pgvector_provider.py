from typing import Optional
from vector_db.db_provider import DBProvider
from langchain_community.vectorstores import PGVector
from langchain_core.vectorstores import VectorStoreRetriever
import os

class PGVectorProvider(DBProvider):
    type = "PGVECTOR"
    url: Optional[str] = None
    collection_name: Optional[str] = None
    retriever: Optional[VectorStoreRetriever] = None
    db: Optional[PGVector] = None
    def __init__(self):
        super().__init__()
        self.url = os.getenv('PGVECTOR_URL')
        self.collection_name = os.getenv('PGVECTOR_COLLECTION_NAME')
        if self.url is None:
            raise ValueError("PGVECTOR_URL is not specified")
        if self.collection_name is None:
            raise ValueError("PGVECTOR_COLLECTION_NAME is not specified")

        pass
  
    @classmethod
    def _get_type(cls) -> str:
        """Returns type of the db provider"""
        return cls.type
    
    def get_retriever(self) -> VectorStoreRetriever:
        if self.retriever is None:
            self.db = PGVector(
                connection_string=self.url,
                collection_name=self.collection_name,
                embedding_function=self.get_embeddings())

            self.retriever = self.db.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 5, "distance_threshold": 0.5})
         
        return self.retriever

      
    