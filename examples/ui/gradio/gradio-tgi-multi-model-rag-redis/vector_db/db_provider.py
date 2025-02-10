from typing import Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

class DBProvider:
    """Base class for DB Provider.
    """
    embeddings: Optional[Embeddings] = None
    def __init__(self) -> None:
        self.embeddings = HuggingFaceEmbeddings()
        pass

    def _get_type(self) -> str:
        pass

    def get_retriever(self) -> VectorStoreRetriever:
        pass

    def get_embeddings(self) -> Embeddings:
        return self.embeddings