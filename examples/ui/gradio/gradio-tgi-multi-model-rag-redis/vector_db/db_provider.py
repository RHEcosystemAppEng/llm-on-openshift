from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings


class DBProvider:
    """Base class for DB Provider."""

    embeddings: Optional[Embeddings] = None

    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
    ) -> None:
        if embeddings is None:
            self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        else:
            self.embeddings = embeddings

    def _get_type(self) -> str:
        pass

    def get_retriever(self) -> VectorStoreRetriever:
        pass

    def get_embeddings(self) -> Embeddings:
        return self.embeddings
