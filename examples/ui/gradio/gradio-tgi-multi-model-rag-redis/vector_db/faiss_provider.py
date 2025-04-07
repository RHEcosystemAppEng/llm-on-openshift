from typing import Optional

from langchain_community.vectorstores.faiss import FAISS
from langchain_core.vectorstores import VectorStoreRetriever

from vector_db.db_provider import DBProvider


class FAISSProvider(DBProvider):
    type = "FAISS"
    url: Optional[str] = None
    index: Optional[str] = None
    schema: Optional[str] = None
    retriever: Optional[VectorStoreRetriever] = None
    db: Optional[FAISS] = None

    def __init__(self):
        super().__init__()
        pass

    @classmethod
    def _get_type(cls) -> str:
        """Returns type of the db provider"""
        return cls.type

    def get_retriever(self) -> VectorStoreRetriever:
        if self.retriever is None:
            self.db = FAISS.from_texts([""], self.get_embeddings())
            self.retriever = self.db.as_retriever()

        return self.retriever
