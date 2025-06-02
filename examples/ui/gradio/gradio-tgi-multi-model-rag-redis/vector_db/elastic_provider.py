import os
from typing import Optional

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_elasticsearch.vectorstores import ElasticsearchStore

from vector_db.db_provider import DBProvider


class ElasticProvider(DBProvider):
    type = "ELASTIC"
    url: Optional[str] = None
    index: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    retriever: Optional[any] = None
    db: Optional[any] = None
    retriever: Optional[VectorStoreRetriever] = None

    def __init__(self):
        super().__init__()
        self.url = os.getenv("ELASTIC_URL")
        self.index = (
            os.getenv("ELASTIC_INDEX") if os.getenv("ELASTIC_INDEX") else "docs"
        )
        self.user = (
            os.getenv("ELASTIC_USER") if os.getenv("ELASTIC_USER") else "elastic"
        )
        self.password = os.getenv("ELASTIC_PASSWORD")
        if self.url is None:
            raise ValueError("ELASTIC_URL is not specified")
        if self.password is None:
            raise ValueError("ELASTIC_PASSWORD is not specified")

        pass

    @classmethod
    def _get_type(cls) -> str:
        """Returns type of the db provider"""
        return cls.type

    def get_retriever(self) -> VectorStoreRetriever:
        if self.retriever is None:
            self.db = ElasticsearchStore(
                embedding=self.get_embeddings(),
                es_url=self.url,
                es_user=self.user,
                es_password=self.password,
                index_name=self.index,
            )

            self.retriever = self.db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4, "distance_threshold": 0.5},
            )

        return self.retriever
