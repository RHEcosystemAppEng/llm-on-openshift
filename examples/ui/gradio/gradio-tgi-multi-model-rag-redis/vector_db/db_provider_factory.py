from langchain_core.vectorstores import VectorStoreRetriever

from vector_db.db_provider import DBProvider
from vector_db.elastic_provider import ElasticProvider
from vector_db.faiss_provider import FAISSProvider
from vector_db.mssql_provider import MSSQLProvider
from vector_db.pgvector_provider import PGVectorProvider
from vector_db.redis_provider import RedisProvider

PGVECTOR = "PGVECTOR"
REDIS = "REDIS"
FAISS = "FAISS"
ELASTIC = "ELASTIC"
MSSQL = "MSSQL"


class DBFactory:
    providers: dict[str, DBProvider] = {}

    def __init__(self):
        pass

    def create_db_provider(self, type: str) -> DBProvider:
        if type == PGVECTOR:
            return PGVectorProvider()
        elif type == REDIS:
            return RedisProvider()
        elif type == FAISS:
            return FAISSProvider()
        elif type == ELASTIC:
            return ElasticProvider()
        elif type == MSSQL:
            return MSSQLProvider()
        else:
            raise ValueError(type)

    def get_db_provider(self, type: str):
        if type not in self.providers:
            self.providers[type] = self.create_db_provider(type)

        return self.providers[type]

    def get_retriever(self, type: str) -> VectorStoreRetriever:
        return self.get_db_provider(type).get_retriever()

    @classmethod
    def get_providers(cls) -> list[str]:
        return [PGVECTOR, REDIS, FAISS, ELASTIC]
