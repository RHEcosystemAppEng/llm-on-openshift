from generator.template import GENERATE_PROPOSAL_TEMPLATE, QUERY_UPDATE_PROPOSAL_TEMPLATE, UPDATE_PROPOSAL_TEMPLATE
from langchain.prompts import PromptTemplate
import os
from langchain.chains import RetrievalQA
from vector_db.db_provider_factory import FAISS, DBFactory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain.retrievers.multi_query import MultiQueryRetriever

############################
# LLM chain implementation #
############################


class QueryHelper:
    def __init__(self):
        self.retriever = None
        self.init_retriever()

    def init_retriever(self):
        if self.retriever is None:
            try:
                type = os.getenv("DB_TYPE") if os.getenv("DB_TYPE") else "REDIS"
                if type is None:
                    raise ValueError("DB_TYPE is not specified")
                print(f"Retriever DB: {type}")
                db_factory = DBFactory()
                self.retriever = db_factory.get_retriever(type)
            except Exception as e:
                print(e)
                print(
                    f"{type} server is unavailable. Project proposal will be generated without RAG content."
                )
                self.retriever = db_factory.get_retriever(FAISS)  
        return self.retriever

    def retrieve_context(self, query, k=6):
        """Retrieve relevant context from RAG database."""
        retrieved_docs = self.retriever.get_relevant_documents(query, k=k)
        return "\n".join([doc.page_content for doc in retrieved_docs])
    
    def retrieve_context_with_source(self, query, k=6):
        """Retrieve relevant context with source information from RAG database."""
        retrieved_docs = self.retriever.get_relevant_documents(query, k=k)
        context_with_source = "\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in retrieved_docs])
        source_documents =  [f"Source: {doc.metadata.get('source', 'Unknown')}" for doc in retrieved_docs]
        return source_documents, context_with_source
        

    def get_qa_chain(self, llm):
        generate_proposal_prompt = PromptTemplate.from_template(GENERATE_PROPOSAL_TEMPLATE)

        multi_query_retriever = MultiQueryRetriever.from_llm(
                        retriever=self.retriever, llm=llm , parser_key="lines"
                    )
        
        import logging

        logging.basicConfig()
        logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

        return RetrievalQA.from_chain_type(
            llm,
            retriever=multi_query_retriever,
            chain_type_kwargs={"prompt": generate_proposal_prompt},
            return_source_documents=True,
        )

    def get_update_proposal_chain(self, llm):
        update_proposal_prompt = PromptTemplate.from_template(UPDATE_PROPOSAL_TEMPLATE)
        query_update_proposal_prompt = PromptTemplate.from_template(QUERY_UPDATE_PROPOSAL_TEMPLATE)
        combine_docs_chain = create_stuff_documents_chain(llm, update_proposal_prompt)

        return RunnableParallel({'context': query_update_proposal_prompt| RunnableLambda(lambda x: x.text)  | self.retriever, 'old_proposal': lambda x:x['old_proposal'], 'user_query': lambda x: x['user_query']}) | RunnableParallel({"source_documents": lambda x: x['context'], 'result': combine_docs_chain})