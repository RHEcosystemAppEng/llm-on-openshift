from generator.template import CONTEXTUALIZE_Q_PROMPT, GENERATE_PROPOSAL_TEMPLATE, Q_A_PROMPT, Q_AND_A_PROMPT, QUERY_UPDATE_PROPOSAL_TEMPLATE, UPDATE_PROPOSAL_TEMPLATE
from langchain.prompts import PromptTemplate
import os
from langchain.chains import RetrievalQA
from vector_db.db_provider_factory import FAISS, DBFactory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


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
        

    def get_proposal_template_chain(self, llm):
        generate_proposal_prompt = PromptTemplate.from_template(GENERATE_PROPOSAL_TEMPLATE)

        # multi_query_retriever = MultiQueryRetriever.from_llm(
        #                 retriever=self.retriever, llm=llm , parser_key="lines"
        #             )
        
        # import logging

        # logging.basicConfig()
        # logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

        return RetrievalQA.from_chain_type(
            llm,
            retriever=self.retriever,
            chain_type_kwargs={"prompt": generate_proposal_prompt},
            return_source_documents=True,
        )

    def get_update_proposal_chain(self, llm):
        update_proposal_prompt = PromptTemplate.from_template(UPDATE_PROPOSAL_TEMPLATE)
        query_update_proposal_prompt = PromptTemplate.from_template(QUERY_UPDATE_PROPOSAL_TEMPLATE)
        combine_docs_chain = create_stuff_documents_chain(llm, update_proposal_prompt)

        return RunnableParallel({'context': query_update_proposal_prompt| RunnableLambda(lambda x: x.text)  | self.retriever, 'old_proposal': lambda x:x['old_proposal'], 'user_query': lambda x: x['user_query']}) | RunnableParallel({"source_documents": lambda x: x['context'], 'result': combine_docs_chain})


    def get_qa_chain(self, llm):
        contextualize_q_system_prompt = """
                                        ### [INST]
                                        Given a chat history and the latest user question 
                                        which might reference context in the chat history, 
                                        formulate a standalone single question 
                                        which can be understood without the chat history. 
                                        Do NOT answer the question, 
                                        just reformulate it in a single question if needed and 
                                        otherwise return it as is.
                                        [/INST]
                                        """
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
                                    [
                                        ("system", contextualize_q_system_prompt),
                                        MessagesPlaceholder("chat_history"),
                                        ("human", "{input}"),
                                    ]
                                )
        
        history_aware_retriever = create_history_aware_retriever(
                                    llm, self.retriever, contextualize_q_prompt
                                )

        qa_system_prompt = """
                            ### [INST]
                            You are an assistant for question-answering tasks. 
                            Use the following pieces of retrieved context to answer the question. 
                            Answer the following question concisely and directly without asking any additional questions.
                            Answer questions concisely and do not ask follow-up questions.
                            If you don't know the answer, just say that you don't know. 
                            Use three sentences maximum and keep the answer concise.
                            ### Context:
                            {context}

                            [/INST]
                            """
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        return rag_chain
    
 
    def get_conversational_retrieval_chain(self, llm):
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm, chain_type="stuff", retriever=self.retriever
        )

    def get_question_answer_chain(self, llm):
        history_aware_retriever = create_history_aware_retriever(
            llm, self.retriever, CONTEXTUALIZE_Q_PROMPT)

        question_answer_chain = create_stuff_documents_chain(llm, Q_AND_A_PROMPT)
        return create_retrieval_chain(history_aware_retriever, 
                                      question_answer_chain, 
                                      return_generated_question=True)
    
    