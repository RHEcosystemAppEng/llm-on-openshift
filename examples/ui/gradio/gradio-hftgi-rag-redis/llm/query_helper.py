from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import os
from langchain.vectorstores.redis import Redis
from langchain.chains import RetrievalQA


REDIS_URL = os.getenv('REDIS_URL')
REDIS_INDEX = os.getenv('REDIS_INDEX')

############################
# LLM chain implementation #
############################

# Document store: Redis vector store
embeddings = HuggingFaceEmbeddings()
rds = Redis.from_existing_index(
    embeddings,
    redis_url=REDIS_URL,
    index_name=REDIS_INDEX,
    schema="redis_schema.yaml"
)

prompt_template = """<s>[INST] <<SYS>>
Instructions:
- You are a helpful assistant.
- You are an expert in writing a project proposal for products owned by Red Hat for the customer information provided in the question below.
- Your job is to look at the  question and create the project proposal addressed to the customer mentioned in question.
- Base your answer on the provided context and question and not on prior knowledge.
- The proposal should contain headings and sub-headings and each heading and sub-heading should be in bold.
- Each section in the proposal should contain only three items.
- Proposal should be minimum of 500 lines.
<</SYS>>

Question: {question}
Context: {context}

[/INST]
"""

QA_CHAIN_PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"])

def get_qa_chain(llm):
    return RetrievalQA.from_chain_type(
        llm,
        retriever=rds.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4, "distance_threshold": 0.5}),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True
    )
