from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

VECTOR_DB_QUERY_TEMPLATE = """
    Generate a structured project proposal for the product **{product}** addressed to the company **{company}**.
    The proposal should include the following related to the product  **{product}**:

    1. **Project Title**
    2. **Executive Summary**
    3. **Background & Probleççm Statement**
    3. **Overview**
    4. **Features**
    5. **Proposed Solution**
    6. **Implementation Plan**
    7. **Benefits & Business Value**
    8. **Risks & Mitigation Strategies**
    9. **Support & Maintenance**
    10. **Conclusion & Next Steps**
"""


GENERATE_PROPOSAL_TEMPLATE = """
### [INST]
Instructions:
    You are an AI assistant specialized in writing structured project proposals.
    Use ONLY the provided context to generate the proposal.
    Do NOT use any external knowledge beyond what is in the context.

    ### Context:
    {context}

    ---
    ### QUESTION:
    {question}
    Generate the project proposal in markdown format.
    Keep the response structured, concise, and professional.
[/INST]
"""


UPDATE_PROPOSAL_TEMPLATE = """
### [INST]
Instructions:
- You are an AI assistant tasked with updating a project proposal.
- Update the old proposal based on the user query, using the provided old proposal, context, and question.
- Do not rely on prior knowledge; base your response solely on the provided information.
- Update the proposal in markdown format.
- Modify only the content based on the user's request, while keeping everything else the same, but you can renumber sections.

Context:
{context}

Old Proposal:
{old_proposal}

### User Query:
{user_query}
[/INST]
"""

QUERY_UPDATE_PROPOSAL_TEMPLATE = """
### [INST]
Old Proposal:
{old_proposal}

### User Query:
{user_query}
[/INST]
"""


### Contextualize question ###
CONTEXTUALIZE_Q_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

CONTEXTUALIZE_Q_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

Q_AND_A_SYSTEM_PROMPT = """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to
    answer the question. If you don't know the answer,
    just say that you don't know. Use three sentences maximum
    and keep the answer concise.
    Context: {context}

    """

Q_AND_A_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", Q_AND_A_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

Q_A_PROMPT = """
### [INST]
Instructions:
    You are an AI assistant for question-answering tasks. Use the following pieces of retrieved context to
    answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum
    and keep the answer concise.

    ### Context:
    {context}

    ---
    ### QUESTION:
    {question}

    Helpful Answer:
[/INST]
"""
