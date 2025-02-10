
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

QUERY_UPDATE_PROPOSAL_TEMPLATE = """"
### [INST]
Old Proposal:
{old_proposal}

### User Query:
{user_query}
[/INST]
"""