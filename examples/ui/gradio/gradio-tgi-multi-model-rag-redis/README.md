# Gradio UI for RAG using Hugging Face Text Generation Inference server and Redis

This is a simple UI example for a RAG-based Chatbot using Gradio, Hugging Face TGI server, and Redis as a vector database.

You can refer to those different notebooks to get a better understanding of the flow:

- [Data Ingestion to Redis with Langchain](../../../notebooks/langchain/Langchain-Redis-Ingest.ipynb)
- [Redis querying with Langchain](../../../notebooks/langchain/Langchain-Redis-Query.ipynb)
- [Full RAG example with Redis and Langchain](../../../notebooks/langchain/Langchain-Redis-RAG-HFTGI.ipynb)

## Requirements

- A Hugging Face Text Generation Inference server with a deployed LLM. This example is based on Llama2 but depending on your LLM you may need to adapt the prompt.
- A Redis installation with a Database and an Index already populated with documents. See [here](../../../../redis_deployment/README.md) for deployment instructions.
- An index in the Redis DB that you have populated with documents. See [here](../../../notebooks/langchain/Langchain-Redis-Ingest.ipynb) for an example.

## Deployment on OpenShift

A pre-built container image of the application is available at: `quay.io/rh-aiservices-bu/gradio-hftgi-rag-redis:latest`

In the `deployment` folder, you will find the files necessary to deploy the application:

- `cm_redis_schema.yaml`: A ConfigMap containing the schema for the database you have created in Redis (see the ingestion Notebook).
- `deployment.yaml`: you must provide the URL of your inference server in the placeholder on L54 and Redis information on L56 and L58. Please feel free to modify other parameters as you see fit.
- `service.yaml`
- `route.yaml`
- `provider-config.yaml`: This is a configuration file that contains the configuration of various providers. The current supported providers are
  - Hugging Face
  - OpenAI
  - NVIDIA

The different parameters you can/must pass as environment variables in the deployment are:

- REDIS_URL - mandatory
- REDIS_INDEX - mandarory

The deployment replicas is set to 0 initially to let you properly fill in those parameters. Don't forget to scale it up if you want see something 😉!

## Deploy Locally
Create venv:
`python -m venv .venv`
Activate:
`source .venv/bin activate`
Install requirements-local.txt
`pip install -r requirements.txt`

Configurations files:
- `.env`: You must provide REDIS or PGVECTOR 
- `config.yaml` You must specift an llm provider, such as Hugging Face or OpenAI, and fill the necessary fields for this provider.

Run the app:
`python app.py`