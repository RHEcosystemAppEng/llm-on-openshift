import os
import random
import time
from collections.abc import Generator
from queue import Empty, Queue
from threading import Thread
from typing import Optional
import os
from markdown import markdown
import pdfkit

import gradio as gr
from prometheus_client import start_http_server, Counter
import tempfile
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceTextGenInference
from langchain.prompts import PromptTemplate
from langchain.vectorstores.redis import Redis
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import requests

load_dotenv()

# Parameters

APP_TITLE = os.getenv('APP_TITLE', 'Talk with your documentation')

INFERENCE_SERVER_URL = os.getenv('INFERENCE_SERVER_URL')
MAX_NEW_TOKENS = int(os.getenv('MAX_NEW_TOKENS', 512))
TOP_K = int(os.getenv('TOP_K', 10))
TOP_P = float(os.getenv('TOP_P', 0.95))
TYPICAL_P = float(os.getenv('TYPICAL_P', 0.95))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.01))
REPETITION_PENALTY = float(os.getenv('REPETITION_PENALTY', 1.03))
PDF_FILE_DIR = os.getenv('PDF_FILE_DIR', tempfile.gettempdir())

REDIS_URL = os.getenv('REDIS_URL')
REDIS_INDEX = os.getenv('REDIS_INDEX')
TIMEOUT = int(os.getenv('TIMEOUT', 30))

# Start Prometheus metrics server
start_http_server(8000)

# Create metric
FEEDBACK_COUNTER = Counter("feedback_stars", "Number of feedbacks by stars", ["stars", "model_id"])
MODEL_USAGE_COUNTER = Counter('model_usage', 'Number of times a model was used', ['model_id'])
REQUEST_TIME = Gauge('request_duration_seconds', 'Time spent processing a request', ['model_id'])

start_time = time.perf_counter() # start and end time to get the precise timing of the request
        
def get_model_id():
    model_id = "Unavailable"
    try:
        r = requests.get(f'{INFERENCE_SERVER_URL}/info')
        if r.status_code == 200:
            model_id = r.json()['model_id']
        end_time = time.perf_counter()
        # Record successful request time
        REQUEST_TIME.labels(model_id=model_id).set(end_time - start_time)
    except TimeoutError:  # or whatever exception your client throws on timeout
            end_time = time.perf_counter()
    return model_id

model_id = get_model_id()
# PDF Generation
def create_pdf(text):
    output_filename = os.path.join(PDF_FILE_DIR, 'Test.pdf')
    html_text = markdown(text, output_format='html4')
    pdfkit.from_string(html_text, output_filename)

# Streaming implementation
class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: any) -> None:
        return self.q.empty()

def remove_source_duplicates(input_list):
    unique_list = []
    for item in input_list:
        if item.metadata['source'] not in unique_list:
            unique_list.append(item.metadata['source'])
    return unique_list


def stream(input_text) -> Generator:
    # Create a Queue
    job_done = object()

    # Create a function to call - this will run in a thread
    def task():
        MODEL_USAGE_COUNTER.labels(model_id=model_id).inc() 
        resp = qa_chain({"query": input_text})
        sources = remove_source_duplicates(resp['source_documents'])
        create_pdf(resp['result'])
        if len(sources) != 0:
            q.put("\n*Sources:* \n")
            for source in sources:
                q.put("* " + str(source) + "\n")
        q.put(job_done)

    # Create a thread and start the function
    t = Thread(target=task)
    t.start()

    content = ""

    # Get each new token from the queue and yield for our generator
    while True:
        try:
            next_token = q.get(True, timeout=1)
            if next_token is job_done:
                break
            if isinstance(next_token, str):
                content += next_token
                yield next_token, content
        except Empty:
            continue

# A Queue is needed for Streaming implementation
q = Queue()

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

# LLM
llm = HuggingFaceTextGenInference(
    inference_server_url=INFERENCE_SERVER_URL,
    max_new_tokens=MAX_NEW_TOKENS,
    top_k=TOP_K,
    top_p=TOP_P,
    typical_p=TYPICAL_P,
    temperature=TEMPERATURE,
    repetition_penalty=REPETITION_PENALTY,
    streaming=True,
    verbose=False,
    callbacks=[QueueCallback(q)]
)

prompt_template = """
<s>[INST] <<SYS>>
{question}
Our goal is to generate $100,000 in sales within the first six months.
The proposal should include pricing strategy, marketing plan, and details on the unique features of our products. 
The proposal should include an overview of the product, its features and benefits, pricing, implementation timeline, support options, and testimonials from satisfied customers. 
Additionally, please include a marketing plan for promoting the product to the target audience.
Proposal should be minimum of 1000 lines and should be complete.
<</SYS>>

Context: {context}

[/INST]
"""

QA_CHAIN_PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"])

print(QA_CHAIN_PROMPT)


# qa_chain = RetrievalQA.from_chain_type(
#     llm,
#     retriever=rds.as_retriever(
#         search_type="stuff",
#         search_kwargs={"k": 4, "distance_threshold": 0.5}),
#     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
#     return_source_documents=True
#     )

qa_chain = RetrievalQA.from_chain_type(
    llm,
    chain_type='stuff',
    retriever=rds.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# Gradio implementation
def ask_llm(customer, product):
    query = f"Please generate a Sales Proposal for our {product} product to sell to {customer}."
    resp = qa_chain({"query": query })
    for next_token, content in stream(query):
        yield(content)

# Gradio implementation
css = "#output-container {font-size:0.8rem !important}"
with gr.Blocks(title="HatBot") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            customer_box = gr.Textbox(label="Customer", info="Enter the customer name")
            product_dropdown = gr.Dropdown(
             ["Red Hat OpenShift", "Red Hat OpenShift Data Science", "Red Hat AMQ Streams"], label="Product", info="Select the product to generate proposal"
            )
            with gr.Row():
                submit_button = gr.Button("Generate")
                clear_button = gr.ClearButton()

            gr.HTML(f"<div><span id='model_id'>Model: {model_id}</span></div>")
            radio = gr.Radio(["1", "2", "3", "4", "5"], label="Rate the model")

        with gr.Column(scale=2):
            output_answer = gr.Textbox(label="Project Proposal", readonly=True, lines=10, elem_id="output-container", scale=4, max_lines=10)
            source = gr.Textbox(label="Sources", elem_id="source-container", readonly=True, lines=5, scale=4, max_lines=5)
            download_button = gr.Button("Download as PDF")
        
    download_button.click(ask_llm, inputs=[customer_box, product_dropdown], outputs=[output_answer])
    submit_button.click(ask_llm, inputs=[customer_box, product_dropdown], outputs=[output_answer])
    clear_button.click(lambda: [None, None ,None , None], 
                       inputs=[], 
                       outputs=[customer_box,product_dropdown,output_answer,radio])

    
    @radio.input(inputs=radio, outputs=output_answer)
    def get_feedback(star):
        print("Rating: " + star)
        # Increment the counter based on the star rating received
        FEEDBACK_COUNTER.labels(stars=str(star), model_id=model_id).inc()
        return f"Received {star} star feedback. Thank you!"

if __name__ == "__main__":
    demo.queue().launch(
        server_name='0.0.0.0',
        share=False,
        favicon_path='./assets/robot-head.ico')