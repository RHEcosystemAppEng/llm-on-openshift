import time

import threading
import os
from generator.template import VECTOR_DB_QUERY_TEMPLATE
import pdfkit
from prometheus_client import Gauge, Counter
from langchain.chains import LLMChain
from collections.abc import Generator
from queue import Empty, Queue
from threading import Thread
from markdown import markdown
from generator.query_helper import QueryHelper
from langchain.llms.base import LLM


PDF_FILE_DIR = "proposal-docs"

# Create metric
FEEDBACK_COUNTER = Counter(
    "feedback_stars", "Number of feedbacks by stars", ["stars", "model_id"]
)
MODEL_USAGE_COUNTER = Counter(
    "model_usage", "Number of times a model was used", ["model_id"]
)
REQUEST_TIME = Gauge(
    "request_duration_seconds", "Time spent processing a request", ["model_id"]
)

class ProposalGenerator:
  """Generator project proposal for a given product addressed to a company"""

  def __init__(self, product: str, company: str, session_id: str):
    self.product = product
    self.company = company
    self.lock = threading.Lock()
    self.session_id = session_id

  def generate_proposal(self, llm: LLM, model_id: str , que: Queue, product: str, company: str) -> Generator:

    query_helper = QueryHelper()
    query = VECTOR_DB_QUERY_TEMPLATE.format(product = product, company = company)

    proposal_chain = query_helper.get_qa_chain(llm)

    model_input = {'query': query}

    return self.stream(proposal_chain, que, model_input, model_id)
  
  def update_proposal(self, llm: LLM, model_id: str , que: Queue, old_proposal: str, user_query: str) -> Generator:

    query_helper = QueryHelper()

    proposal_chain = query_helper.get_update_proposal_chain(llm)
    
    model_input = {'old_proposal': old_proposal, 'user_query': user_query}

    return self.stream(proposal_chain, que, model_input, model_id)
  

  def get_pdf_file(self)->str:
    return os.path.join("./assets", PDF_FILE_DIR, f"proposal-{self.session_id}.pdf")


  def create_pdf(self, text: str):
    try:
        output_filename = self.get_pdf_file()
        html_text = markdown(text, output_format="html4")
        pdfkit.from_string(html_text, output_filename)
    except Exception as e:
        print(e)

  # Function to initialize all star ratings to 0
  def initialize_feedback_counters(self, model_id: str):
    for star in range(1, 6):  # For star ratings 1 to 5
        FEEDBACK_COUNTER.labels(stars=str(star), model_id=model_id).inc(0)

  def remove_source_duplicates(self, input_list):
    unique_list = []
    for item in input_list:
        if item.metadata["source"] not in unique_list:
            unique_list.append(item.metadata["source"])
    return unique_list
  
  def stream(self, chain: LLMChain, que: Queue, model_input: dict, model_id: str) -> Generator:
    # Create a Queue
    job_done = object()
    # Create a function to call - this will run in a thread
    def task():
        MODEL_USAGE_COUNTER.labels(model_id=model_id).inc()
        # Call this function at the start of your application
        self.initialize_feedback_counters(model_id)
        with self.lock:
            start_time = (
                time.perf_counter()
            )  # start and end time to get the precise timing of the request
            try:
                resp = chain.invoke(input=model_input)
                end_time = time.perf_counter()
                sources = self.remove_source_duplicates(resp["source_documents"])
                REQUEST_TIME.labels(model_id=model_id).set(end_time - start_time)
                self.create_pdf(resp["result"])
                if len(sources) != 0:
                    que.put("\n*Sources:* \n")
                    for source in sources:
                        que.put("* " + str(source) + "\n")
            except Exception as e:
                print(e)
                que.put("Error executing request. Contact the administrator.")

            que.put(job_done)

    # Create a thread and start the function
    t = Thread(target=task)
    t.start()

    content = ""

    # Get each new token from the queue and yield for our generator
    while True:
        try:
            next_token = que.get(True, timeout=100)
            if next_token is job_done:
                break
            if isinstance(next_token, str):
                content += next_token
                yield next_token, content
        except Empty:
            continue



    