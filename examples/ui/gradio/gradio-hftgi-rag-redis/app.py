import os
import time
from collections.abc import Generator
from queue import Empty
from threading import Thread
import os
from markdown import markdown
from llm.llm_factory import LLMFactory
import pdfkit
import uuid
import threading
import gradio as gr
from prometheus_client import Gauge, start_http_server, Counter
from dotenv import load_dotenv
from utils import config_loader
import llm.query_helper as QueryHelper
from scheduler.round_robin import RoundRobinScheduler

# test data to schedule
# data = [ ("A", 6), 
#         ("B", 5), 
#         ("C", 4), 
#         ("D", 3), 
#         ("E", 2),
#         ("F", 1),
#       ]


# initialization
load_dotenv()
config_loader.init_config()
llm_factory = LLMFactory()
llm_factory.init_providers(config_loader.config)
provider_model_weightage_list = config_loader.get_provider_model_weightage_list()
# initialize scheduler
sched = RoundRobinScheduler(provider_model_weightage_list)


# Parameters
APP_TITLE = os.getenv('APP_TITLE', 'Talk with your documentation')
PDF_FILE_DIR = "proposal-docs"
TIMEOUT = int(os.getenv('TIMEOUT', 30))

# Start Prometheus metrics server
start_http_server(8000)

# Create metric
FEEDBACK_COUNTER = Counter("feedback_stars", "Number of feedbacks by stars", ["stars", "model_id"])
MODEL_USAGE_COUNTER = Counter('model_usage', 'Number of times a model was used', ['model_id'])
REQUEST_TIME = Gauge('request_duration_seconds', 'Time spent processing a request', ['model_id'])


# PDF Generation
def get_pdf_file(session_id):
    return os.path.join("./assets", PDF_FILE_DIR, f"proposal-{session_id}.pdf")

def create_pdf(text, session_id):
    try:
        output_filename = get_pdf_file(session_id)
        html_text = markdown(text, output_format='html4')
        pdfkit.from_string(html_text, output_filename)
    except Exception as e:
        print(e)

# Function to initialize all star ratings to 0
def initialize_feedback_counters(model_id):
    for star in range(1, 6):  # For star ratings 1 to 5
        FEEDBACK_COUNTER.labels(stars=str(star), model_id=model_id).inc(0)

def remove_source_duplicates(input_list):
    unique_list = []
    for item in input_list:
        if item.metadata['source'] not in unique_list:
            unique_list.append(item.metadata['source'])
    return unique_list

lock = threading.Lock()

def stream(llm, q, input_text, session_id, model_id) -> Generator:
    # Create a Queue
    job_done = object()

    qa_chain = QueryHelper.get_qa_chain(llm)

    # Create a function to call - this will run in a thread
    def task():
        MODEL_USAGE_COUNTER.labels(model_id=model_id).inc() 
        # Call this function at the start of your application
        initialize_feedback_counters(model_id)
        with lock:
            start_time = time.perf_counter() # start and end time to get the precise timing of the request
            try:
                resp = qa_chain({"query": input_text})
                end_time = time.perf_counter()
                sources = remove_source_duplicates(resp['source_documents'])
                REQUEST_TIME.labels(model_id=model_id).set(end_time - start_time)
                create_pdf(resp['result'], session_id)
                if len(sources) != 0:
                    q.put("\n*Sources:* \n")
                    for source in sources:
                        q.put("* " + str(source) + "\n")
            except Exception as e:
                print(e)
                q.put("Error executing request. Contact the administrator.")

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

# Gradio implementation
def ask_llm(provider_model, customer, product):
    session_id = str(uuid.uuid4())
    provider_id, model_id = get_provider_model(provider_model)
    llm, q = llm_factory.get_llm(provider_id, model_id)

    query = f"Generate a Sales Proposal for the product '{product}' to sell to company '{customer}' that includes overview, features, benefits, and support options?"
    for next_token, content in stream(llm, q, query, session_id, model_id):
        # Generate the download link HTML
        download_link_html = f' <input type="hidden" id="pdf_file" name="pdf_file" value="/file={get_pdf_file(session_id)}" />'
        yield content, download_link_html

def get_provider_model(provider_model):
    if provider_model is None:
        return "",""
    s = provider_model.split(": ")
    return s[0], s[1]

def is_provider_visible():
    return config_loader.config.type == "all"

def get_selected_provider():
    if config_loader.config.type == "round_robin":
        return sched.get_next()
    
    provider_list = config_loader.get_provider_model_weightage_list()
    if len(provider_list) > 0:
        return provider_list[0]
    
    return None
        
# Gradio implementation
css = "#output-container {font-size:0.8rem !important}"
with gr.Blocks(title="HatBot") as demo:
    provider_model_list = config_loader.get_provider_model_list()
    provider_model_var = gr.State()
    provider_visible = is_provider_visible()

    with gr.Row():
        with gr.Column(scale=1):
            providers_dropdown = gr.Dropdown(label="Providers", choices=provider_model_list)  
            customer_box = gr.Textbox(label="Customer", info="Enter the customer name")
            product_dropdown = gr.Dropdown(
             ["Red Hat OpenShift", "Red Hat OpenShift Data Science", "Red Hat AMQ Streams"], label="Product", info="Select the product to generate proposal"
            )
            with gr.Row():
                submit_button = gr.Button("Generate")
                clear_button = gr.LogoutButton(value="Clear", icon=None)
            model_text = gr.HTML(visible=~provider_visible)

            def update_models(selected_provider, provider_model):
                provider_id, model_id = get_provider_model(selected_provider)
                m=f"<div><span id='model_id'>Model: {model_id}</span></div>"
                return {
                    provider_model_var: selected_provider,
                    model_text: m
                }
            
            providers_dropdown.input(update_models, inputs=[providers_dropdown, provider_model_var], outputs=[provider_model_var,model_text])
            radio = gr.Radio(["1", "2", "3", "4", "5"], label="Rate the model")
            output_rating = gr.Textbox(elem_id="source-container", interactive=True, label="Rating")

        with gr.Column(scale=2):
            lines = 19
            if provider_visible:
                lines = 26
            output_answer = gr.Textbox(label="Project Proposal", interactive=True, lines=lines, elem_id="output-container", scale=4, max_lines=lines)
            download_button = gr.Button("Download as PDF")
            download_link_html=gr.HTML(visible=False)

    download_button.click(None, [], [], js="() => window.open(document.getElementById('pdf_file').value, '_blank')")
    submit_button.click(ask_llm, inputs=[providers_dropdown, 
                                         customer_box, 
                                         product_dropdown], 
                                outputs=[output_answer,
                                         download_link_html])
    clear_button.click(lambda: [None, None ,None , None, None], 
                       inputs=[], 
                       outputs=[customer_box,
                                product_dropdown,
                                output_answer,
                                radio,
                                output_rating])

    @radio.input(inputs=[radio, provider_model_var], outputs=output_rating)
    def get_feedback(star, provider_model):
        provider_id, model_id = get_provider_model(provider_model)
        print(f"Model: {provider_model}, Rating: {star}")
        # Increment the counter based on the star rating received
        FEEDBACK_COUNTER.labels(stars=str(star), model_id=model_id).inc()
        return f"Received {star} star feedback. Thank you!"

    def initialize(provider_model):
        if provider_model is None:
            provider_model_tuple = get_selected_provider()
            if provider_model_tuple is not None:
                provider_model = provider_model_tuple[0]
        print(provider_model)
        provider_id, model_id = get_provider_model(provider_model)
        provider_visible = is_provider_visible()
        p_dropdown = gr.Dropdown(choices=provider_model_list, label="Providers", visible=provider_visible, value=provider_model)
        m=f"<div><span id='model_id'>Model: {model_id}</span></div>"
        return {
            providers_dropdown: p_dropdown,
            provider_model_var: provider_model,
            model_text: m
        }
    demo.load(initialize, inputs=[provider_model_var], outputs=[providers_dropdown,provider_model_var,model_text])

if __name__ == "__main__":
    demo.queue().launch(
        server_name='0.0.0.0',
        share=False,
        favicon_path='./assets/robot-head.ico',
        allowed_paths=["assets"])