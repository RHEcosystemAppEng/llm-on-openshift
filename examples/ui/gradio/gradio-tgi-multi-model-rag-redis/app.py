import os

import gradio as gr
from dotenv import load_dotenv
from prometheus_client import start_http_server

from ui.configuration_tab import ConfigurationTab
from ui.proposal_generation_tab import ProposalGenerationTab
from ui.question_answer_tab import QuestionAndAnswerTab
from ui.util import create_scheduler

load_dotenv()
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Parameters
APP_TITLE = os.getenv("APP_TITLE", "LLM RAG Demo")
TIMEOUT = int(os.getenv("TIMEOUT", 30))

# Start Prometheus metrics server
start_http_server(8000)

create_scheduler()

# Gradio implementation
css = """
#output-container {font-size:0.8rem !important;}

.width_200 {
     width: 200px;
}

.width_300 {
     width: 300px;
}

.width_100 {
     width: 100px;
}

.width_50 {
     width: 50px;
}

.add_provider_bu {
    max-width: 200px;
}

.markdown-output {
    height: 800px;
}

"""

with gr.Blocks(title=APP_TITLE, css=css) as demo:
    provider_model_var = gr.State()
    with gr.Tab("Proposal Generator Chatbot", id=1) as proposal_tab:
        proposal_generator_tab = ProposalGenerationTab(proposal_tab, demo)
        proposal_generator_tab.generate(gr, provider_model_var)

    with gr.Tab(label="Q & A Chatbot", id=2) as q_a_tab:
        configuration_tab = QuestionAndAnswerTab(q_a_tab, demo)
        configuration_tab.generate(gr, provider_model_var)

    with gr.Tab(
        label="Configuration", id=3, elem_classes="configuration-tab"
    ) as provider_tab:
        configuration_tab = ConfigurationTab(provider_tab, demo)
        configuration_tab.generate(gr, provider_model_var)

if __name__ == "__main__":
    os.environ.pop("LANGCHAIN_TRACING_V2", None)

    demo.queue().launch(
        server_name="0.0.0.0",
        share=False,
        favicon_path="./assets/robot-head.ico",
        allowed_paths=["assets"],
    )
