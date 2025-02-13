
from queue import Queue
import uuid
from generator.proposal_generator import ProposalGenerator
from ui.util import get_llm, get_provider_model, get_selected_provider, is_provider_visible
from queue import Queue
import uuid
from utils import config_loader
#from scheduler.round_robin import RoundRobinScheduler
#import pandas as pd
from generator.proposal_generator import FEEDBACK_COUNTER, ProposalGenerator

class ProposalGenerationTab:
    def __init__(self, tab, parent):
        self.tab = tab
        self.parent =  parent

    # Gradio implementation
    def generate_proposal(self, provider_model, company, product):
        que = Queue()
        session_id = str(uuid.uuid4())
        _, model_id = get_provider_model(provider_model)
        proposal_generator = ProposalGenerator(session_id)
        llm = get_llm(provider_model, que)
        
        download_link_html = f' <input type="hidden" id="pdf_file" name="pdf_file" value="/file={proposal_generator.get_pdf_file()}" />'
        for next_token, content in proposal_generator.generate_proposal(llm, model_id, que, product, company):
            yield content, download_link_html

        
    def update_proposal(self, provider_model: str, company: str, product: str, old_proposal: str, user_query: str):

        que = Queue()
        session_id = str(uuid.uuid4())
        _, model_id = get_provider_model(provider_model)
        proposal_generator = ProposalGenerator(session_id)
        llm = get_llm(provider_model, que)
        
        download_link_html = f' <input type="hidden" id="pdf_file" name="pdf_file" value="/file={proposal_generator.get_pdf_file()}" />'
        for next_token, content in proposal_generator.update_proposal(llm, model_id, que, old_proposal, user_query):
            yield content, download_link_html


    def generate(self, gr, provider_model_var):
        provider_model_list = config_loader.get_provider_model_list()
        provider_visible = is_provider_visible()
    
        with gr.Row():
            with gr.Column(scale=1):
                providers_dropdown = gr.Dropdown(
                    label="Providers", choices=provider_model_list
                )
                customer_box = gr.Textbox(
                    label="Customer", info="Enter the customer name"
                )
                product_text_box = gr.Textbox(
                    label="Product", info="Enter the Red Hat product name"
                )
                with gr.Row():
                    submit_button = gr.Button("Generate")
                    clear_button = gr.ClearButton(value="Clear", icon=None)
                model_text = gr.HTML(visible=~provider_visible)

                def update_models(selected_provider, provider_model):
                    provider_id, model_id = get_provider_model(selected_provider)
                    m = f"<div><span id='model_id'>Model: {model_id}</span></div>"
                    return {provider_model_var: selected_provider, model_text: m}

                providers_dropdown.input(
                    update_models,
                    inputs=[providers_dropdown, provider_model_var],
                    outputs=[provider_model_var, model_text],
                )
                radio = gr.Radio(["1", "2", "3", "4", "5"], label="Rate the model")
                output_rating = gr.Textbox(
                    elem_id="source-container", interactive=True, label="Rating"
                )
                def proposal_gen_tab_selected(provider_model):
                    print("Proposal tab selected")
                    if provider_model is None:
                        provider_model_tuple = get_selected_provider()
                        if provider_model_tuple is not None:
                            provider_model = provider_model_tuple[0]
                    print(provider_model)
                    provider_id, model_id = get_provider_model(provider_model)
                    provider_visible = is_provider_visible()
                    provider_model_list = config_loader.get_provider_model_list()
                    p_dropdown = gr.Dropdown(
                        choices=provider_model_list,
                        label="Providers",
                        visible=provider_visible,
                        value=provider_model,
                    )
                    m = f"<div><span id='model_id'>Model: {model_id}</span></div>"
                     
                    return {
                        providers_dropdown: p_dropdown,
                        model_text: m
                    }

                self.tab.select(
                    proposal_gen_tab_selected,
                    inputs=[provider_model_var],
                    outputs=[providers_dropdown, model_text]
        )

            with gr.Column(scale=2):
                lines = 19
                if provider_visible:
                    lines = 26
                output_answer = gr.Textbox(
                    label="Project Proposal",
                    interactive=True,
                    lines=lines,
                    elem_id="output-container",
                    scale=4,
                    max_lines=lines,
                )

                input_update_proposal = gr.Textbox(
                    label='Update proposal', placeholder='Make the proposal more detailed.', visible=False
                )
                update_proposal_button = gr.Button('Update proposal', visible=False)
                download_button = gr.Button("Download as PDF", visible=False)
                download_link_html = gr.HTML(visible=False)


        download_button.click(
            None,
            [],
            [],
            js="() => window.open(document.getElementById('pdf_file').value, '_blank')",
        )
        def validate_update_proposal_input(text):
            if not text:
                raise gr.Error('Update proposal cannot be blank')
        update_proposal_button.click(
            validate_update_proposal_input,
            inputs=[input_update_proposal]
        ).success(
            self.update_proposal,
            inputs=[providers_dropdown, customer_box, product_text_box, output_answer, input_update_proposal],
            outputs=[output_answer, download_link_html]
        )

        def make_visable_chat_with_pdf():
            return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

        def validate_generate_input(provider, customer, product):

            if not provider:
                raise gr.Error("Provider/Model cannot be blank")

            if not customer:
                raise gr.Error("Customer cannot be blank")

            if not product:
                raise gr.Error("Product cannot be blank")

            return

        submit_button.click(
            validate_generate_input,
            inputs=[providers_dropdown, customer_box, product_text_box],
        ).success(
            self.generate_proposal,
            inputs=[providers_dropdown, customer_box, product_text_box],
            outputs=[output_answer, download_link_html],
        ).success(
            make_visable_chat_with_pdf,
            inputs=None,
            outputs=[input_update_proposal, download_button, update_proposal_button]
        )
        clear_button.click(
            lambda: [None, None, None, None, None],
            inputs=[],
            outputs=[
                customer_box,
                product_text_box,
                output_answer,
                radio,
                output_rating,
            ],
        )

        @radio.input(inputs=[radio, provider_model_var], outputs=output_rating)
        def get_feedback(star, provider_model):
            provider_id, model_id = get_provider_model(provider_model)
            print(f"Model: {provider_model}, Rating: {star}")
            # Increment the counter based on the star rating received
            FEEDBACK_COUNTER.labels(stars=str(star), model_id=model_id).inc()
            return f"Received {star} star feedback. Thank you!"

        self.parent.load(
            proposal_gen_tab_selected,
            inputs=[provider_model_var],
            outputs=[providers_dropdown, model_text])