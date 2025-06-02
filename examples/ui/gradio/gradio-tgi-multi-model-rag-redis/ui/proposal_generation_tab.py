import os
import uuid
from queue import Queue

from generator.proposal_generator import FEEDBACK_COUNTER, ProposalGenerator
from ui.util import (
    get_llm,
    get_provider_model,
    get_selected_provider,
    is_provider_visible,
)
from utils import config_loader


class ProposalGenerationTab:
    def __init__(self, tab, parent):
        self.tab = tab
        self.parent = parent

    # Gradio implementation
    def generate_proposal(self, provider_model, company, product):
        que = Queue()
        session_id = str(uuid.uuid4())
        _, model_id = get_provider_model(provider_model)
        proposal_generator = ProposalGenerator(session_id)
        llm = get_llm(provider_model, True, que)

        for _, content in proposal_generator.generate_proposal(
            llm, model_id, que, product, company
        ):
            yield content, None

        pdf_path = proposal_generator.get_pdf_file()
        yield content, pdf_path

    def update_proposal(
        self,
        provider_model: str,
        old_proposal: str,
        user_query: str,
    ):
        que = Queue()
        session_id = str(uuid.uuid4())
        _, model_id = get_provider_model(provider_model)
        proposal_generator = ProposalGenerator(session_id)
        llm = get_llm(provider_model, True, que)

        for _, content in proposal_generator.update_proposal(
            llm, model_id, que, old_proposal, user_query
        ):
            yield content, None

        pdf_path = proposal_generator.get_pdf_file()
        yield content, pdf_path

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
                model_text = gr.HTML(visible=not provider_visible)

                def update_models(selected_provider, provider_model):
                    _, model_id = get_provider_model(selected_provider)
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
                    if provider_model is None:
                        provider_model_tuple = get_selected_provider()
                        if provider_model_tuple is not None:
                            provider_model = provider_model_tuple[0]

                    _, model_id = get_provider_model(provider_model)
                    provider_visible = is_provider_visible()
                    provider_model_list = config_loader.get_provider_model_list()
                    p_dropdown = gr.Dropdown(
                        choices=provider_model_list,
                        label="Providers",
                        visible=provider_visible,
                        value=provider_model,
                    )
                    m = f"<div><span id='model_id'>Model: {model_id}</span></div>"

                    return {providers_dropdown: p_dropdown, model_text: m}

                self.tab.select(
                    proposal_gen_tab_selected,
                    inputs=[provider_model_var],
                    outputs=[providers_dropdown, model_text],
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
                    label="Update proposal",
                    placeholder="Make the proposal more detailed.",
                    visible=False,
                )
                update_proposal_button = gr.Button("Update proposal", visible=False)
                download_file = gr.File(label="Download as PDF", visible=False)

        def validate_update_proposal_input(text):
            if not text:
                raise gr.Error("Update proposal cannot be blank")

        def make_visible_chat_with_pdf():
            return (
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
            )

        update_proposal_button.click(
            validate_update_proposal_input, inputs=[input_update_proposal]
        ).success(
            lambda: (gr.update(visible=False)),
            inputs=None,
            outputs=[download_file],
        ).success(
            self.update_proposal,
            inputs=[
                providers_dropdown,
                output_answer,
                input_update_proposal,
            ],
            outputs=[output_answer, download_file],
        ).success(
            make_visible_chat_with_pdf,
            inputs=None,
            outputs=[input_update_proposal, download_file, update_proposal_button],
        )

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
            lambda: (
                None,
                gr.update(value="", visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            ),
            inputs=None,
            outputs=[
                output_answer,
                input_update_proposal,
                update_proposal_button,
                download_file,
            ],
        ).success(
            self.generate_proposal,
            inputs=[providers_dropdown, customer_box, product_text_box],
            outputs=[output_answer, download_file],
        ).success(
            make_visible_chat_with_pdf,
            inputs=None,
            outputs=[input_update_proposal, download_file, update_proposal_button],
        )

        clear_button.click(
            lambda: [
                None,
                None,
                None,
                None,
                None,
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            ],
            inputs=[],
            outputs=[
                customer_box,
                product_text_box,
                output_answer,
                radio,
                output_rating,
                input_update_proposal,
                update_proposal_button,
                download_file,
            ],
        )

        @radio.input(inputs=[radio, providers_dropdown], outputs=output_rating)
        def get_feedback(star, provider_model):
            _, model_id = get_provider_model(provider_model)
            print(f"Model: {provider_model}, Rating: {star}")
            # Increment the counter based on the star rating received
            FEEDBACK_COUNTER.labels(stars=str(star), model_id=model_id).inc()
            return f"Received {star} star feedback. Thank you!"

        self.parent.load(
            proposal_gen_tab_selected,
            inputs=[provider_model_var],
            outputs=[providers_dropdown, model_text],
        )
