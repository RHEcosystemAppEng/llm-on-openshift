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


def get_user_id():
    return str(uuid.uuid4())


class QuestionAndAnswerTab:
    def __init__(self, tab, parent):
        self.tab = tab
        self.parent = parent

    def generate(self, gr, provider_model_var):
        provider_model_list = config_loader.get_provider_model_list()
        provider_visible = is_provider_visible()

        with gr.Row():
            with gr.Column(scale=1):
                providers_dropdown = gr.Dropdown(
                    label="Providers", choices=provider_model_list
                )
                model_text = gr.HTML(visible=not provider_visible)

                def update_models(selected_provider):
                    _, model_id = get_provider_model(selected_provider)
                    m = f"<div><span id='model_id'>Model: {model_id}</span></div>"
                    return {provider_model_var: selected_provider, model_text: m}

                providers_dropdown.input(
                    update_models,
                    inputs=[providers_dropdown],
                    outputs=[provider_model_var, model_text],
                )
                radio = gr.Radio(["1", "2", "3", "4", "5"], label="Rate the model")
                output_rating = gr.Textbox(
                    elem_id="source-container", interactive=True, label="Rating"
                )

                def q_and_a_tab_selected(provider_model):
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
                    q_and_a_tab_selected,
                    inputs=[provider_model_var],
                    outputs=[providers_dropdown, model_text],
                )

            with gr.Column(scale=2):
                session_id_text_box = gr.Textbox(value=get_user_id(), visible=False)

                def set_user_response(user_message: str, chat_history) -> tuple:
                    chat_history.append({"role": "user", "content": user_message})
                    chat_history.append({"role": "assistant", "content": ""})
                    return "", chat_history

                def generate_response(
                    provider_model: str,
                    session_id: str,
                    question: str,
                    chat_history: list,
                ):
                    que = Queue()
                    _, model_id = get_provider_model(provider_model)
                    proposal_generator = ProposalGenerator(session_id)
                    llm = get_llm(provider_model, False, que)
                    answer = proposal_generator.get_answer(
                        llm, model_id, que, question, chat_history
                    )
                    chat_history[-1]["content"] = answer
                    return "", chat_history

                chatbot = gr.Chatbot(label="Ask LLM", type="messages")
                msg = gr.Textbox(label="User query")
                clear = gr.ClearButton()

                msg.submit(
                    set_user_response, [msg, chatbot], [msg, chatbot], queue=False
                ).then(
                    generate_response,
                    inputs=[providers_dropdown, session_id_text_box, msg, chatbot],
                    outputs=[msg, chatbot],
                )
                clear.click(
                    lambda: [None, None, None, None],
                    inputs=[],
                    outputs=[chatbot, radio, output_rating, msg],
                    queue=False,
                )

        @radio.input(inputs=[radio, providers_dropdown], outputs=output_rating)
        def get_feedback(star, provider_model):
            _, model_id = get_provider_model(provider_model)
            print(f"Model: {provider_model}, Rating: {star}")
            # Increment the counter based on the star rating received
            FEEDBACK_COUNTER.labels(stars=str(star), model_id=model_id).inc()
            return f"Received {star} star feedback. Thank you!"

        self.parent.load(
            q_and_a_tab_selected,
            inputs=[provider_model_var],
            outputs=[providers_dropdown, model_text],
        )
