import os
import time
from collections.abc import Generator
from queue import Empty, Queue
from threading import Thread
import os
from markdown import markdown
from llm.llm_factory import LLMFactory, NVIDIA
import uuid
import gradio as gr
from prometheus_client import start_http_server
from dotenv import load_dotenv
from utils import config_loader
from scheduler.round_robin import RoundRobinScheduler
import pandas as pd
from utils.callback import QueueCallback
from generator.proposal_generator import FEEDBACK_COUNTER, ProposalGenerator

os.environ["REQUESTS_CA_BUNDLE"] = ""
# initialization
load_dotenv()
config_loader.init_config()
llm_factory = LLMFactory()
llm_factory.init_providers(config_loader.config)

global sched

# Parameters
APP_TITLE = os.getenv("APP_TITLE", "LLM RAG Demo")
TIMEOUT = int(os.getenv("TIMEOUT", 30))

# Start Prometheus metrics server
start_http_server(8000)

def create_scheduler():
    global sched
    provider_model_weight_list = config_loader.get_provider_model_weight_list()
    # initialize scheduler
    sched = RoundRobinScheduler(provider_model_weight_list)

create_scheduler()


def remove_source_duplicates(input_list):
    unique_list = []
    for item in input_list:
        if item.metadata["source"] not in unique_list:
            unique_list.append(item.metadata["source"])
    return unique_list

def get_llm(provider_model, que):
    
    callback = QueueCallback(que)
    provider_id, model_id = get_provider_model(provider_model)
    return llm_factory.get_llm(provider_id, model_id, callback)

# Gradio implementation
def generate_proposal(provider_model, company, product):
    que = Queue()
    session_id = str(uuid.uuid4())
    _, model_id = get_provider_model(provider_model)
    proposal_generator = ProposalGenerator(product, company, session_id)
    llm = get_llm(provider_model, que)
    
    download_link_html = f' <input type="hidden" id="pdf_file" name="pdf_file" value="/file={proposal_generator.get_pdf_file()}" />'
    for next_token, content in proposal_generator.generate_proposal(llm, model_id, que, product, company):
        yield content, download_link_html

    
def update_proposal(provider_model: str, company: str, product: str, old_proposal: str, user_query: str):

    que = Queue()
    session_id = str(uuid.uuid4())
    _, model_id = get_provider_model(provider_model)
    proposal_generator = ProposalGenerator(product, company, session_id)
    llm = get_llm(provider_model, que)
    
    download_link_html = f' <input type="hidden" id="pdf_file" name="pdf_file" value="/file={proposal_generator.get_pdf_file()}" />'
    for next_token, content in proposal_generator.update_proposal(llm, model_id, que, old_proposal, user_query):
        yield content, download_link_html
   

def get_provider_model(provider_model):
    if provider_model is None:
        return "", ""
    s = provider_model.split(": ")
    return s[0], s[1]


def is_provider_visible():
    return config_loader.config.type == "all"


def get_selected_provider():
    if config_loader.config.type == "round_robin":
        return sched.get_next()

    provider_list = config_loader.get_provider_model_weight_list()
    if len(provider_list) > 0:
        return provider_list[0]

    return None


# Gradio implementation
css = """
#output-container {font-size:0.8rem !important}

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
"""


def get_provider_list_as_df():
    provider_list = config_loader.get_provider_display_list()
    df = pd.DataFrame(provider_list)
    df = df.rename(
        columns={
            "provider_name": "Provider",
            "enabled": "Enabled",
            "url": "URL",
            "model_name": "Model",
        }
    )
    return df


with gr.Blocks(title="HatBot", css=css) as demo:
    with gr.Tab("Chatbot"):
        provider_model_list = config_loader.get_provider_model_list()
        provider_model_var = gr.State()
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
                    clear_button = gr.LogoutButton(value="Clear", icon=None)
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
            update_proposal,
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
            generate_proposal,
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

    with gr.Tab(
        label="Configuration", elem_classes="configuration-tab"
    ) as provider_tab:

        with gr.Accordion("Type"):
            type_dropdown = gr.Dropdown(
                ["round_robin", "all"],
                label="Type",
                value=config_loader.config.type,
                info="Select LLM providers based on type (round_robin,  all)",
            )

            update_type_btn = gr.Button("Submit", elem_classes="add_provider_bu")

            def update_type(type):
                config_loader.config.type = type
                create_scheduler()
                return {
                    type_dropdown: gr.Dropdown(
                        ["round_robin", "all"],
                        label="Type",
                        value=type,
                        info="Select LLM providers based on type (round_robin,  all)",
                    )
                }

            update_type_btn.click(
                update_type, inputs=[type_dropdown], outputs=[type_dropdown]
            ).success(None, outputs=[type_dropdown], js="window.location.reload()")

        with gr.Accordion("Providers"):
            df = get_provider_list_as_df()
            dataframe_ui = gr.Dataframe(value=df, interactive=False)
            add_btn = gr.Button("Add Provider", elem_classes="add_provider_bu")

        with gr.Accordion(label="Add Provider",visible=False) as add_provider_accordian:
            with gr.Blocks() as add_provider_table:
                with gr.Row():
                    with gr.Column():
                        llm_providers = LLMFactory.get_providers()
                        add_provider_dropdown = gr.Dropdown(
                            choices=llm_providers,
                            label="Providers",
                            info="Select the LLM provider",
                            elem_classes="configuration-tab-components",
                        )
                        add_model_text_box = gr.Textbox(
                            label="Model",
                            info="Enter the model name",
                            elem_classes="configuration-tab-components",
                        )
                        add_url_text_box = gr.Textbox(
                            label="URL",
                            info="Enter the URL",
                            elem_classes="configuration-tab-components",
                        )
                        add_api_key_text_box = gr.Textbox(
                            label="API Key",
                            info="Enter the API Key",
                            type="password",
                            elem_classes="configuration-tab-components",
                        )
                        enable_checkbox = gr.Checkbox(value=True, label="Enabled")
                    with gr.Column():
                        add_deployment_type_dropdown = gr.Dropdown(
                            ["Local", "Remote"],
                            label="Deployment type",
                            info="Model server deployment type",
                            visible=False,
                            elem_classes="configuration-tab-components",
                        )
                        add_weight_text_box = gr.Textbox(
                            label="Weight",
                            info="Enter the weight",
                            value=1,
                            elem_classes="configuration-tab-components",
                        )
                        add_param_temperature = gr.Textbox(
                            label="Temperature",
                            info="Enter the temperature",
                            value=0.01,
                            elem_classes="configuration-tab-components",
                        )
                        add_param_max_tokens = gr.Textbox(
                            label="Max Tokens",
                            info="Enter the maximum number of tokens",
                            value=512,
                            elem_classes="configuration-tab-components",
                        )

                        @add_provider_dropdown.change(
                            inputs=[add_provider_dropdown],
                            outputs=[add_deployment_type_dropdown],
                        )
                        def onChangeProviderSelection(provider_name):
                            visible = False
                            deployment_dropdown = gr.Dropdown(
                                ["Local", "Remote"],
                                label="Deployment type",
                                info="Model server deployment type",
                                visible=visible,
                                elem_classes="configuration-tab-components",
                            )
                            return {add_deployment_type_dropdown: deployment_dropdown}

                with gr.Row():
                    delete_button = gr.Button(
                        "Delete", elem_classes="add_provider_bu", visible=False
                    )
                    add_provider_submit_button = gr.Button(
                        "Add", elem_classes="add_provider_bu"
                    )

                    @delete_button.click(
                        inputs=[
                            add_provider_dropdown,
                            add_model_text_box,
                        ],
                        outputs=[
                            providers_dropdown,
                            dataframe_ui,
                            add_provider_dropdown,
                            add_model_text_box,
                            add_url_text_box,
                            add_weight_text_box,
                            add_param_temperature,
                            add_param_max_tokens,
                            enable_checkbox,
                            delete_button,
                            add_provider_submit_button,
                            add_provider_accordian,
                        ],
                    )
                    def delete_provider(provider, model):

                        config_loader.delete_provider(provider, model)
                        create_scheduler()
                        p_dropdown = gr.Dropdown(
                            interactive=True,
                            choices=provider_model_list,
                            label="Providers",
                            visible=provider_visible,
                        )

                        df = get_provider_list_as_df()
                        df_component = gr.Dataframe(
                            headers=["Provider", "Model", "URL", "Enabled"], value=df
                        )
                        add_p_dropdown = gr.Dropdown(
                            interactive=True,
                            choices=llm_providers,
                            label="Providers",
                            info="Select the LLM provider",
                            elem_classes="configuration-tab-components",
                        )
                        return {
                            providers_dropdown: p_dropdown,
                            dataframe_ui: df_component,
                            add_provider_dropdown: add_p_dropdown,
                            add_model_text_box: gr.Textbox(interactive=True, value=""),
                            add_url_text_box: "",
                            add_weight_text_box: 1,
                            add_param_temperature: 0.01,
                            add_param_max_tokens: 512,
                            enable_checkbox: gr.Checkbox(value=True, label="Enabled"),
                            delete_button: gr.Button(
                                "Delete", elem_classes="add_provider_bu", visible=False
                            ),
                            add_provider_submit_button: gr.Button(
                                "Add", elem_classes="add_provider_bu"
                            ),
                            add_provider_accordian: gr.Accordion(
                                visible=False
                            ),
                        }

            def df_select_callback(df: pd.DataFrame, evt: gr.SelectData):
                row_index = evt.index[0]
                col_index = evt.index[1]
                value = evt.value
                provider_name = df.iat[row_index, 0]
                model_name = df.iat[row_index, 1]

                print(f"\n(Row, Column) = ({row_index}, {col_index}). Value = {value}")
                print(f"\n(Provider, Model) = ({provider_name}, {provider_name}).")

                provider_cfg, model_cfg = config_loader.get_provider_model(
                    provider_name, model_name
                )

                if provider_cfg is None or model_cfg is None:
                    return None, None

                # TODO: Implement remote / local
                return {
                    add_provider_dropdown: gr.Dropdown(
                        interactive=False, value=provider_name
                    ),
                    add_model_text_box: gr.Textbox(interactive=False, value=model_name),
                    add_url_text_box: (
                        model_cfg.url if model_cfg.url else provider_cfg.url
                    ),
                    add_weight_text_box: model_cfg.weight,
                    add_param_temperature: (
                        model_cfg.params["temperature"]
                        if model_cfg.params and "temperature" in model_cfg.params
                        else ""
                    ),
                    add_param_max_tokens: (
                        model_cfg.params["max_new_tokens"]
                        if model_cfg.params and "max_new_tokens" in model_cfg.params
                        else ""
                    ),
                    enable_checkbox: (
                        provider_cfg.url
                        if model_cfg.url in (None, "")
                        else model_cfg.url
                    ),
                    delete_button: gr.Button(
                        "Delete", elem_classes="add_provider_bu", visible=True
                    ),
                    add_provider_submit_button: gr.Button(
                        "Update", elem_classes="add_provider_bu"
                    ),
                    add_provider_accordian: gr.Accordion(
                        label="Modify Provider",visible=True
                    ),
                }

            dataframe_ui.select(
                df_select_callback,
                inputs=[dataframe_ui],
                outputs=[
                    add_provider_dropdown,
                    add_model_text_box,
                    add_url_text_box,
                    add_weight_text_box,
                    add_param_temperature,
                    add_param_max_tokens,
                    enable_checkbox,
                    delete_button,
                    add_provider_submit_button,
                    add_provider_accordian,
                ],
            )

            def add_provider_bu_callback():
                provider_dropdown = gr.Dropdown(
                    interactive=True,
                    choices=llm_providers,
                    label="Providers",
                    info="Select the LLM provider",
                    elem_classes="configuration-tab-components",
                )
                return {
                    add_provider_dropdown: provider_dropdown,
                    add_model_text_box: gr.Textbox(interactive=True, value=""),
                    add_url_text_box: "",
                    add_weight_text_box: 1,
                    add_param_temperature: 0.01,
                    add_param_max_tokens: 512,
                    enable_checkbox: gr.Checkbox(value=True, label="Enabled"),
                    delete_button: gr.Button(
                        "Delete", elem_classes="add_provider_bu", visible=False
                    ),
                    add_provider_submit_button: gr.Button(
                        "Add", elem_classes="add_provider_bu"
                    ),
                    add_provider_accordian: gr.Accordion(
                        label="Add Provider",visible=True
                    ),
                }

            add_btn.click(
                add_provider_bu_callback,
                inputs=[],
                outputs=[
                    add_provider_dropdown,
                    add_model_text_box,
                    add_url_text_box,
                    add_weight_text_box,
                    add_param_temperature,
                    add_param_max_tokens,
                    enable_checkbox,
                    delete_button,
                    add_provider_submit_button,
                    add_provider_accordian,
                ],
            )

        def initialize(provider_model):
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
            df = get_provider_list_as_df()
            df_component = gr.Dataframe(
                headers=["Provider", "Model", "URL", "Enabled"], value=df
            )
            td = gr.Dropdown(
                ["round_robin", "all"],
                label="Type",
                value=config_loader.config.type,
                info="Select LLM providers based on type (round_robin,  all)",
            )
            return {
                providers_dropdown: p_dropdown,
                provider_model_var: provider_model,
                model_text: m,
                dataframe_ui: df_component,
                type_dropdown: td,
            }

        def validate_add_provider(
            provider_name, model_name, url, temperature, max_toxens, weight
        ):

            if not provider_name:
                raise gr.Error("Provider cannot be blank")

            if not model_name:
                raise gr.Error("Model cannot be blank")

            if not url:
                raise gr.Error("URL cannot be blank")

            try:
                int(weight)
            except ValueError:
                raise gr.Error("Weight should be Integer")

            try:
                int(max_toxens)
            except ValueError:
                raise gr.Error("Max tokens should be Integer")

            try:
                float(temperature)
            except ValueError:
                raise gr.Error("Temperature should be float")

            return

        def add_provider(
            selected_provider,
            provider_name,
            model_name,
            url,
            api_key,
            enabled,
            temperature,
            max_toxens,
            local_or_remote,
            weight,
        ):

            if local_or_remote == "Remote" and provider_name == NVIDIA:
                model_name = f"{local_or_remote}-{model_name}"

            params = [
                {
                    "name": "temperature",
                    "value": temperature,
                },
                {
                    "name": "max_new_toxens",
                    "value": max_toxens,
                },
            ]

            config_loader.add_provider_and_model(
                provider_name, model_name, url, api_key, enabled, params, int(weight)
            )
            llm_factory.init_providers(config_loader.config)
            provider_model_list = config_loader.get_provider_model_list()
            provider_visible = is_provider_visible()
            p_dropdown = gr.Dropdown(
                interactive=True,
                choices=provider_model_list,
                label="Providers",
                visible=provider_visible,
                value=selected_provider,
            )
            gr.Info("Provider added successfully!")
            df = get_provider_list_as_df()
            df_component = gr.Dataframe(
                headers=["Provider", "Model", "URL", "Enabled"], value=df
            )
            create_scheduler()
            return {
                providers_dropdown: p_dropdown,
                add_provider_dropdown: None,
                add_model_text_box: None,
                add_url_text_box: None,
                add_api_key_text_box: None,
                add_param_temperature: 0.01,
                add_param_max_tokens: 512,
                add_deployment_type_dropdown: None,
                add_weight_text_box: 1,
                dataframe_ui: df_component,
                delete_button: gr.Button(
                    "Delete", elem_classes="add_provider_bu", visible=False
                ),
                add_provider_submit_button: gr.Button(
                    "Add", elem_classes="add_provider_bu"
                ),
                add_provider_accordian: gr.Accordion(
                        label="Modify Provider",visible=False
                ),
            }

        add_provider_submit_button.click(
            validate_add_provider,
            inputs=[
                add_provider_dropdown,
                add_model_text_box,
                add_url_text_box,
                add_param_temperature,
                add_param_max_tokens,
                add_weight_text_box,
            ],
        ).success(
            add_provider,
            inputs=[
                provider_model_var,
                add_provider_dropdown,
                add_model_text_box,
                add_url_text_box,
                add_api_key_text_box,
                enable_checkbox,
                add_param_temperature,
                add_param_max_tokens,
                add_deployment_type_dropdown,
                add_weight_text_box,
            ],
            outputs=[
                providers_dropdown,
                add_provider_dropdown,
                add_model_text_box,
                add_url_text_box,
                add_api_key_text_box,
                add_param_temperature,
                add_param_max_tokens,
                add_deployment_type_dropdown,
                add_weight_text_box,
                dataframe_ui,
                delete_button,
                add_provider_submit_button,
                add_provider_accordian,
            ],
        )

        demo.load(
            initialize,
            inputs=[provider_model_var],
            outputs=[providers_dropdown, provider_model_var, model_text, dataframe_ui, type_dropdown],
        )


if __name__ == "__main__":
    os.environ.pop("LANGCHAIN_TRACING_V2", None)

    demo.queue().launch(
        server_name="0.0.0.0",
        share=False,
        favicon_path="./assets/robot-head.ico",
        allowed_paths=["assets"],
    )
