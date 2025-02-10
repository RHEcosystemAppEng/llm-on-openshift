import pandas as pd
from llm.llm_factory import LLMFactory, NVIDIA
from ui.util import create_scheduler, get_llm_factory, get_provider_list_as_df, get_provider_model, get_selected_provider, is_provider_visible
from utils import config_loader

class ConfigurationTab:
    def __init__(self, tab, parent):
        self.tab = tab
        self.parent = parent
    def generate(self, gr, provider_model_var):
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
                            choices=config_loader.get_provider_model_list(),
                            label="Providers",
                            visible=is_provider_visible(),
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

        # def initialize(provider_model):
        #     if provider_model is None:
        #         provider_model_tuple = get_selected_provider()
        #         if provider_model_tuple is not None:
        #             provider_model = provider_model_tuple[0]
        #     print(provider_model)
        #     provider_id, model_id = get_provider_model(provider_model)
        #     provider_visible = is_provider_visible()
        #     provider_model_list = config_loader.get_provider_model_list()
        #     p_dropdown = gr.Dropdown(
        #         choices=provider_model_list,
        #         label="Providers",
        #         visible=provider_visible,
        #         value=provider_model,
        #     )
        #     m = f"<div><span id='model_id'>Model: {model_id}</span></div>"
        #     df = get_provider_list_as_df()
        #     df_component = gr.Dataframe(
        #         headers=["Provider", "Model", "URL", "Enabled"], value=df
        #     )
        #     td = gr.Dropdown(
        #         ["round_robin", "all"],
        #         label="Type",
        #         value=config_loader.config.type,
        #         info="Select LLM providers based on type (round_robin,  all)",
        #     )
        #     return {
        #         providers_dropdown: p_dropdown,
        #         provider_model_var: provider_model,
        #         model_text: m,
        #         dataframe_ui: df_component,
        #         type_dropdown: td,
        #     }

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
            get_llm_factory().init_providers(config_loader.config)
            gr.Info("Provider added successfully!")
            df = get_provider_list_as_df()
            df_component = gr.Dataframe(
                headers=["Provider", "Model", "URL", "Enabled"], value=df
            )
            create_scheduler()
            return {
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


        def initialize(provider_model):
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
                provider_model_var: provider_model,
                dataframe_ui: df_component,
                type_dropdown: td,
            }

        self.tab.select(
            initialize, inputs=[provider_model_var], outputs=[provider_model_var, dataframe_ui, type_dropdown]
        )
