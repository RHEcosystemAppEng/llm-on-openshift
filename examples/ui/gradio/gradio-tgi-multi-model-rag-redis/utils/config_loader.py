"""Configuration loader."""

import os
import traceback
from io import TextIOBase

import yaml

from utils.config import Config, ModelConfig, ProviderConfig

config = None
llm_config = None


def load_config_from_stream(stream: TextIOBase) -> Config:
    """Load configuration from a YAML stream."""
    data = yaml.safe_load(stream)
    c = Config(data)
    return c


def init_config() -> None:
    """Load configuration from a YAML file."""
    global config
    global llm_config

    config_file = os.environ.get("CONFIG_FILE", "config.yaml")
    try:
        with open(config_file, "r") as f:
            config = load_config_from_stream(f)
            llm_config = config.llm_providers

    except Exception as e:
        print(f"Failed to load config file {config_file}: {e!s}")
        print(traceback.format_exc())
        raise e


def get_default_model():
    if config.type == "default":
        return config.default_model


def get_default_provider():
    if config.type == "default":
        return config.default_provider


def get_provider_model_dict():
    provider_model_list = {}
    if config.type == "default":
        provider_model_list[get_default_provider()] = [get_default_model()]
    else:
        providers = config.llm_providers.providers
        for provider_name in providers:
            provider_cfg = llm_config.providers.get(provider_name)
            if provider_cfg and provider_cfg.enabled:
                models = []
                for model_name in provider_cfg.models:
                    if provider_cfg.models[model_name].enabled:
                        models.append(model_name)
                if len(models) > 0:
                    provider_model_list[provider_name] = models
    return provider_model_list


def get_provider_model_weight_list():
    provider_model_list = []
    if config.type == "default":
        provider_model_name = get_default_provider_model()
        provider_model_list.append((provider_model_name, 1))
    else:
        providers = config.llm_providers.providers
        for provider_name in providers:
            provider_cfg = llm_config.providers.get(provider_name)
            if provider_cfg and provider_cfg.enabled:
                for model_name in provider_cfg.models:
                    model_cfg = provider_cfg.models[model_name]
                    if model_cfg.enabled:
                        provider_model_name = f"{provider_name}: {model_name}"
                        provider_model_list.append(
                            (provider_model_name, model_cfg.weight)
                        )
    return provider_model_list


def get_provider_model_list():
    provider_model_list = []
    if config.type == "default":
        provider_model_name = f"{get_default_provider()}: {get_default_model()}"
        provider_model_list.append(provider_model_name)
    else:
        providers = config.llm_providers.providers
        for provider_name in providers:
            provider_cfg = llm_config.providers.get(provider_name)
            if provider_cfg and provider_cfg.enabled:
                for model_name in provider_cfg.models:
                    model_cfg = provider_cfg.models[model_name]
                    if model_cfg.enabled:
                        provider_model_name = f"{provider_name}: {model_name}"
                        provider_model_list.append(provider_model_name)
    return provider_model_list


def get_provider_display_list():
    provider_display_list = []
    providers = config.llm_providers.providers
    for provider_name in providers:
        provider_cfg = llm_config.providers.get(provider_name)
        for model_name in provider_cfg.models:
            provider = {
                "provider_name": provider_name,
            }
            model_cfg = provider_cfg.models[model_name]
            enabled = (
                False
                if provider_cfg.enabled == False
                else provider_cfg.enabled and model_cfg.enabled
            )
            url = provider_cfg.url if model_cfg.url in (None, "") else model_cfg.url
            provider["model_name"] = model_name
            provider["url"] = url
            provider["enabled"] = enabled
            provider_display_list.append(provider)
    return provider_display_list


def get_default_provider_model():
    return f"{get_default_provider()}: {get_default_model()}"


def get_provider_model(provider_name, model_name):
    if provider_name is None or model_name is None:
        return None, None
    providers = config.llm_providers.providers
    if provider_name in config.llm_providers.providers:
        provider_cfg = providers.get(provider_name)
        if model_name in provider_cfg.models:
            model_cfg = provider_cfg.models[model_name]
    return provider_cfg, model_cfg


def update_provider(provider, model, enabled):
    provider_cfg, model_cfg = get_provider_model(provider, model)
    if provider_cfg is None or model_cfg is None:
        return
    if provider_cfg.enabled == False and enabled:
        # loop through all the models and set it to False
        for model_name in provider_cfg.models:
            m_cfg = provider_cfg.models[model_name]
            m_cfg.enabled = False

        model_cfg.enabled = enabled
    else:
        model_cfg.enabled = enabled


def delete_provider(provider, model):
    if provider is None or model is None:
        return
    providers = config.llm_providers.providers
    provider_cfg = providers.get(provider)
    if model in provider_cfg.models:
        del provider_cfg.models[model]


def add_provider_and_model(
    provider_name: str,
    model_name: str,
    url: str,
    api_key: str,
    enabled: bool,
    params: dict,
    weight: int,
):

    if provider_name in config.llm_providers.providers:
        provider_cfg = config.llm_providers.providers.get(provider_name)
    else:
        # create a dictionary for provider config
        provider_dict = dict({"name": provider_name, "enabled": True})
        provider_cfg = ProviderConfig(provider_dict)
        config.llm_providers.providers[provider_name] = provider_cfg

    provider_cfg.weight = weight
    if provider_cfg.enabled == False and enabled:
        # loop through all the models and set it to False
        for model_name in provider_cfg.models:
            m_cfg = provider_cfg.models[model_name]
            m_cfg.enabled = False

    if model_name in provider_cfg.models:
        model_cfg = provider_cfg.models.get(model_name)
        model_cfg.url = url
        model_cfg.enabled = enabled
        model_cfg.credentials = api_key if api_key.strip() else model_cfg.credentials
        param_dict = {}
        if params:
            for param in params:
                param_dict[param["name"]] = param["value"]

        model_cfg.params.update(param_dict)
    else:
        model_dict = dict(
            {
                "name": model_name,
                "url": url,
                "weight": weight,
                "credentials": api_key,
                "enabled": True,
                "params": params,
            }
        )
        model_cfg = ModelConfig(model_dict)
        provider_cfg.models[model_name] = model_cfg
