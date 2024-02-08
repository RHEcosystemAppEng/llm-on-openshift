"""Configuration loader."""

import traceback
from io import TextIOBase
import os

import yaml
from utils.config import Config

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

def get_provider_model_weightage_list():
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
                    model_cfg =  provider_cfg.models[model_name]
                    if model_cfg.enabled:
                        provider_model_name = f"{provider_name}: {model_name}"
                        provider_model_list.append((provider_model_name, model_cfg.weight))
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
                    model_cfg =  provider_cfg.models[model_name]
                    if model_cfg.enabled:
                        provider_model_name = f"{provider_name}: {model_name}"
                        provider_model_list.append(provider_model_name)
    return provider_model_list

def get_default_provider_model():
    return f"{get_default_provider()}: {get_default_model()}"
