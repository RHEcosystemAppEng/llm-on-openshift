import pandas as pd

from generator.callback import QueueCallback
from llm.llm_factory import LLMFactory
from scheduler.round_robin import RoundRobinScheduler
from utils import config_loader

config_loader.init_config()
llm_factory = LLMFactory()
llm_factory.init_providers(config_loader.config)

global sched


def is_provider_visible():
    return config_loader.config.type == "all"


def get_llm_factory():
    return llm_factory


def remove_source_duplicates(input_list):
    unique_list = []
    for item in input_list:
        if item.metadata["source"] not in unique_list:
            unique_list.append(item.metadata["source"])
    return unique_list


def create_scheduler():
    global sched
    provider_model_weight_list = config_loader.get_provider_model_weight_list()
    # initialize scheduler
    sched = RoundRobinScheduler(provider_model_weight_list)


def get_provider_model(provider_model):
    if provider_model is None:
        return "", ""
    s = provider_model.split(": ")
    return s[0], s[1]


def get_llm(provider_model, streaming, que):
    callback = None
    if streaming:
        callback = QueueCallback(que)
    provider_id, model_id = get_provider_model(provider_model)
    return llm_factory.get_llm(provider_id, model_id, callback)


def get_selected_provider():
    if config_loader.config.type == "round_robin":
        return sched.get_next()

    provider_list = config_loader.get_provider_model_weight_list()
    if len(provider_list) > 0:
        return provider_list[0]

    return None


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
