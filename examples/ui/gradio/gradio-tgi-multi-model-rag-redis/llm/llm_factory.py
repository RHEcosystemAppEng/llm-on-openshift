from typing import Dict
from llm.huggingface_provider import HuggingFaceProvider
from llm.llm_provider import LLMProvider
from llm.nemo_provider import NeMoProvider
from llm.openai_provider import OpenAIProvider
from langchain.llms.base import LLM

from llm.openshift_ai_vllm import OpenShiftAIvLLM

HUGGING_FACE = "Hugging Face"
NVIDIA = "NVIDIA"
OPENAI = "OpenAI"
OPENSHIFT_AI_VLLM="OpenShift AI (vLLM)"

class LLMFactory:
    _providers: Dict[str, LLMProvider] = {}
    def __init__(self):
        pass

    def _create_key(self, provider, model):
        return f"{provider}:{model}"
    
    def init_providers(self, config):
        self._providers = {}
        providers = config.llm_providers.providers
        for provider_name in providers:
            provider_cfg = providers[provider_name]
            for model_name in provider_cfg.models:
                model_cfg = provider_cfg.models[model_name]
                self._register_llm_provider(config, provider_cfg.name, model_cfg.name)


    def _register_llm_provider(self, config, provider, model):
        key = self._create_key(provider, model)
        self._providers[key] = self._create_llm_provider(config, provider, model)

    def _get_NeMo_Provider(self, config, provider, model):
        if model.startswith("Remote-"):
            local_model = model.replace("Remote-", "")
            provider_cfg = config.llm_providers.providers[provider]
            for model_name in provider_cfg.models:
                if model_name.lower() == local_model.lower():
                    return NeMoProvider(provider, model_name, None)
        return NeMoProvider(provider, model, None)
    
    def _create_llm_provider(self, config, provider, model) -> LLMProvider:
        if provider == OPENAI:
            return OpenAIProvider(provider, model, None)
        elif provider == NVIDIA:
            return self._get_NeMo_Provider(config, provider, model)
        elif provider == HUGGING_FACE:
            return HuggingFaceProvider(provider, model, None)
        elif provider == OPENSHIFT_AI_VLLM:
            return OpenShiftAIvLLM(provider, model, None)
        else:
            raise ValueError(provider, model)
        
    def get_llm(self, provider, model, callback) -> LLM:
        key = self._create_key(provider, model)
        provider = self._providers[key]
        if provider is not None:
            return provider.get_llm(callback)

    @classmethod 
    def get_providers(cls) -> list:
        return [HUGGING_FACE, NVIDIA, OPENAI, OPENSHIFT_AI_VLLM]
