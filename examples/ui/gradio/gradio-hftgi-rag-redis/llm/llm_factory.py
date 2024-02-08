from llm.huggingface_provider import HuggingFaceProvider
from llm.llm_provider import LLMProvider
from llm.openai_provider import OpenAIProvider
from langchain.llms.base import LLM
from queue import Queue

class LLMFactory:
    _providers: dict[str, LLMProvider] = {}
    def __init__(self):
        pass
        #self._init_providers()

    def _create_key(self, provider, model):
        return f"{provider}:{model}"
    
    def init_providers(self, config):
        providers = config.llm_providers.providers
        for provider_name in providers:
            provider_cfg = providers[provider_name]
            for model_name in provider_cfg.models:
                model_cfg = provider_cfg.models[model_name]
                self._register_llm_provider(provider_cfg.name, model_cfg.name)


    def _register_llm_provider(self, provider, model):
        key = self._create_key(provider, model)
        self._providers[key] = self._create_llm_provider(provider, model)

    
    def _create_llm_provider(self, provider, model) -> LLMProvider:
        if provider == 'OpenAI':
            return OpenAIProvider(provider, model, None)
        elif provider == 'Hugging Face':
            return HuggingFaceProvider(provider, model, None)
        else:
            raise ValueError(provider, model)
        
    def get_llm(self, provider, model) -> (LLM, Queue):
        key = self._create_key(provider, model)
        provider = self._providers[key]
        if provider is not None:
            return provider.get_llm()

