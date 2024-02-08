"""LLM backend libraries loader."""

from typing import Optional
from utils import config_loader
from langchain.llms.base import LLM
from queue import Queue
from langchain.callbacks.base import BaseCallbackHandler

from utils.config import ProviderConfig

class LLMConfigurationError(Exception):
    """LLM configuration is wrong."""


class MissingProviderError(LLMConfigurationError):
    """Provider is not specified."""


class MissingModelError(LLMConfigurationError):
    """Model is not specified."""


class UnsupportedProviderError(LLMConfigurationError):
    """Provider is not supported or is unknown."""


class ModelConfigMissingError(LLMConfigurationError):
    """No configuration exists for the requested model name."""


class ModelConfigInvalidError(LLMConfigurationError):
    """Model configuration is not valid."""

class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: any) -> None:
        return self.q.empty()


class LLMProvider:
    """Load LLM backend.
    """
    _llm_instance: Optional [LLM] = None
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        params: Optional[dict] = None,
    ) -> None:
        self._queue = Queue()
        if provider is None:
            msg = "Missing provider"
            print(msg)
            raise MissingProviderError(msg)
        self.provider = provider
        if model is None:
            msg = "Missing model"
            print(msg)
            raise MissingModelError(msg)
        self.model = model
        self.provider_config = self._get_provider_config()
        self.model_config = self.provider_config.models.get(self.model)

    
    # TODO: refactor after config implementation OLS-89
    def _get_provider_config(self) -> ProviderConfig:
        cfg = config_loader.llm_config.providers.get(self.provider)
        if not cfg:
            msg = f"Unsupported LLM provider {self.provider}"
            print(msg)
            raise UnsupportedProviderError(msg)

        model = cfg.models.get(self.model)
        if not model:
            msg = (
                f"No configuration provided for model {self.model} under "
                f"LLM provider {self.provider}"
            )
            print(msg)
            raise ModelConfigMissingError(msg)
        return cfg

    def get_llm(self) -> (LLM, Queue):
      pass
    
    def _get_llm_url(self, default: str) -> str:
        return (
            self.provider_config.models[self.model].url
            if self.provider_config.models[self.model].url is not None
            else (
                self.provider_config.url
                if self.provider_config.url is not None
                else default
            )
        )

    def _get_llm_credentials(self) -> str:
        return (
            self.provider_config.models[self.model].credentials
            if self.provider_config.models[self.model].credentials is not None
            else self.provider_config.credentials
        )
    
    def status(self):
        """Provide LLM schema as a string containing formatted and indented JSON."""
        import json

        return json.dumps(self.llm.schema_json, indent=4)
