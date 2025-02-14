"""LLM backend libraries loader."""

from typing import Optional, Tuple
from utils import config_loader
from langchain.llms.base import LLM

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

    def get_model(self) -> str:
        return self.model
    
    def get_provider(self) -> str:
        return self.provider
    
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

    def get_llm(self, callback) -> LLM:
      return None
    
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
