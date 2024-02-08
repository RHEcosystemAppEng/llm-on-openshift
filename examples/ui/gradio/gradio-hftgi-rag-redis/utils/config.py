"""Config classes for the configuration structure."""

from typing import Optional

def _get_attribute_from_file(data: dict, file_name_key: str) -> Optional[str]:
    """Retrieve value of an attribute from a file."""
    file_path = data.get(file_name_key)
    if file_path is not None:
        try:
            with open(file_path, mode="r") as f:
                return f.read().rstrip()
        except Exception as err:
            print (err)
    return None


class ModelConfig():
    """Model configuration."""

    name: Optional[str] = None
    url: Optional[str] = None
    credentials: Optional[str] = None
    enabled: Optional[bool] = None
    weight: Optional[int] = None

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        if data is None:
            return
        self.name = data.get("name", None)
        self.url = data.get("url", None)
        self.credentials = _get_attribute_from_file(data, "credentials_path")
        self.enabled = data.get("enabled", True)
        self.weight = data.get("weight", 1)
        self.params = {}
        param_data = data.get("params", None)
        if param_data:
            for param in param_data:
                self.params[param["name"]] = param["value"]


class ProviderConfig():
    """LLM provider configuration."""

    name: Optional[str] = None
    url: Optional[str] = None
    credentials: Optional[str] = None
    enabled: Optional[bool] = None
    
    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        self.models = {}
        if data is None:
            return
        self.name = data.get("name", None)
        self.url = data.get("url", None)
        self.credentials = _get_attribute_from_file(data, "credentials_path")
        self.enabled = data.get("enabled", True)

        for m in data["models"]:
            model = ModelConfig(m)
            self.models[m["name"]] = model

class LLMProviders():
    """LLM providers configuration."""

    providers: dict[str, ProviderConfig] = {}

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        if data is None:
            return
        for p in data:
            provider = ProviderConfig(p)
            self.providers[p["name"]] = provider
 
class Config():
    """Global service configuration."""

    llm_providers: Optional[LLMProviders] = None
    default_provider: Optional[str] = None
    default_model: Optional[str] = None
    type: Optional[str] = None

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        if data is None:
            return
        v = data.get("llm_providers")
        if v is not None:
            self.llm_providers = LLMProviders(v)
        self.default_provider = data.get("default_provider", None)
        self.default_model = data.get("default_model", None)
        self.type = data.get("type", "default")


