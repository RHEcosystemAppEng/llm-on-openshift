from typing import Optional
import inspect
from langchain.llms.base import LLM
from llm.llm_provider import LLMProvider, QueueCallback
from queue import Queue
import os


class OpenAIProvider(LLMProvider):
  def __init__(self, provider, model, params):
    super().__init__(provider, model, params)
    pass

  def _openai_llm_instance(self) -> LLM:
        print(f"[{inspect.stack()[0][3]}] Creating OpenAI LLM instance")
        try:
            #from langchain.llms import OpenAI
           from langchain.chat_models import ChatOpenAI
        except Exception as e:
            print(
                "Missing openai libraries. Openai provider will be unavailable."
            )
            raise e
        if self._llm_instance is None:
          params: dict = {
              "base_url": self._get_llm_url("https://api.openai.com/v1"),
              "openai_api_key": self._get_llm_credentials(),
              "model": self.model,
              "model_kwargs": {},  # TODO: add model args
              "organization": None,
              "timeout": None,
              "cache": None,
              "streaming": True,
              "temperature": 0.01,
              "max_tokens": 512,
              "top_p": 0.95,
              "verbose": False,
              "callbacks": [QueueCallback(self._queue)]
          }
          os.environ["OPENAI_API_KEY"] =  self._get_llm_credentials()
          if self.model_config.params:
            params.update(self.model_config.params)  # override parameters

          self._llm_instance = ChatOpenAI(**params)
          print(f"[{inspect.stack()[0][3]}] OpenAI LLM instance {self._llm_instance}")
        return self._llm_instance

  def get_llm(self) -> (LLM, Queue):
      return self._openai_llm_instance(), self._queue