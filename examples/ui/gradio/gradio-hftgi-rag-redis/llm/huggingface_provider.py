from typing import Optional
import inspect
import os
from queue import Queue
from langchain.llms.base import LLM
from llm.llm_provider import LLMProvider, QueueCallback
#from langchain_core.callbacks.manager import CallbackManager
# Streaming implementation

class HuggingFaceProvider(LLMProvider):
  
  def __init__(self, provider, model, params):
    super().__init__(provider, model, params)
    pass

  def _tgi_llm_instance(self) -> LLM:
        """Note: TGI does not support specifying the model, it is an instance per model."""
        print(
            f"[{inspect.stack()[0][3]}] Creating Hugging Face TGI LLM instance"
        )
        try:
            from langchain.llms import HuggingFaceTextGenInference
        except Exception as e:
            print(
                "Missing HuggingFaceTextGenInference libraries. HuggingFaceTextGenInference "
                "provider will be unavailable."
            )
            raise e
        if self._llm_instance is None:
          params: dict = {
              "inference_server_url": self._get_llm_url(""),
              "model_kwargs": {},  # TODO: add model args
              "max_new_tokens": 512,
              "cache": None,
              "temperature": 0.01,
              "top_k": 10,
              "top_p": 0.95,
              "repetition_penalty": 1.03,
              "streaming": True,
              "verbose": False,
              "callbacks": [QueueCallback(self._queue)]
          }
          if self.model_config.params:
            params.update(self.model_config.params)  # override parameters
          self._llm_instance = HuggingFaceTextGenInference(**params)

        print(
            f"[{inspect.stack()[0][3]}] Hugging Face TGI LLM instance {self._llm_instance}"
        )

        return self._llm_instance

  def get_llm(self) -> (LLM, Queue):
      return self._tgi_llm_instance(), self._queue
