from typing import Optional, Tuple
import inspect
import os
from queue import Queue
from langchain.llms.base import LLM
from llm.huggingface_text_gen_inference import HuggingFaceTextGenInference
from llm.llm_provider import LLMProvider
#from langchain_core.callbacks.manager import CallbackManager
# Streaming implementation

class HuggingFaceProvider(LLMProvider):
  
  def __init__(self, provider, model, params):
    super().__init__(provider, model, params)
    pass

  def _tgi_llm_instance(self, callback) -> LLM:
    """Note: TGI does not support specifying the model, it is an instance per model."""
    print(
        f"[{inspect.stack()[0][3]}] Creating Hugging Face TGI LLM instance"
    )
    # try:
    #     from langchain.llms import HuggingFaceTextGenInference
    # except Exception as e:
    #     print(
    #         "Missing HuggingFaceTextGenInference libraries. HuggingFaceTextGenInference "
    #         "provider will be unavailable."
    #     )
    #     raise e
    streaming = True if callback else False
    params: dict = {
        "inference_server_url": self._get_llm_url(""),
#         "model_kwargs": {},  # TODO: add model args
#          "max_new_tokens": 512,
        "cache": None,
        "temperature": 0.01,
        "top_k": 10,
        "top_p": 0.95,
        "repetition_penalty": 1.03,
        "streaming": streaming,
        "verbose": False
    }
    if streaming == True: 
      params["callbacks"] = [callback] 
      # if self.model_config.params:
      #   params.update(self.model_config.params)  # override parameters
    self._llm_instance = HuggingFaceTextGenInference(**params)

    print(
        f"[{inspect.stack()[0][3]}] Hugging Face TGI LLM instance {self._llm_instance}"
    )

    return self._llm_instance

  def get_llm(self, callback) -> LLM:
    return self._tgi_llm_instance(callback)
