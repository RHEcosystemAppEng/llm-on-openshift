from typing import Optional, Tuple
import inspect
from langchain.llms.base import LLM
from openai import AsyncOpenAI
from llm.llm_provider import LLMProvider
from queue import Queue
import os
import httpx

def update_base_url(request: httpx.Request) -> None:
  # rewrite the path segment to what the proxy expects
  print(request.url.path)
  print(request.content)

  if request.url.path.endswith("/chat/completions"):
      print(request.url.path)
      path = request.url.path.replace("/chat/completions","")
      print(path)
      request.url = request.url.copy_with(path=path)

def log_request(request):
    print(f"Request event hook: {request.method} {request.url} - Waiting for response")

def log_response(response):
    request = response.request
    print(f"Response event hook: {request.method} {request.url} - Status {response.status_code}")

class NeMoProvider(LLMProvider):
  def __init__(self, provider, model, params):
    super().__init__(provider, model, params)
    pass

  def _nemo_llm_instance(self, callback) -> LLM:
    print(f"[{inspect.stack()[0][3]}] Creating OpenAI LLM instance")
    try:
      from langchain.chat_models import ChatOpenAI
      from openai import OpenAI
      from llm.localai import ChatOpenAILocal
    except Exception as e:
      print(
          "Missing openai libraries. Openai provider will be unavailable."
      )
      raise e
    creds = self._get_llm_credentials()
    if creds in (None, ''):
        creds = "dummy-api-key" # ChatOpenAI requires creds to be not none

      # Creating an object of custom handler
    streaming = True if callback else False
    params: dict = {
        "base_url": self._get_llm_url(""),
        "model": self.model,
#        "model_kwargs": {},  # TODO: add model args
        "organization": None,
        "timeout": None,
        "cache": None,
        "streaming": streaming,
        "temperature": 0.01,
        "max_tokens": 512,
        #"top_p": 0.95,
        "verbose": True,
        "callbacks": [callback]
    }
    if streaming == True: 
      params["callbacks"] = [callback] 
    # if self.model_config.params:
    #   params.update(self.model_config.params)
    os.environ["OPENAI_API_KEY"] =  creds

    if self.model.startswith("Remote-"):
      httpx_client = httpx.Client(event_hooks={'request': [log_request, update_base_url], 'response': [log_response]})
      client = OpenAI(
          base_url=self._get_llm_url(""),
          http_client=httpx_client,
      )

      async_client = AsyncOpenAI(
          base_url=self._get_llm_url(""),
          http_client=httpx_client,
      )
      params["client"] = client.chat.completions
      params["async_client"] = async_client.chat.completions
      self._llm_instance = ChatOpenAILocal(**params)
    else:
      async_client=httpx.AsyncClient(verify=False)
      http_client=httpx.Client(verify=False)
      self._llm_instance = ChatOpenAI(**params, async_client=async_client, http_client=http_client)

    print(f"[{inspect.stack()[0][3]}] OpenAI LLM instance {self._llm_instance}")
    return self._llm_instance

  def get_llm(self, callback) -> LLM:
    return self._nemo_llm_instance(callback)