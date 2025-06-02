import inspect
import os
from typing import AsyncIterator, Iterator, Optional

from langchain.llms.base import LLM
from langchain_core.outputs import GenerationChunk
from langchain_huggingface import HuggingFaceEndpoint

from llm.llm_provider import LLMProvider


class PatchedHuggingFaceEndpoint(HuggingFaceEndpoint):
    """
    * Removes duplicate **stop** kwarg that LangChain adds when `stream=True`.
    * Forwards every streamed token to the standard callback pipeline, so
      the existing `QueueCallback` keeps working and the Gradio UI updates
      live.
    """

    # forward token to callbacks when no run-manager was created
    def _fire_callbacks(self, text: str) -> None:
        for cb in self.callbacks:
            try:
                cb.on_llm_new_token(text)
            except Exception:
                pass

    def _stream(self, *args, **kwargs) -> Iterator[GenerationChunk]:
        kwargs.pop("stop", None)  # <- deduplicate
        for chunk in super()._stream(*args, **kwargs):
            # if the caller didn't pass its own run_manager, make sure the
            # QueueCallback (attached via llm.callbacks) still receives tokens
            if kwargs.get("run_manager", None) is None:
                self._fire_callbacks(chunk.text)
            yield chunk

    async def _astream(self, *args, **kwargs) -> AsyncIterator[GenerationChunk]:
        kwargs.pop("stop", None)
        async for chunk in super()._astream(*args, **kwargs):
            if kwargs.get("run_manager", None) is None:
                self._fire_callbacks(chunk.text)
            yield chunk

    def _call(self, prompt: str, *args, **kwargs):
        kwargs.pop("stop", None)
        return super()._call(prompt, *args, **kwargs)


class HuggingFaceProvider(LLMProvider):
    def __init__(self, provider, model, params):
        super().__init__(provider, model, params)

    def _tgi_llm_instance(self, callback) -> LLM:
        print(f"[{inspect.stack()[0][3]}] Creating Hugging Face TGI LLM instance")

        api_token = str(self._get_llm_credentials())

        streaming = bool(callback)
        params: dict = {
            "endpoint_url": self._get_llm_url(""),
            # "model_kwargs": {},  TODO: add model args
            "huggingfacehub_api_token": api_token,
            "cache": None,
            "temperature": 0.01,
            "top_k": 10,
            "top_p": 0.95,
            "repetition_penalty": 1.03,
            "streaming": streaming,
            "verbose": True,
            "max_new_tokens": 1024,
        }

        if streaming:
            params["callbacks"] = [callback]

        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token

        self._llm_instance = PatchedHuggingFaceEndpoint(**params)

        print(
            f"[{inspect.stack()[0][3]}] Hugging Face TGI LLM instance initialized: {self._llm_instance}"
        )

        return self._llm_instance

    def get_llm(self, callback) -> LLM:
        return self._tgi_llm_instance(callback)
