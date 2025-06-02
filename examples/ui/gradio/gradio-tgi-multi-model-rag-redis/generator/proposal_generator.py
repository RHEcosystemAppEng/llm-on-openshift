import os
import threading
import time
from collections.abc import Generator
from queue import Empty, Queue
from threading import Thread

import pdfkit
from langchain.chains import LLMChain
from langchain.llms.base import LLM
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from markdown import markdown
from prometheus_client import Counter, Gauge

from generator.query_helper import QueryHelper
from generator.template import VECTOR_DB_QUERY_TEMPLATE

PDF_FILE_DIR = "proposal-docs"

store = {}

FEEDBACK_COUNTER = Counter(
    "feedback_stars", "Number of feedbacks by stars", ["stars", "model_id"]
)
MODEL_USAGE_COUNTER = Counter(
    "model_usage", "Number of times a model was used", ["model_id"]
)
REQUEST_TIME = Gauge(
    "request_duration_seconds", "Time spent processing a request", ["model_id"]
)


class ProposalGenerator:
    """Generator project proposal for a given product addressed to a company"""

    def __init__(self, session_id: str):
        self.lock = threading.Lock()
        self.session_id = session_id

    def generate_proposal(
        self, llm: LLM, model_id: str, que: Queue, product: str, company: str
    ) -> Generator:
        query = VECTOR_DB_QUERY_TEMPLATE.format(product=product, company=company)
        proposal_chain = QueryHelper().get_proposal_template_chain(llm)
        return self.stream(proposal_chain, que, {"query": query}, model_id)

    def update_proposal(
        self, llm: LLM, model_id: str, que: Queue, old_proposal: str, user_query: str
    ) -> Generator:
        proposal_chain = QueryHelper().get_update_proposal_chain(llm)
        model_input = {"old_proposal": old_proposal, "user_query": user_query}
        return self.stream(proposal_chain, que, model_input, model_id)

    def get_pdf_file(self) -> str:
        return os.path.join("./assets", PDF_FILE_DIR, f"proposal-{self.session_id}.pdf")

    def create_pdf(self, text: str):
        try:
            output_filename = self.get_pdf_file()
            html_text = markdown(text, output_format="html4")
            pdfkit.from_string(html_text, output_filename)
        except Exception as e:
            print(f"[PDF ERROR] {e}")

    # Function to initialize all star ratings to 0
    def initialize_feedback_counters(self, model_id: str):
        for star in range(1, 6):
            FEEDBACK_COUNTER.labels(stars=str(star), model_id=model_id).inc(0)

    def remove_source_duplicates(self, input_list):
        return list(
            dict.fromkeys(
                item.metadata.get("source")
                for item in input_list
                if item.metadata.get("source")
            )
        )

    def get_session_history(self) -> BaseChatMessageHistory:
        if self.session_id not in store:
            store[self.session_id] = ChatMessageHistory()
        return store[self.session_id]

    def remove_role(self, message: str) -> str:
        if message:
            for role in ["Assistant:", "AI:", "Human:"]:
                message = message.replace(role, "", 1)
            if "</think>" in message:
                message = message.split("</think>")[-1]
        return message

    def format_chat_history(self, chat_history: list):
        formatted = []
        for msg in chat_history:
            role, content = msg["role"], self.remove_role(msg["content"])
            if role == "user":
                formatted.append(HumanMessage(content=content))
            elif role == "assistant":
                formatted.append(AIMessage(content=content))
        return formatted

    def post_process(self, answer: str) -> str:
        answer = answer.split("Human:")[0]
        return answer.replace("AI:", "").replace("Assistant:", "")

    def get_answer(
        self, llm: LLM, model_id: str, que: Queue, query: str, chat_history: list
    ):
        MODEL_USAGE_COUNTER.labels(model_id=model_id).inc()
        self.initialize_feedback_counters(model_id)
        chain = QueryHelper().get_qa_chain(llm)
        model_input = {
            "input": query,
            "chat_history": self.format_chat_history(chat_history),
        }
        start_time = time.perf_counter()
        resp = chain.invoke(input=model_input)
        REQUEST_TIME.labels(model_id=model_id).set(time.perf_counter() - start_time)
        return self.post_process(resp["answer"])

    def _get_existing_sources(self, text: str) -> list:
        """
        Parse `text` and return any sources already listed at the end.
        A source line is a bullet (`*` or `-`) immediately following a
        header like `# Sources` or `*Sources:*`.
        """
        lines = text.strip().splitlines()
        existing = []

        # Walk from bottom-up until we hit a 'Sources' header
        for i in range(len(lines) - 1, -1, -1):
            stripped = lines[i].strip().lower()
            if stripped.startswith("# sources") or stripped.startswith("*sources:"):
                # Everything below this header that's a bullet is a source
                for src_line in lines[i + 1 :]:
                    src_line_stripped = src_line.strip()
                    if src_line_stripped.startswith(
                        "*"
                    ) or src_line_stripped.startswith("-"):
                        existing.append(src_line_stripped[1:].lstrip())
                break

        return existing

    def append_sources_to_queue(self, resp: dict, que: Queue):
        """
        Push sources to the queue.

        The first time a non-empty source list appears, the function outputs
        the header “\n*Sources:*\n” followed by each source as a bullet.

        On subsequent calls, it appends *only* sources that have not
        appeared before—no duplicate bullets, no extra headers.
        """
        if "source_documents" in resp and resp["source_documents"] is not None:
            sources = self.remove_source_duplicates(resp["source_documents"])
        else:
            context = resp.get("context", [])
            sources = list(dict.fromkeys([doc.metadata["source"] for doc in context]))

        if not sources:
            return

        existing_sources = self._get_existing_sources(resp.get("result", ""))
        if existing_sources:
            for source in sources:
                if source not in existing_sources:
                    que.put(f"* {source}\n")
        else:
            que.put("\n\n*Sources:* \n")
            for source in sources:
                que.put(f"* {source}\n")

    def stream(
        self, chain: LLMChain, que: Queue, model_input: dict, model_id: str
    ) -> Generator:
        # Create a Queue
        job_done = object()

        # Create a function to call - this will run in a thread
        def task():
            MODEL_USAGE_COUNTER.labels(model_id=model_id).inc()
            # Call this function at the start of your application
            self.initialize_feedback_counters(model_id)
            with self.lock:
                start_time = (
                    time.perf_counter()
                )  # start and end time to get the precise timing of the request
                try:
                    resp = chain.invoke(input=model_input)

                    end_time = time.perf_counter()
                    REQUEST_TIME.labels(model_id=model_id).set(end_time - start_time)
                    self.create_pdf(resp["result"])
                    self.append_sources_to_queue(resp, que)

                except Exception as e:
                    print(e)
                    que.put("Error executing request. Contact the administrator.")

                que.put(job_done)

        # Create a thread and start the function
        t = Thread(target=task)
        t.start()

        content = ""

        # Get each new token from the queue and yield for our generator
        while True:
            try:
                next_token = que.get(True, timeout=100)
                if next_token is job_done:
                    break
                if isinstance(next_token, str):
                    content += next_token
                    yield next_token, content
            except Empty:
                continue
