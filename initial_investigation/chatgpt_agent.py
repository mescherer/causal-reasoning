"""Minimal interface for probing an OpenAI model.

Setup
-----
1. Put your API key in ``.env``::

       OPENAI_API_KEY=your_api_key_here

2. Create an agent and send it a prompt::

       from chatgpt_agent import ChatGPTAgent

       agent = ChatGPTAgent(
           instructions="Answer causal-reasoning questions concisely."
       )
       answer = agent.ask("If X causes Y, what happens when X is removed?")
       print(answer)

Running ``python chatgpt_agent.py`` also starts a small interactive prompt.
Each call to :meth:`ChatGPTAgent.ask` is independent; the class does not retain
conversation history.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


DEFAULT_INSTRUCTIONS = (
    "You are a careful research assistant. Answer the user's question directly, "
    "state important assumptions, and do not invent facts."
)


class ChatGPTAgent:
    """A small, stateless wrapper around the OpenAI Responses API.

    Parameters
    ----------
    instructions:
        Behavior applied to every request. Put the agent's role, constraints,
        and desired response style here.
    model:
        OpenAI model name. Defaults to ``OPENAI_MODEL`` from ``.env``, or
        ``gpt-5-mini`` when that setting is absent.
    api_key:
        Optional explicit key. Normally, leave this unset and store the key in
        the ignored ``.env`` file as ``OPENAI_API_KEY``.
    client:
        Optional compatible client, primarily useful for tests.
    """

    def __init__(
        self,
        instructions: str = DEFAULT_INSTRUCTIONS,
        model: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError("instructions must be a non-empty string")

        load_dotenv(Path(__file__).with_name(".env"))

        self.instructions = instructions.strip()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")

        if client is not None:
            self._client = client
            return

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "OPENAI_API_KEY is missing. Add it to the project's .env file."
            )

        # Import only when a real client is requested, which keeps unit tests
        # fast and prevents them from making network calls.
        from openai import OpenAI

        self._client = OpenAI(api_key=resolved_key)

    def ask(self, prompt: str) -> str:
        """Send one prompt to the configured model and return its text answer."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        response = self._client.responses.create(
            model=self.model,
            instructions=self.instructions,
            input=prompt.strip(),
        )
        return response.output_text


def main() -> None:
    """Run a minimal terminal interface for manual probing."""
    agent = ChatGPTAgent()
    print("Enter a prompt, or type 'quit' to stop.")

    while True:
        prompt = input("> ").strip()
        if prompt.lower() in {"quit", "exit"}:
            break
        if prompt:
            print(agent.ask(prompt))


if __name__ == "__main__":
    main()
