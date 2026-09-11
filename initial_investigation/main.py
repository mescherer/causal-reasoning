import json
import os

from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    function_tool,
    set_tracing_disabled,
)
from openai import AsyncOpenAI

from scientist import Scientist
from world import World


DEFAULT_QUESTION = "Determine the causal relationships among A, B, and C."
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

SCIENTIST_INSTRUCTIONS = """
You are a scientist investigating an unknown causal system containing variables
A, B, and C. Your goal is to infer the directed causal relationships among the
variables under a finite experimental budget.

Use the budget tool to check the remaining budget. Use the experiment tool to
request observational or interventional data. An omitted variable is
uncontrolled; a supplied value intervenes on that variable. Never request more
experiments than the remaining budget.

When you have enough evidence or cannot run another useful experiment, state
"Investigation concluded." Then provide a concise textual causal map, listing
the proposed directed edges, their signs or strengths when supported, and any
remaining uncertainty.
""".strip()


def create_gemini_model(model_name=None):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Export it in the current terminal."
        )

    set_tracing_disabled(True)
    client = AsyncOpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)
    return OpenAIChatCompletionsModel(
        model=model_name or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        openai_client=client,
    )


def build_scientist_agent(world, scientist, model):
    @function_tool
    def experiment(
        n_experiments: int = 1,
        A: float | None = None,
        B: float | None = None,
        C: float | None = None,
    ) -> str:
        """Run experiments on the causal world.

        Args:
            n_experiments: Number of experiments to run.
            A: Optional intervention value for A.
            B: Optional intervention value for B.
            C: Optional intervention value for C.
        """
        interventions = ", ".join(
            f"{name}={value}"
            for name, value in (("A", A), ("B", B), ("C", C))
            if value is not None
        ) or "none"
        print(
            f"[agent] Experiment request: count={n_experiments}, "
            f"interventions={interventions}, budget={scientist.budget}",
            flush=True,
        )
        result = scientist.experiment(world, n_experiments, A, B, C)
        if isinstance(result, str):
            print(f"[agent] Experiment rejected: {result}", flush=True)
        else:
            print(
                f"[agent] Experiment complete: "
                f"budget remaining={scientist.budget}",
                flush=True,
            )
        return result if isinstance(result, str) else json.dumps(result)

    @function_tool
    def budget() -> int:
        """Return the number of experiments still available."""
        remaining = scientist.budget
        print(f"[agent] Budget check: {remaining} remaining", flush=True)
        return remaining

    return Agent(
        name="Causal Scientist",
        instructions=SCIENTIST_INSTRUCTIONS,
        model=model,
        tools=[experiment, budget],
    )


def main(
    question=DEFAULT_QUESTION,
    experimental_budget=10,
    model_name=None,
    runner=Runner,
    model_factory=create_gemini_model,
):
    world = World()
    scientist = Scientist(experimental_budget)
    model = model_factory(model_name)
    agent = build_scientist_agent(world, scientist, model)
    result = runner.run_sync(
        agent,
        question,
        max_turns=max(5, experimental_budget * 2 + 5),
    )
    print(result.final_output)
    print(f"Remaining budget: {scientist.budget}")
    return result.final_output


if __name__ == "__main__":
    main()
