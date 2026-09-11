import asyncio
import json
from types import SimpleNamespace

import main as main_module
import pytest
from agents.tool_context import ToolContext
from scientist import Scientist


class StubWorld:
    def __init__(self):
        self.calls = []

    def observe(self, n_experiments, A, B, C):
        self.calls.append((n_experiments, A, B, C))
        return {"experiment: 0": {"A": 1.0, "B": 2.0, "C": 3.0}}


def get_tool(agent, name):
    return next(tool for tool in agent.tools if tool.name == name)


def invoke(tool, arguments=None):
    payload = json.dumps(arguments or {})
    context = ToolContext(
        context=None,
        tool_name=tool.name,
        tool_call_id="test-call",
        tool_arguments=payload,
    )
    return asyncio.run(tool.on_invoke_tool(context, payload))


def test_agent_exposes_experiment_and_budget_tools():
    agent = main_module.build_scientist_agent(
        StubWorld(), Scientist(3), model="test-model"
    )

    assert {tool.name for tool in agent.tools} == {"experiment", "budget"}


def test_experiment_tool_observes_world_and_consumes_budget(capsys):
    world = StubWorld()
    scientist = Scientist(3)
    agent = main_module.build_scientist_agent(world, scientist, model="test-model")

    output = invoke(
        get_tool(agent, "experiment"),
        {"n_experiments": 2, "A": 4.0, "B": None, "C": 1.0},
    )

    assert json.loads(output) == {
        "experiment: 0": {"A": 1.0, "B": 2.0, "C": 3.0}
    }
    assert world.calls == [(2, 4.0, None, 1.0)]
    assert scientist.budget == 1
    assert capsys.readouterr().out == (
        "[agent] Experiment request: count=2, interventions=A=4.0, C=1.0, "
        "budget=3\n"
        "[agent] Experiment complete: budget remaining=1\n"
    )


def test_tools_cannot_exceed_budget(capsys):
    world = StubWorld()
    scientist = Scientist(1)
    agent = main_module.build_scientist_agent(world, scientist, model="test-model")

    output = invoke(
        get_tool(agent, "experiment"),
        {"n_experiments": 2, "A": None, "B": None, "C": None},
    )

    assert output == "Failed to run. Experiments exceed the remaining budget."
    assert invoke(get_tool(agent, "budget")) == 1
    assert scientist.budget == 1
    assert world.calls == []
    assert capsys.readouterr().out == (
        "[agent] Experiment request: count=2, interventions=none, budget=1\n"
        "[agent] Experiment rejected: Failed to run. Experiments exceed the "
        "remaining budget.\n"
        "[agent] Budget check: 1 remaining\n"
    )


def test_main_runs_agent_workflow_without_calling_api(monkeypatch, capsys):
    captured = {}

    class StubRunner:
        @staticmethod
        def run_sync(agent, question, max_turns):
            captured["agent"] = agent
            captured["question"] = question
            captured["max_turns"] = max_turns
            return SimpleNamespace(final_output="Investigation concluded. A -> B.")

    monkeypatch.setattr(main_module, "World", StubWorld)

    def model_factory(model_name):
        captured["model_name"] = model_name
        return "gemini-test-model"

    output = main_module.main(
        question="Find the graph.",
        experimental_budget=4,
        model_name="gemini-test",
        runner=StubRunner,
        model_factory=model_factory,
    )

    assert output == "Investigation concluded. A -> B."
    assert captured["question"] == "Find the graph."
    assert captured["max_turns"] == 13
    assert captured["model_name"] == "gemini-test"
    assert captured["agent"].model == "gemini-test-model"
    assert {tool.name for tool in captured["agent"].tools} == {
        "experiment",
        "budget",
    }
    assert capsys.readouterr().out == (
        "Investigation concluded. A -> B.\nRemaining budget: 4\n"
    )


def test_create_gemini_model_uses_exported_key(monkeypatch):
    captured = {}

    def async_client(**kwargs):
        captured["client"] = kwargs
        return "gemini-client"

    def chat_model(**kwargs):
        captured["model"] = kwargs
        return "configured-model"

    def disable_tracing(disabled):
        captured["tracing_disabled"] = disabled

    monkeypatch.setenv("GEMINI_API_KEY", "secret-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-custom-model")
    monkeypatch.setattr(main_module, "AsyncOpenAI", async_client)
    monkeypatch.setattr(main_module, "OpenAIChatCompletionsModel", chat_model)
    monkeypatch.setattr(main_module, "set_tracing_disabled", disable_tracing)

    result = main_module.create_gemini_model()

    assert result == "configured-model"
    assert captured["tracing_disabled"] is True
    assert captured["client"] == {
        "api_key": "secret-key",
        "base_url": main_module.GEMINI_BASE_URL,
    }
    assert captured["model"] == {
        "model": "gemini-custom-model",
        "openai_client": "gemini-client",
    }


def test_create_gemini_model_requires_exported_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is missing"):
        main_module.create_gemini_model()
