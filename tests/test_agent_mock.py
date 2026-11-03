"""Agent-loop tests with a scripted fake model (no API key, no cost)."""
import json
from types import SimpleNamespace as NS

import pytest

from wattara import config, llm
from wattara.agent import WattaraAgent
from wattara.evaluate import grade


def call(i, name, args=None):
    return NS(id=f"c{i}", function=NS(name=name, arguments=json.dumps(args or {})))


def reply(content=None, calls=None, finish="stop"):
    return NS(choices=[NS(message=NS(content=content, tool_calls=calls), finish_reason=finish)])


@pytest.fixture(scope="module")
def agent():
    return WattaraAgent(verbose=False)


def script(monkeypatch, replies):
    """Make llm.safe_call return the scripted replies in order."""
    it = iter(replies)
    monkeypatch.setattr(llm, "safe_call", lambda *a, **k: next(it))


def test_normal_flow_and_grounded_numbers(agent, monkeypatch):
    plan = [{"action": "cut_standby", "appliance": "console"},
            {"action": "reduce_usage_minutes", "appliance": "water_heater", "minutes": 10}]
    script(monkeypatch, [
        reply(calls=[call(1, "get_baseline"), call(2, "get_anomalies")]),
        reply(calls=[call(3, "submit_plan", {"actions": plan})]),
        reply("Plan saves 24.12 kWh. Do you approve?"),
    ])
    res = agent.run("cut my bill 10%")
    assert res["plan"]["valid"] and not res["used_fallback"]
    assert res["tools_used"] == ["get_baseline", "get_anomalies", "submit_plan"]


def test_step_cap_triggers_forced_final_answer(agent, monkeypatch):
    """Regression test for the g03 failure: the loop ran out of steps and returned no answer."""
    endless = [reply(calls=[call(i, "get_baseline")]) for i in range(config.MAX_STEPS)]
    script(monkeypatch, endless + [reply("Final answer after the step cap.")])
    res = agent.run("whatever")
    assert res["used_fallback"] and res["answer"] == "Final answer after the step cap."


def test_empty_answer_uses_fallback(agent, monkeypatch):
    script(monkeypatch, [reply(None, None, finish="length"), reply("Recovered answer.")])
    res = agent.run("whatever")
    assert res["truncated"] and res["answer"] == "Recovered answer."


def test_fridge_in_a_plan_is_blocked_by_code_even_if_the_model_tries(agent, monkeypatch):
    bad = [{"action": "cut_standby", "appliance": "fridge"}]
    script(monkeypatch, [reply(calls=[call(1, "submit_plan", {"actions": bad})]),
                         reply("I cannot control the fridge.")])
    res = agent.run("turn the fridge off")
    assert res["plan"] is None


def test_grade_flags_invented_numbers(agent):
    res = {"plan": None, "answer": "You will save 99.99 kWh. Approve?", "grounded_numbers": {26.01},
           "tools_used": [], "used_fallback": False, "truncated": False}
    checks, ungrounded = grade("cut 10%", "vague", None, [], res, agent.home.appliances)
    assert not checks["numbers_grounded"] and 99.99 in ungrounded
