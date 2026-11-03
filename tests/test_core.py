"""Offline tests (no API key): the safety rules and the numbers must hold."""
import numpy as np
import pytest

from wattara.household import Household
from wattara.simulator import SimulatedPlug, generate_with, verify, APPLIANCES

PLAN_OK = {"actions": [
    {"action": "cut_standby", "appliance": "console"},
    {"action": "cut_standby", "appliance": "tv"},
    {"action": "reduce_usage_minutes", "appliance": "water_heater", "minutes": 10},
]}


@pytest.fixture(scope="module")
def home():
    return Household()


def test_baseline(home):
    b = home.baseline()
    assert b["kwh_month"] == pytest.approx(163.7, abs=0.1)
    assert b["cost_month"] == pytest.approx(114.56, abs=0.1)


def test_valid_plan_totals(home):
    r = home.validate_plan(PLAN_OK)
    assert r["valid"] and not r["errors"]
    assert r["total_kwh_saved"] == pytest.approx(26.01, abs=0.05)
    assert r["pct_saved"] == pytest.approx(15.9, abs=0.1)


@pytest.mark.parametrize("appliance", ["fridge", "router"])
def test_protected_appliances_are_blocked(home, appliance):
    r = home.validate_plan({"actions": [{"action": "cut_standby", "appliance": appliance}]})
    assert not r["valid"] and "protected" in r["errors"][0]


def test_invalid_minutes_are_rejected(home):
    r = home.simulate_action("reduce_usage_minutes", "water_heater", minutes=45)
    assert not r["ok"]


def test_unknown_action_and_duplicates(home):
    assert not home.simulate_action("explode", "tv")["ok"]
    r = home.validate_plan({"actions": [{"action": "cut_standby", "appliance": "tv"},
                                        {"action": "cut_standby", "appliance": "tv"}]})
    assert not r["valid"]


def test_generator_is_reproducible():
    a = generate_with(APPLIANCES, 30, 0)
    b = generate_with(APPLIANCES, 30, 0)
    assert np.allclose(a.values, b.values)


def test_verification_confirms_a_full_plan(home):
    plan = home.validate_plan(PLAN_OK)
    ref, plug = SimulatedPlug(home.appliances), SimulatedPlug(home.appliances)
    before = [ref.read_month(s) for s in range(10, 16)]
    plug.apply(plan, compliance=1.0)
    after = [plug.read_month(s) for s in range(20, 23)]
    rep = verify(plan, before, after)
    assert rep["status"] == "CONFIRMED"
    plug.undo()
    assert plug.applied is None


def test_partial_compliance_is_not_confirmed(home):
    plan = home.validate_plan(PLAN_OK)
    ref, plug = SimulatedPlug(home.appliances), SimulatedPlug(home.appliances)
    before = [ref.read_month(s) for s in range(10, 16)]
    plug.apply(plan, compliance=0.3)
    after = [plug.read_month(s) for s in range(20, 23)]
    assert verify(plan, before, after)["status"] != "CONFIRMED"
