"""Household simulator, simulated smart plug and savings verification.

Everything here is SIMULATED data. There is no hardware: `SimulatedPlug` implements the
same small interface (apply / undo / read_month) that a real smart-plug adapter would have.
"""
import copy

import numpy as np
import pandas as pd

from . import config

# Typical EXAMPLE values (watts, duty cycle, standby watts). Adjust and cite your sources.
#   hours   -> list of (start_hour, end_hour) when the appliance is in use
#   duty    -> fraction of time a cycling appliance (fridge) is drawing power
#   protected -> the agent may NEVER control it (safety rule enforced in code)
APPLIANCES = {
    "fridge":       {"watts": 120,  "duty": 0.35, "standby": 0,  "protected": True},
    "water_heater": {"watts": 3500, "hours": [(6, 7)],  "standby": 0,  "protected": False},
    "tv":           {"watts": 100,  "hours": [(19, 22)], "standby": 3,  "protected": False},
    "router":       {"watts": 10,   "always_on": True, "standby": 0,  "protected": True},
    "console":      {"watts": 120,  "hours": [(20, 22)], "standby": 10, "protected": False},
}


def generate_with(apps, days=30, seed=0):
    """Hourly energy (kWh) per appliance for `days` days, with 8 % random noise.

    An appliance may carry a "scale" (used by SimulatedPlug to model shorter usage).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-10-01", periods=days * 24, freq="h")
    out = {}
    for name, a in apps.items():
        scale = a.get("scale", 1.0)
        kwh = np.zeros(len(idx))
        for i, t in enumerate(idx):
            h = t.hour
            if a.get("always_on"):
                p = a["watts"]
            elif "duty" in a:
                p = a["watts"] * a["duty"]
            elif any(s <= h < e for s, e in a.get("hours", [])):
                p = a["watts"] * scale
            else:
                p = a.get("standby", 0)
            kwh[i] = max(0, p * rng.normal(1, 0.08)) / 1000
        out[name] = kwh
    return pd.DataFrame(out, index=idx)


class SimulatedPlug:
    """Applies a validated plan to a simulated household (a real plug would expose the same methods)."""

    def __init__(self, appliances=None):
        self.base = appliances or APPLIANCES
        self.applied = None
        self.compliance = 1.0          # 1.0 = user follows the plan fully, 0.6 = only partly

    def _config(self):
        apps = copy.deepcopy(self.base)
        if not self.applied:
            return apps
        for step in self.applied["actions"]:
            a = apps[step["appliance"]]
            if step["action"] == "cut_standby":
                a["standby"] = a.get("standby", 0) * (1 - self.compliance)
            elif step["action"] == "reduce_usage_minutes":
                hours_on = sum(e - s for s, e in a["hours"])
                a["scale"] = a.get("scale", 1.0) - self.compliance * (step["minutes"] / 60) / hours_on
        return apps

    def apply(self, plan_result, compliance=1.0):
        self.applied, self.compliance = plan_result, compliance

    def undo(self):
        """Every action is reversible."""
        self.applied = None

    def read_month(self, seed):
        """One simulated month of readings (a different `seed` = a different month of noise)."""
        return generate_with(self._config(), 30, seed)


def monthly_kwh(df):
    return float(df.sum().sum() / (len(df) / 24) * 30)


def verify(plan_result, dfs_before, dfs_after):
    """Compare PREDICTED savings with MEASURED savings.

    Uses the mean of several months before/after, because a single month is too noisy
    to judge a saving of a few kWh.
    """
    before = float(np.mean([monthly_kwh(d) for d in dfs_before]))
    after = float(np.mean([monthly_kwh(d) for d in dfs_after]))
    measured = round(before - after, 2)
    predicted = plan_result["total_kwh_saved"]
    ratio = measured / predicted if predicted else 0
    status = "CONFIRMED" if ratio >= 0.85 else "PARTIAL" if ratio >= 0.5 else "NOT REACHED"
    return {"status": status,
            "months_before": len(dfs_before), "months_after": len(dfs_after),
            "predicted_kwh": predicted, "measured_kwh": measured,
            "measured_pct_of_bill": round(measured / before * 100, 1),
            "measured_cost_saved_pen": round(measured * config.TARIFF_PER_KWH, 2),
            "co2_avoided_kg": round(measured * config.EMISSION_FACTOR_KG_PER_KWH, 2)}
