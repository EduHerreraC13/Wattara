"""Household model: baseline consumption, standby loads, savings simulator and plan validator.

KEY DESIGN RULE: the language model never does arithmetic. Every kWh / soles figure comes from
this module, and `validate_plan` re-computes the totals from scratch (it never trusts the model).
"""
import copy

from . import config
from .simulator import APPLIANCES, generate_with

ALLOWED_ACTIONS = ("cut_standby", "reduce_usage_minutes")


class Household:
    def __init__(self, appliances=None, days=30, seed=0, tariff=config.TARIFF_PER_KWH):
        self.appliances = copy.deepcopy(appliances or APPLIANCES)
        self.days = days
        self.tariff = tariff
        self.df = generate_with(self.appliances, days, seed)   # hourly kWh per appliance (simulated)

    # ------------------------------ measurements ------------------------------
    def baseline(self):
        """Projected monthly consumption (kWh) and cost (soles) of the household as it is."""
        kwh_month = float(self.df.sum().sum() / self.days * 30)
        return {"kwh_month": round(kwh_month, 1),
                "cost_month": round(kwh_month * self.tariff, 2)}

    def standby_kwh_day(self, name):
        """Idle (standby) energy per day, measured from the 0-6 h readings when nothing is in use."""
        a = self.appliances[name]
        if a.get("always_on") or "duty" in a:          # router / fridge are never "idle"
            return 0.0
        night = float(self.df[name].between_time("00:00", "05:59").median())
        hours_on = sum(e - s for s, e in a.get("hours", []))
        return night * (24 - hours_on)

    def standby_loads(self):
        """Monthly standby energy per appliance (only those that matter)."""
        rows = []
        for name, a in self.appliances.items():
            kwh = self.standby_kwh_day(name) * 30
            if kwh > 0.05:
                rows.append({"appliance": name, "kwh_month": round(kwh, 2),
                             "cost_month": round(kwh * self.tariff, 2),
                             "protected": a["protected"]})
        return rows

    # -------------------------------- actions ---------------------------------
    def simulate_action(self, action, appliance, minutes=0):
        """Estimate the monthly savings of ONE action. Nothing is applied."""
        if appliance not in self.appliances:
            return {"ok": False, "error": f"unknown appliance: {appliance}"}
        a = self.appliances[appliance]
        if a["protected"]:                              # SAFETY RULE, enforced in code
            return {"ok": False, "error": f"{appliance} is protected and cannot be controlled"}
        if action == "cut_standby":
            kwh = self.standby_kwh_day(appliance) * 30
        elif action == "reduce_usage_minutes":
            if "hours" not in a or not (0 < minutes <= config.MAX_CUT_MIN):
                return {"ok": False,
                        "error": f"minutes must be 1-{config.MAX_CUT_MIN} on a scheduled appliance"}
            kwh = a["watts"] / 1000 * minutes / 60 * 30
        else:
            return {"ok": False, "error": f"unknown action: {action}"}
        return {"ok": True, "action": action, "appliance": appliance, "minutes": minutes,
                "kwh_month_saved": round(kwh, 2),
                "cost_month_saved": round(kwh * self.tariff, 2)}

    def validate_plan(self, plan):
        """Official check of a plan proposed by the model: re-simulates every action.

        Returns the validated totals, or the list of errors (e.g. a protected appliance).
        """
        errors, results, seen = [], [], set()
        for step in plan.get("actions", []):
            key = (step.get("action"), step.get("appliance"))
            if key in seen:
                errors.append(f"duplicate action: {key}")
                continue
            seen.add(key)
            r = self.simulate_action(step.get("action"), step.get("appliance"),
                                     step.get("minutes", 0))
            if r["ok"]:
                results.append(r)
            else:
                errors.append(r["error"])
        kwh = round(sum(r["kwh_month_saved"] for r in results), 2)
        cost = round(sum(r["cost_month_saved"] for r in results), 2)
        base = self.baseline()["kwh_month"]
        return {"valid": not errors and bool(results), "errors": errors,
                "actions": results, "total_kwh_saved": kwh,
                "total_cost_saved": cost, "pct_saved": round(kwh / base * 100, 1)}
