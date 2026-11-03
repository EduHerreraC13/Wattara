"""End-to-end demo: goal -> agent plan -> human approval -> apply -> verify -> undo.

  python run_demo.py --offline                 # no API key needed (fixed example plan)
  python run_demo.py --goal "Cut my bill 15% but I need hot water in the mornings"
  python run_demo.py --goal "..." --yes --compliance 0.6   # auto-approve; user follows only 60 % of the plan
"""
import argparse

from wattara import config
from wattara.household import Household
from wattara.simulator import SimulatedPlug, verify

# Used by --offline so judges can see the pipeline without an API key.
EXAMPLE_PLAN = {"actions": [
    {"action": "cut_standby", "appliance": "console"},
    {"action": "cut_standby", "appliance": "tv"},
    {"action": "reduce_usage_minutes", "appliance": "water_heater", "minutes": 10},
]}


def main():
    ap = argparse.ArgumentParser(description="Wattara demo")
    ap.add_argument("--goal", default="I want to cut my electricity bill by 15%, but we need hot water in the mornings.")
    ap.add_argument("--offline", action="store_true", help="skip the model and use a fixed example plan")
    ap.add_argument("--yes", action="store_true", help="approve the plan automatically")
    ap.add_argument("--compliance", type=float, default=1.0, help="share of the plan the user actually follows (0-1)")
    args = ap.parse_args()

    home = Household()
    print("Baseline:", home.baseline(), "(simulated household, example tariff)")

    if args.offline:
        plan = home.validate_plan(EXAMPLE_PLAN)
    else:
        from wattara.agent import WattaraAgent          # imported here so --offline never needs the API
        agent = WattaraAgent(home)
        print("GOAL:", args.goal, "\n")
        result = agent.run(args.goal)
        print("\n=== AGENT ANSWER ===\n" + str(result["answer"]))
        plan = result["plan"]
        if plan is None:
            print("\nNo valid plan was submitted, nothing to apply.")
            return

    print("\n=== OFFICIAL PLAN (computed by code) ===")
    for s in plan["actions"]:
        print(f" - {s['action']} on {s['appliance']} {('(' + str(s['minutes']) + ' min/day)') if s['minutes'] else ''}"
              f" -> {s['kwh_month_saved']} kWh/month")
    print(f"Expected: {plan['total_kwh_saved']} kWh/month | S/ {plan['total_cost_saved']} | {plan['pct_saved']} % of the bill")

    if not args.yes and input("\nApprove and apply this plan? (y/n): ").strip().lower() not in ("y", "s", "yes", "si", "sí"):
        print("Plan NOT applied. Nothing changed.")
        return

    # Apply on a simulated plug, then compare predicted vs measured savings over several months.
    reference = SimulatedPlug(home.appliances)                      # household without the plan
    before = [reference.read_month(seed) for seed in range(10, 16)]  # 6 reference months
    plug = SimulatedPlug(home.appliances)
    plug.apply(plan, compliance=args.compliance)
    after = [plug.read_month(seed) for seed in range(20, 23)]        # 3 months with the plan applied
    report = verify(plan, before, after)
    print("\n=== VERIFICATION (simulated measurements) ===")
    for k, v in report.items():
        print(f"{k}: {v}")
    if report["status"] != "CONFIRMED":
        print("Note: measured savings are below the prediction.")
    plug.undo()
    print("Plan reverted (every action is reversible).")


if __name__ == "__main__":
    main()
