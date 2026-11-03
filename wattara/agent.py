"""Wattara agent: Muse Spark reasons with tools; code does all the numbers and enforces the rules.

Flow per request
  1. get_anomalies / get_baseline / get_standby_loads  -> understand the household
  2. simulate_action                                  -> evaluate options (nothing is applied)
  3. submit_plan                                      -> OFFICIAL validation (re-computed in code)
  4. the model presents the plan and asks for approval  (applying it is a separate, human-approved step)
"""
import json

from . import config, llm
from .household import Household
from .ml import AnomalyService

SYSTEM_PROMPT = """You are Wattara, a home energy assistant for a household in Peru. Money is in Peruvian soles (PEN); energy in kWh.
Rules:
1. Never do arithmetic or invent numbers. Every kWh or PEN figure you state must come from a tool result.
2. Workflow: call get_baseline and get_standby_loads, use simulate_action to evaluate options, then call submit_plan with the chosen actions. submit_plan returns the official validated totals.
3. Only two actions exist: cut_standby, and reduce_usage_minutes (1-20 minutes per day, on scheduled appliances). Protected appliances (such as the fridge and router) can never be controlled. If asked to control one, refuse briefly and offer alternatives.
4. Respect the user's comfort needs. Shortening water-heater minutes is fine when the user still needs hot water, but never remove it entirely.
4b. Choose the LEAST intrusive plan that reaches the goal. Start with cut_standby actions, then use the smallest reduction needed. Do not exceed the goal by more than about 2 percentage points.
5. If the goal cannot be reached with the allowed actions, submit the best possible plan and say honestly how far it gets.
6. Nothing is applied until the user approves: present the plan and ask for approval.
7. Answer in the user's language, in at most 120 words, quoting totals exactly as returned by submit_plan.
Additional rules:
5b. Call get_anomalies before planning. If an appliance behaves abnormally, tell the user in one sentence using only the fields returned (never invent watts or soles), and prioritize actions on that appliance. If the anomalous appliance is protected, say it cannot be controlled and recommend having it inspected."""

# Field names that carry money get an explicit "_pen" suffix, so the model cannot assume dollars.
_RENAME = {"cost_month": "cost_month_pen",
           "cost_month_saved": "cost_month_saved_pen",
           "total_cost_saved": "total_cost_saved_pen"}


def with_units(x):
    """Recursively rename money fields so that tool results state their currency."""
    if isinstance(x, dict):
        return {_RENAME.get(k, k): with_units(v) for k, v in x.items()}
    if isinstance(x, list):
        return [with_units(i) for i in x]
    return x


def build_tools_spec(appliance_names):
    """JSON-schema description of the tools the model may call."""
    action_enum = {"type": "string", "enum": ["cut_standby", "reduce_usage_minutes"]}
    appliance_enum = {"type": "string", "enum": list(appliance_names)}
    return [
        {"type": "function", "function": {
            "name": "get_baseline",
            "description": "Baseline monthly consumption (kWh) and cost in Peruvian soles (PEN).",
            "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {
            "name": "get_standby_loads",
            "description": "Monthly standby (idle) consumption per appliance, with cost in PEN and whether the appliance is protected.",
            "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {
            "name": "simulate_action",
            "description": "Estimate monthly savings (kWh and PEN) of ONE action. Does not apply anything.",
            "parameters": {"type": "object", "properties": {
                "action": action_enum, "appliance": appliance_enum,
                "minutes": {"type": "integer",
                            "description": "Daily minutes to cut (1-20); only for reduce_usage_minutes."}},
                "required": ["action", "appliance"]}}},
        {"type": "function", "function": {
            "name": "submit_plan",
            "description": "Submit the final plan for official validation. Returns the validated totals (kWh, PEN, % of baseline) or errors to fix.",
            "parameters": {"type": "object", "properties": {
                "actions": {"type": "array", "items": {"type": "object", "properties": {
                    "action": action_enum, "appliance": appliance_enum,
                    "minutes": {"type": "integer"}},
                    "required": ["action", "appliance"]}}},
                "required": ["actions"]}}},
        {"type": "function", "function": {
            "name": "get_anomalies",
            "description": "Checks recent consumption for abnormal behavior per appliance. Returns, for each anomalous appliance, how many times above normal it is (times_normal) and whether it is protected. Reports no watts or soles.",
            "parameters": {"type": "object", "properties": {
                "last_days": {"type": "integer", "description": "Days to inspect, default 14."}}}}},
    ]


def _collect_numbers(obj, acc):
    """Gather every number found in a tool result (used to check the model's answer)."""
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _collect_numbers(v, acc)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        acc.add(round(float(obj), 2))


class WattaraAgent:
    def __init__(self, household=None, anomalies=None, verbose=True):
        self.home = household or Household()
        self.anomalies = anomalies or AnomalyService(self.home.appliances)
        self.verbose = verbose
        self.last_plan = None                       # last plan that passed the official validation
        self.tools_spec = build_tools_spec(self.home.appliances)
        self.funcs = {                              # tool name -> Python function (all numbers come from here)
            "get_baseline": lambda: with_units({**self.home.baseline(), "currency": "PEN"}),
            "get_standby_loads": lambda: with_units(self.home.standby_loads()),
            "simulate_action": lambda action, appliance, minutes=0:
                with_units(self.home.simulate_action(action, appliance, minutes)),
            "submit_plan": self._submit_plan,
            "get_anomalies": lambda last_days=14: self.anomalies.report(last_days),
        }

    def _submit_plan(self, actions):
        result = self.home.validate_plan({"actions": actions})
        if result["valid"]:
            self.last_plan = result                 # keep the OFFICIAL plan (computed by code, not by the model)
        return with_units(result)

    def _force_final_answer(self, messages):
        """If the tool loop hits its step cap, ask once more for the final answer (tools disabled)."""
        messages.append({"role": "user", "content":
                         "Stop calling tools. Give your final answer to the user now, "
                         "using only the tool results above."})
        try:
            r = llm.safe_call(messages, tools=self.tools_spec, tool_choice="none")
            return r.choices[0].message.content
        except Exception as e:                      # never crash the demo because of this extra call
            return None

    def run(self, goal):
        """Run the agent on a natural-language goal. Returns a dict with the answer and audit data."""
        self.last_plan = None
        tools_used, grounded, finish_reasons = [], set(), []
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": goal}]
        answer = None

        for _ in range(config.MAX_STEPS):
            r = llm.safe_call(messages, tools=self.tools_spec)
            choice = r.choices[0]
            msg = choice.message
            finish_reasons.append(getattr(choice, "finish_reason", None))
            if not msg.tool_calls:                  # the model answered in plain text: we are done
                answer = msg.content
                break
            messages.append(msg)
            for call in msg.tool_calls:
                args = {}
                try:
                    args = json.loads(call.function.arguments or "{}")
                    result = self.funcs[call.function.name](**args)
                except Exception as e:              # a bad tool call is reported back to the model
                    result = {"error": str(e)}
                tools_used.append(call.function.name)
                _collect_numbers(result, grounded)
                if self.verbose:
                    print(f"  [tool] {call.function.name}({json.dumps(args, ensure_ascii=False)})")
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": json.dumps(result)})

        used_fallback = not answer                  # step cap reached, or empty/truncated answer
        if used_fallback:
            answer = self._force_final_answer(messages) or "(no final answer was produced)"

        return {"goal": goal, "answer": answer, "plan": self.last_plan,
                "tools_used": tools_used, "grounded_numbers": grounded,
                "used_fallback": used_fallback,
                "truncated": any(fr == "length" for fr in finish_reasons)}
