"""Thin wrapper around the Meta Model API with token accounting and a local spending cap.

Every model call in the project goes through `safe_call`, so cost is always tracked.
NOTE: Meta offers no hard spending limit, so this cap is only a local safeguard.
"""
from . import config

usage = {"usd": 0.0, "calls": 0, "tin": 0, "tout": 0}   # running totals for this process
budget = {"usd": config.BUDGET_USD}                       # mutable cap (see set_budget)
verbose = {"on": True}                                    # print one usage line per call
_client = None


def get_client():
    """Create the OpenAI-compatible client lazily, so offline scripts never need a key."""
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=config.get_api_key(), base_url=config.API_BASE_URL)
    return _client


def set_budget(usd: float) -> None:
    budget["usd"] = float(usd)


def reset_usage() -> None:
    usage.update(usd=0.0, calls=0, tin=0, tout=0)


def safe_call(messages, tools=None, max_tokens=None, tool_choice="auto"):
    """Call the model, update the usage counters and refuse to go beyond the local budget."""
    if usage["usd"] >= budget["usd"]:
        raise RuntimeError(
            f"Local budget reached: ${usage['usd']:.4f} >= ${budget['usd']:.2f}. "
            "Raise it with WATTARA_BUDGET_USD or llm.set_budget(...)."
        )
    kwargs = dict(model=config.MODEL, messages=messages,
                  max_tokens=max_tokens or config.AGENT_MAX_TOKENS)
    if tools:
        kwargs.update(tools=tools, tool_choice=tool_choice)
    if config.REASONING_EFFORT:
        kwargs["reasoning_effort"] = config.REASONING_EFFORT

    r = get_client().chat.completions.create(**kwargs)

    tin, tout = r.usage.prompt_tokens, r.usage.completion_tokens
    cost = tin * config.PRICE_IN_PER_M / 1e6 + tout * config.PRICE_OUT_PER_M / 1e6
    usage["usd"] += cost
    usage["calls"] += 1
    usage["tin"] += tin
    usage["tout"] += tout
    if verbose["on"]:
        print(f"[usage] +{tin} in / +{tout} out | ${cost:.6f} | "
              f"total ${usage['usd']:.6f} ({usage['calls']} calls)")
    return r
