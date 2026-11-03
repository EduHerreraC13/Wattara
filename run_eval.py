"""Evaluate the agent on 15 hand-written goals (uses the API: roughly $0.5 of tokens).

  python run_eval.py                 # all goals (finished ones are cached in results/eval_results.json)
  python run_eval.py g02 g10         # only these goals
"""
import sys

from wattara.agent import WattaraAgent
from wattara.evaluate import run_eval, summarize

if __name__ == "__main__":
    ids = sys.argv[1:] or None
    results = run_eval(WattaraAgent(), ids=ids)
    summarize(results)
