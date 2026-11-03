# Wattara

**An energy-planning agent for households in Peru.**  
Built for the **Meta Global AI Developer Hackathon 2026 — Indie Developer track** (*Build for Your World*).

Wattara turns a household's electricity-saving goal into a concrete plan. Instead of just recommending generic habits, it looks at consumption patterns, checks possible actions, calculates their expected savings, and asks for permission before applying anything.

For this hackathon, Wattara runs on **synthetic household data and simulated devices**. It does not connect to a real electricity meter or smart plug, and the savings reported here are not real-world savings.

## The problem

An electricity bill tells a household how much it owes, but not necessarily which appliances or habits explain the cost. Even when someone wants to save energy, the trade-offs matter: turning off equipment is not always safe, and cutting consumption should not mean giving up everyday needs.

I built Wattara to explore a different approach: the user describes a goal in everyday language, and the system produces a plan that can be checked before anyone acts on it.

For example:

> “Quiero reducir mi factura de luz un 15%, pero necesito agua caliente por las mañanas.”

## What the prototype does

1. **Reads the household baseline:** estimated monthly consumption, bill and standby loads.
2. **Finds unusual usage:** a per-appliance anomaly detector highlights consumption patterns worth investigating.
3. **Explores alternatives:** Muse Spark 1.3, accessed through the Meta Model API, calls Python tools to simulate allowed actions.
4. **Checks the proposal:** Python independently recalculates the numbers and rejects invalid actions. The refrigerator and router are protected in code.
5. **Requests approval:** a proposed plan is not applied until the user confirms it. The CLI also has a `--yes` option for automated demonstrations.
6. **Checks the outcome:** a simulated device adapter compares predicted and simulated measured savings, then restores the original state.

The language model decides *what to explore* and explains the options. **Python calculates the savings and enforces the action rules.**

## Architecture

```text
User goal (Spanish or English)
          |
          v
Muse Spark 1.3 (Meta Model API)
          |
          +--> get_baseline
          +--> get_standby_loads
          +--> get_anomalies
          +--> simulate_action
          |
          v
      submit_plan
          |
          v
Python validator (recalculates totals; rejects disallowed actions)
          |
          v
Show proposal --> User approval --> SimulatedPlug.apply()
                                           |
                                           v
                              Verify savings --> undo()
```

The supported actions are deliberately limited to **cutting standby consumption** and **reducing daily usage minutes** for eligible scheduled appliances. No physical hardware is controlled.

## Run it locally

Wattara is a Python project; **Google Colab is not required**. The setup below was tested on **Windows with Python 3.13.3**. Other Python versions may work, but dependency compatibility can vary because `requirements.txt` currently uses unpinned package versions.

### Windows (PowerShell / VS Code)

Open the repository folder in VS Code (`File > Open Folder`), then open its terminal. Run these commands from the folder containing `requirements.txt` and `run_demo.py`:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

You can run the project **without activating the environment**. Using `.\.venv\Scripts\python.exe` makes sure every command uses the correct installation.

Check the offline components first:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe run_demo.py --offline --yes
.\.venv\Scripts\python.exe run_ml_eval.py
```

Expected results from the tested setup: **14 passing tests**, an offline plan with about **26.01 kWh/month** in predicted savings, and the forecasting/anomaly-detection metrics shown below. The offline example uses a fixed plan: it does **not** call Muse Spark.

If VS Code does not show `Python: Select Interpreter`, install or enable Microsoft's **Python** extension and reload the window. Selecting the interpreter is useful for the editor, but is not necessary when you use the explicit `.venv` command shown above.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
python run_demo.py --offline --yes
python run_ml_eval.py
```

### Run the agent with Meta Model API

1. Obtain your own key from [Meta's developer portal](https://dev.meta.ai/api-keys). Account requirements and API charges are determined by Meta.
2. Create a local `.env` file from the supplied template.
3. Add your key to `META_API_KEY` inside `.env`. **Never commit or share that file.**

On Windows:

```powershell
Copy-Item .env.example .env
```

On macOS / Linux:

```bash
cp .env.example .env
```

Your local `.env` should contain:

```dotenv
META_API_KEY=replace_with_your_own_key
```

This project's variable is named `META_API_KEY` (see `wattara/config.py`). Judges and other developers should use **their own** API keys.

Run an interactive example on Windows:

```powershell
.\.venv\Scripts\python.exe run_demo.py --goal "Quiero reducir mi factura de luz un 15%, pero necesito agua caliente por las mañanas."
```

Or, with the environment activated on any supported platform:

```bash
python run_demo.py --goal "Turn the fridge off at night to save energy."
```

The second example tests a protected-device request. A working key is needed for live agent calls; the offline tests do not need one.

**Usage and cost:** In one local live demonstration, the agent made **5 API calls**, and the application's token-price estimate was **US$0.0298**. Actual billing can differ. `WATTARA_BUDGET_USD` (default `0.50`) provides a **local, estimated per-process safeguard** for normal demos, not a provider-enforced spending limit. The 15-goal evaluation script raises its local budget to at least **US$2.00**; review that cost before running it. Avoid entering real household or personal data; the prototype was designed for synthetic data.

Optional settings are documented in `.env.example` and `wattara/config.py`, including `WATTARA_MODEL`, `WATTARA_MAX_TOKENS`, `WATTARA_MAX_STEPS` and `WATTARA_REASONING_EFFORT`.

## Common setup issues

| Symptom | What to check |
| --- | --- |
| `No suitable Python runtime found` after `py -3.11 -m venv .venv` | Python 3.11 may not be installed. Run `python --version`, then create the environment with `python -m venv .venv` using the installed Python. |
| `.\.venv\Scripts\python.exe` is not recognized | The virtual environment was not created. Run `python -m venv .venv` successfully before installing packages. |
| `No module named pytest` or `No module named lightgbm` | Install dependencies using `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`. Avoid mixing global Python and `.venv`. |
| `Python: Select Interpreter` does not appear in VS Code | Install/enable the Microsoft Python extension, reload VS Code, or continue using the explicit `.venv` Python path. |
| PowerShell blocks `Activate.ps1` | Activation is optional. Run scripts directly with `.\.venv\Scripts\python.exe`. |
| `META_API_KEY not found` | Create `.env` next to `run_demo.py` and set `META_API_KEY`. Do not paste the key into source code. |
| `Pandas4Warning` about `dayofyear` / `dayofweek` | These are deprecation warnings, not test failures. The tested installation reported **14 passed, 425 warnings**; date-property aliases should be updated in a future cleanup. |
| Agent answer is empty, cut off or reaches the step limit | Reasoning can use output tokens. The default is now 3000 output tokens with an eight-step tool loop and a final-answer fallback; behavior can still vary between runs. |

## Results on simulated data

The figures below were reproduced locally with `run_ml_eval.py` and `run_demo.py`. They evaluate the implementation **within the simulator**, not its performance in actual Peruvian households.

### Bill forecasting

| Task | Wattara | Comparison |
| --- | ---: | ---: |
| Daily consumption prediction, 1-day-ahead MAE | **0.755 kWh** | 0.947 kWh (same weekday last week); **0.752 kWh** (7-day average) |
| Monthly bill projection from day 10, mean percentage error over 4 test months | **1.70%** | **5.09%** (naive projection) |

The monthly projection is more accurate than its chosen baseline. The **daily model does not outperform the 7-day-average baseline**, which is an important limitation rather than a result to hide.

### Appliance anomaly detection

| Method | Precision | Recall |
| --- | ---: | ---: |
| Per-appliance residual scores | **0.962** | **1.00** |
| Threshold on total daily household consumption | 1.00 | 0.04 |

The synthetic test injected **75 faulty days**, all of which were detected by the per-appliance approach; it also produced **3 false alerts over 150 healthy days**. Tested fault types included abnormal console standby use, a degraded refrigerator and a water-heater leak. These fault patterns were deliberately simulated.

### Example plan and verification

For a simulated baseline of **163.7 kWh/month** (**S/ 114.56**, using an example tariff):

| Suggested action | Estimated savings |
| --- | ---: |
| Cut console standby consumption | 6.62 kWh/month |
| Cut TV standby consumption | 1.89 kWh/month |
| Reduce water-heater use by 10 minutes/day | 17.50 kWh/month |
| **Validated plan total** | **26.01 kWh/month (15.9%; S/ 18.20)** |

After manual approval, the simulator reported **26.6 kWh/month** in measured-model savings and `CONFIRMED`, then reverted the actions. Both the baseline and the follow-up measurements are simulated; this does not prove a real home would save the same amount or retain hot water at the desired times.

### Agent behavior: earlier exploratory evaluation

An earlier evaluation used **15 hand-written goals** in Spanish and English, with Python checks for protected devices, arithmetic grounding, feasibility and refusals. The reported final run found **0/15 protected-device actions**, **0/15 detected unsupported figures**, and **3/3 refusals for unsafe requests**; **14/15 goals** passed every check in that run.

**Evidence status:** The original JSON logs for the full 15-goal evaluation are **not included in the supplied project archive**, so these historical figures cannot currently be independently inspected from this repository. The current live example above was run successfully; the full set can be regenerated with a Meta API key:

```bash
python run_eval.py
```

That command makes paid API calls and caches completed cases in `results/eval_results.json`. To rerun a specific case, such as `g03`, remove its existing cached entry first (keeping a backup), then run `python run_eval.py g03`.

In an earlier run, `g03` hit the tool-step limit. The implementation now has an eight-step cap and a final-response fallback, tested offline in `tests/test_agent_mock.py`. **A post-fix live API retest of `g03` is not yet documented.** The evaluation's text checks also use some keyword heuristics and do not replace human review.

## Design choices and limitations

- **Rules outside the model.** The validator recalculates submitted actions and rejects protected appliances in Python. Prompt instructions alone are not the safety mechanism.
- **Human confirmation.** The interactive demo asks before applying a plan. `--yes` intentionally skips that prompt for scripted tests; it only affects simulated devices.
- **No real device integration.** `SimulatedPlug` provides `apply()`, `undo()` and `read_month()`; there is no actual smart-plug control.
- **Synthetic data.** The ML training and evaluation data follow the simulator's assumptions. Detection quality and savings will need independent, real-world validation.
- **Separate simulated datasets.** The anomaly service and the household planner do not share a single real measured household. Anomaly messages therefore report relative changes, not verified monetary losses.
- **Forecast information leakage.** The monthly forecast uses actual future-month temperatures in its current evaluation rather than only temperatures available at prediction time.
- **Illustrative local parameters.** The tariff (**S/ 0.70/kWh**) and emission factor (**0.20 kg CO₂/kWh**) are placeholders, not official Peru-wide values.
- **Small action catalogue.** The planner currently supports only two action types, and it does not model comfort constraints such as actual hot-water availability in detail.

## Next steps

1. **Validate with independent data.** Use public appliance-level electricity datasets and evaluate across different household profiles, not only injected faults.
2. **Make the estimates locally meaningful.** Use documented Peruvian tariffs and appropriate emissions factors; replace future observed temperatures with forecasts available at decision time.
3. **Improve comfort constraints.** Represent schedules and non-negotiable appliance usage explicitly, rather than assuming a reduction in minutes preserves the same service.
4. **Integrate devices safely.** Add an opt-in smart-plug adapter, logging, user overrides and rollback tests before any real deployment.
5. **Strengthen repeatability.** Run the agent evaluation multiple times, save anonymized raw outputs, document the tested package versions and measure latency, cost and failure rates.
6. **Build a practical interface.** Move beyond the terminal with a simple household dashboard and clear explanations of approved versus proposed actions.

## Repository structure

```text
wattara/
  agent.py          Muse Spark tool loop and response handling
  config.py         API configuration and illustrative parameters
  household.py      Consumption baseline, action simulation and validation
  llm.py            Meta API client, usage accounting and local cost limit
  ml.py             LightGBM forecaster and appliance anomaly detector
  simulator.py      Synthetic household data and simulated device adapter
  evaluate.py       Fifteen-goal agent evaluation
run_demo.py         Interactive and offline demos
run_ml_eval.py      ML evaluation
run_eval.py         Live agent evaluation (requires Meta API)
tests/              Offline checks, including a mock-agent flow
results/            Offline evidence and space for agent evaluation logs
notebooks/          Optional Colab quickstart
.env.example        Local API configuration template; no real credentials
```

## Project history and tools

Wattara was developed as a solo project during the hackathon, starting after the October 26, 2026 kickoff. There was no pre-existing Wattara code. I used Claude and NotebookLM for planning, drafting code and reviewing documentation; the prototype was executed and tested locally in VS Code. Google Colab was used during early development but is not required to run the project.

## License

MIT — see [`LICENSE`](LICENSE).
