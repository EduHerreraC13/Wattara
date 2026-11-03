"""Machine-learning layer (no API calls, no tokens spent).

1. Bill forecaster  - LightGBM on daily consumption + weather + weekday features.
2. Anomaly detector - per-appliance residual z-scores (explainable: it says WHICH appliance and signal).

IMPORTANT: the data are simulated by `generate_rich` with the same rules the detector learns,
so these metrics prove the PIPELINE works, not accuracy on real homes (see README, Limitations).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression

from . import config

APPS = ["fridge", "water_heater", "tv", "console", "router"]

def generate_rich(days=420, start="2025-11-01", seed=42, faults=None):
    """More realistic simulated home: weather, weekends, variable habits and optional faults.
    faults = {"console_standby": (d0, d1), "fridge_degraded": (d0, d1), "heater_leak": (d0, d1)}"""
    rng = np.random.default_rng(seed)
    faults = faults or {}
    dates = pd.date_range(start, periods=days, freq="D")
    doy = dates.dayofyear.values
    temp = 21 + 5.5 * np.sin(2 * np.pi * (doy + 46) / 365)
    noise = np.zeros(days)
    for i in range(1, days):
        noise[i] = 0.7 * noise[i - 1] + rng.normal(0, 1.0)
    temp = temp + noise

    idx = pd.date_range(start, periods=days * 24, freq="h")
    kwh = {a: np.zeros(days * 24) for a in APPS}
    fault_day = np.zeros(days, dtype=bool)
    for d in range(days):
        weekend = dates[d].dayofweek >= 5
        on = lambda name: name in faults and faults[name][0] <= d < faults[name][1]
        if faults and any(on(n) for n in faults):
            fault_day[d] = True
        f_morning = float(np.clip(rng.normal(0.55 + 0.02 * (21 - temp[d]), 0.1), 0.2, 0.95))
        evening_shower = rng.random() < 0.3
        for h in range(24):
            i = d * 24 + h
            # fridge: works harder when it is hot
            duty = np.clip(0.30 + 0.012 * (temp[d] - 21), 0.2, 0.5)
            if on("fridge_degraded"):
                duty *= 1.5
            kwh[APPS[0]][i] = 120 * duty * rng.normal(1, 0.08) / 1000
            # electric water heater (morning shower, sometimes an evening one)
            p = 0.0
            m_hour = 7 if weekend else 6
            if h == m_hour:
                p = 3500 * f_morning
            if h == 21 and evening_shower:
                p = 3500 * 0.35
            if on("heater_leak") and h == 3:
                p = 3500 * 0.15
            kwh[APPS[1]][i] = p * rng.normal(1, 0.08) / 1000
            # tv
            tv_hours = range(14, 23) if weekend else range(19, 22)
            tv_on = h in tv_hours and rng.random() < (0.6 if weekend else 0.9)
            kwh[APPS[2]][i] = (100 if tv_on else 3) * rng.normal(1, 0.08) / 1000
            # game console
            con_hours = range(15, 22) if weekend else range(20, 22)
            con_on = h in con_hours and rng.random() < (0.7 if weekend else 0.5)
            standby_w = 60 if on("console_standby") else 10
            kwh[APPS[3]][i] = (120 if con_on else standby_w) * rng.normal(1, 0.08) / 1000
            # router
            kwh[APPS[4]][i] = 10 * rng.normal(1, 0.08) / 1000
    hourly = pd.DataFrame(kwh, index=idx)
    daily = hourly.groupby(hourly.index.date).sum()
    daily.index = pd.to_datetime(daily.index)
    nh = hourly.between_time("00:00", "05:59")
    night = nh.groupby(nh.index.date).max() * 1000  # watts, maximum reading between 0 and 6 h
    night.index = pd.to_datetime(night.index)
    night.columns = [c + "_night_w" for c in night.columns]
    out = pd.concat([daily, night], axis=1)
    out["total"] = daily.sum(axis=1)
    out["temp"] = temp
    out["dow"] = out.index.dayofweek
    out["fault"] = fault_day
    return out

# ================== 1) PRONÓSTICO ==================
def feats(hist, dow, temp):
    h = np.asarray(hist, dtype=float)
    return [dow, temp, h[-1], h[-7], h[-7:].mean(), h[-28:].mean()]

FEATURE_NAMES = ["dow", "temp", "lag1", "lag7", "mean7", "mean28"]

def build_xy(daily):
    X, y, ix = [], [], []
    tot = daily["total"].values
    for t in range(28, len(daily)):
        X.append(feats(tot[:t], daily["dow"].iloc[t], daily["temp"].iloc[t]))
        y.append(tot[t]); ix.append(daily.index[t])
    return pd.DataFrame(X, columns=FEATURE_NAMES, index=ix), pd.Series(y, index=ix)

def train_forecaster(daily_train):
    X, y = build_xy(daily_train)
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=8,
                          min_child_samples=10, verbose=-1, random_state=0)
    m.fit(X, y)
    return m

def project_month(model, daily, month, day_of_month, tariff):
    """Project the month's bill from the first `day_of_month` observed days (recursive day-by-day forecast)."""
    cur = daily[daily.index.to_period("M") == month]
    observed = float(cur["total"].iloc[:day_of_month].sum())
    hist = list(daily["total"].loc[:cur.index[day_of_month - 1]].values)
    extra = 0.0
    for k in range(day_of_month, len(cur)):
        row = cur.iloc[k]
        x = pd.DataFrame([feats(hist, row["dow"], row["temp"])], columns=FEATURE_NAMES)
        yhat = float(model.predict(x)[0])
        hist.append(yhat); extra += yhat
    kwh = observed + extra
    return kwh, kwh * tariff

def naive_month(daily, month, day_of_month, tariff):
    cur = daily[daily.index.to_period("M") == month]
    kwh = float(cur["total"].iloc[:day_of_month].mean() * len(cur))
    return kwh, kwh * tariff

# ================== 2) ANOMALÍAS (residuos explicables por aparato) ==================
def _ctx(d):
    return np.c_[d["temp"].values, (d["dow"] >= 5).astype(int).values]

def train_detector(daily_healthy, k=5.0):
    """For every signal (daily kWh and night-time watts per appliance) learn its normal value
    given temperature and weekend, and how much it normally deviates (MAD)."""
    models = {}
    for a in APPS:
        for col in (a, a + "_night_w"):
            lr = LinearRegression().fit(_ctx(daily_healthy), daily_healthy[col].values)
            res = daily_healthy[col].values - lr.predict(_ctx(daily_healthy))
            mad = np.median(np.abs(res - np.median(res))) * 1.4826 + 1e-9
            models[col] = (lr, np.median(res), mad)
    return {"models": models, "k": k}

def detect(det, d):
    """Return (flag per day, DataFrame of z-scores per signal) so every alert has an explainable cause."""
    z = {}
    for col, (lr, med, mad) in det["models"].items():
        z[col] = (d[col].values - lr.predict(_ctx(d)) - med) / mad
    Z = pd.DataFrame(z, index=d.index)
    return (Z.abs() > det["k"]).any(axis=1).values, Z

def prf(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    tp = int((y & p).sum()); fp = int((~y & p).sum()); fn = int((y & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return round(prec, 3), round(rec, 3), tp, fp, fn

def run_ml_eval():
    TARIFF = config.TARIFF_PER_KWH
    mae = lambda a, b: float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))

    # ---- Forecast ----
    daily = generate_rich(days=480, seed=42)
    split = 330
    model = train_forecaster(daily.iloc[:split])
    test = daily.iloc[split - 28:]
    Xt, yt = build_xy(test)
    pred = model.predict(Xt)
    print("== DAILY FORECAST (1 day ahead) ==")
    print(f"MAE LightGBM: {mae(yt, pred):.3f} kWh | baseline same-weekday-last-week: {mae(yt, Xt['lag7']):.3f} | "
          f"baseline 7-day mean: {mae(yt, Xt['mean7']):.3f} | average daily use: {daily['total'].mean():.2f} kWh")

    print("== BILL PROJECTION FROM DAY 10 ==")
    errs_m, errs_n = [], []
    for mth in sorted(set(daily.iloc[split:].index.to_period("M"))):
        cur = daily[daily.index.to_period("M") == mth]
        if cur.index[0] < daily.index[split] or len(cur) < 28 or cur.index[-1] != cur.index[0] + pd.offsets.MonthEnd(0):
            continue
        real = float(cur["total"].sum()) * TARIFF
        pm = project_month(model, daily, mth, 10, TARIFF)[1]
        pn = naive_month(daily, mth, 10, TARIFF)[1]
        errs_m.append(abs(pm - real) / real * 100); errs_n.append(abs(pn - real) / real * 100)
        print(f"{mth}: actual S/ {real:.1f} | model S/ {pm:.1f} | naive S/ {pn:.1f}")
    print(f"Mean error (%): model {np.mean(errs_m):.2f} | naive {np.mean(errs_n):.2f}")

    # ---- Anomalies ----
    print("== ANOMALY DETECTION (injected faults) ==")
    healthy = generate_rich(days=330, seed=1)
    det = train_detector(healthy)
    F = {"console_standby": (20, 45), "fridge_degraded": (60, 85), "heater_leak": (100, 125)}
    faulty = generate_rich(days=150, seed=2, faults=F)
    flags, Z = detect(det, faulty)
    p, r, tp, fp, fn = prf(faulty["fault"], flags)
    print(f"Per-appliance residuals -> precision {p} | recall {r} (TP {tp}, FP {fp}, FN {fn})")
    mu, sd = healthy["total"].mean(), healthy["total"].std()
    base_flags = (faulty["total"] > mu + 3 * sd).values
    p2, r2, tp2, fp2, fn2 = prf(faulty["fault"], base_flags)
    print(f"Baseline: threshold on total daily kWh -> precision {p2} | recall {r2} (TP {tp2}, FP {fp2}, FN {fn2})")
    for name, (a, b) in F.items():
        print(f"  {name}: detected on {flags[a:b].mean()*100:.0f} % of its days")
    # false alarms on a new healthy home
    h2 = generate_rich(days=150, seed=3)
    f2, _ = detect(det, h2)
    print(f"New healthy home: {f2.sum()} false alarms in {len(h2)} days")
    # most likely cause on a day with a fault
    day = faulty.index[30]
    print("Cause (day 30):", Z.loc[day].abs().sort_values(ascending=False).head(2).round(1).to_dict())


# ================== 3) SERVICE USED BY THE AGENT ==================
class AnomalyService:
    """Detector trained on a healthy simulated home, applied to a demo home with one injected fault.

    The demo fault: the console draws ~6x its normal standby power at night.
    The report only contains ratios ("times normal"), never watts or soles, so it cannot
    contradict the household model used by the planner.
    """

    def __init__(self, appliances, fault=None):
        self.appliances = appliances
        self.det = train_detector(generate_rich(days=330, seed=1))
        self.scenario = generate_rich(days=91, start="2026-09-01", seed=7,
                                      faults=fault or {"console_standby": (60, 91)})

    def report(self, last_days=14):
        recent = self.scenario.iloc[-int(last_days):]
        flags, Z = detect(self.det, recent)
        findings = []
        for col, (lr, med, mad) in self.det["models"].items():
            if (Z[col].abs() > self.det["k"]).mean() >= 0.5:      # abnormal on at least half the days
                expected = float((lr.predict(_ctx(recent)) + med).mean())
                ratio = float(recent[col].mean()) / max(expected, 1e-9)
                if ratio < 1.5:                                    # only EXCESS consumption matters
                    continue
                app = col.replace("_night_w", "")
                findings.append({
                    "appliance": app,
                    "signal": "night_standby_draw" if col.endswith("_night_w") else "daily_energy",
                    "times_normal": round(ratio, 1),
                    "protected": self.appliances[app]["protected"]})
        return {"window_days": int(last_days), "days_flagged": int(flags.sum()), "findings": findings}
