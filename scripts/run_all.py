"""Validate the PSM pipeline on LaLonde / Dehejia-Wahba data.

Ground truth = NSW experimental estimate (treated vs randomized control).
Observational versions replace the NSW control group with CPS or PSID survey samples.
Question: does the pipeline recover the experimental benchmark?
"""
import sys, pathlib
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import psm, figures

RAW = ROOT / "data/raw/lalonde"
OUT = ROOT / "outputs" / "tables"; OUT.mkdir(parents=True, exist_ok=True)
COLS = ["treat", "age", "educ", "black", "hisp", "married", "nodegree", "re74", "re75", "re78"]
load = lambda n: pd.read_csv(RAW / f"{n}.txt", sep=r"\s+", header=None, names=COLS)
nsw_t, nsw_c = load("nswre74_treated"), load("nswre74_control")
pools = {"CPS": load("cps_controls"), "PSID": load("psid_controls")}


def features(df: pd.DataFrame, spec: str) -> pd.DataFrame:
    X = df[["age", "educ", "black", "hisp", "married", "nodegree"]].copy()
    if spec == "no_earnings":          # deliberately omit pre-treatment earnings
        return X
    X["re74"], X["re75"] = df.re74, df.re75
    X["u74"], X["u75"] = (df.re74 == 0).astype(int), (df.re75 == 0).astype(int)
    if spec == "dw":                   # Dehejia-Wahba style nonlinear terms
        X["age2"], X["age3"] = df.age ** 2, df.age ** 3
        X["educ2"] = df.educ ** 2
        X["re74_2"], X["re75_2"] = df.re74 ** 2, df.re75 ** 2
        X["educ_re74"] = df.educ * df.re74
    return X


BALANCE_COLS = ["age", "educ", "black", "hisp", "married", "nodegree", "re74", "re75", "u74", "u75"]

# ------------------------------------------------------------ benchmark
truth = nsw_t.re78.mean() - nsw_c.re78.mean()
from scipy import stats
tt = stats.ttest_ind(nsw_t.re78, nsw_c.re78, equal_var=False)
lo, hi = tt.confidence_interval(0.95)
rows = [dict(pool="NSW (RCT)", spec="-", method="experimental benchmark", est=truth, ci_lo=lo, ci_hi=hi, n=len(nsw_t) + len(nsw_c))]
print(f"Experimental benchmark: {truth:,.0f}  95% CI [{lo:,.0f}, {hi:,.0f}]")

# ------------------------------------------------------------ observational runs
for pool_name, pool in pools.items():
    df = pd.concat([nsw_t, pool], ignore_index=True)
    t, y = df.treat.values.astype(int), df.re78.values
    naive = y[t == 1].mean() - y[t == 0].mean()
    rows.append(dict(pool=pool_name, spec="-", method="naive difference", est=naive, ci_lo=np.nan, ci_hi=np.nan, n=len(df)))
    print(f"\n=== {pool_name} controls (n={len(pool):,}) | naive diff = {naive:,.0f}")

    for spec in ["no_earnings", "basic", "dw"]:
        X = features(df, spec)
        Xb = features(df, "basic")[BALANCE_COLS]
        ps = psm.fit_propensity(X, t, model="logit", C=1.0)
        auc_note = f"ps mean t={ps[t==1].mean():.3f} c={ps[t==0].mean():.3f}; controls with ps>0.5: {(ps[t==0]>0.5).sum()}"
        print(f"\n  spec={spec:12s} {auc_note}")

        # matching variants
        variants = {
            "match caliper=0.01 raw, no replace": dict(caliper=0.01, scale="raw", replace=False),
            "match caliper=0.2 logit-SD, no replace": dict(caliper=0.2, scale="logit", replace=False),
            "match caliper=0.2 logit-SD, with replace": dict(caliper=0.2, scale="logit", replace=True),
        }
        for label, kw in variants.items():
            pairs = psm.nn_match(ps, t, seed=0, **kw)
            if len(pairs) < 10:
                print(f"    {label:42s} -> only {len(pairs)} pairs"); continue
            r = psm.att_matched(y, pairs)
            bal = psm.balance_table(Xb, t, pairs)
            mx = bal["matched"].abs().max()
            rows.append(dict(pool=pool_name, spec=spec, method=label, est=r["est"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], n=r["n_pairs"], max_abs_smd=mx))
            print(f"    {label:42s} ATT={r['est']:8,.0f}  CI[{r['ci_lo']:7,.0f},{r['ci_hi']:7,.0f}]  pairs={r['n_pairs']:3d}  max|SMD|={mx:.2f}")
            if spec == "dw" and "with replace" in label:
                figures.love_plot(bal, f"love_{pool_name.lower()}", f"{pool_name}: balance, DW spec")
                figures.overlap_plot(ps, t, f"overlap_{pool_name.lower()}", f"{pool_name}: propensity overlap")

        # weighting / doubly robust
        for est_name, kw in {"IPW ATT (trim .01-.99)": dict(estimand="ATT", trim=(.01, .99)),
                             "IPW ATE stabilized (trim .05-.95)": dict(estimand="ATE", trim=(.05, .95))}.items():
            r = psm.ipw_estimate(y, t, ps, **kw)
            rows.append(dict(pool=pool_name, spec=spec, method=est_name, est=r["est"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], n=r["n_used"]))
            print(f"    {est_name:42s} est={r['est']:8,.0f}  CI[{r['ci_lo']:7,.0f},{r['ci_hi']:7,.0f}]  n={r['n_used']}")
        r = psm.aipw_att(y, t, ps, X, trim=(.01, .99), n_boot=300)
        rows.append(dict(pool=pool_name, spec=spec, method="AIPW / doubly robust ATT", est=r["est"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], n=r["n_used"]))
        print(f"    {'AIPW / doubly robust ATT':42s} est={r['est']:8,.0f}  CI[{r['ci_lo']:7,.0f},{r['ci_hi']:7,.0f}]  n={r['n_used']}")

res = pd.DataFrame(rows)
res["covers_truth"] = (res.ci_lo <= truth) & (res.ci_hi >= truth)
res.to_csv(OUT / "results.csv", index=False)
print("\nSaved", OUT / "results.csv")
