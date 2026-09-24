"""Reusable propensity-score toolkit: PS model, caliper matching, balance, estimators.

Clean-room implementation. Every function takes plain numpy/pandas objects plus
explicit treatment / outcome / covariate names, so the same code runs on any dataset.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from bisect import bisect_left
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
import statsmodels.api as sm


# ---------------------------------------------------------------- propensity
# sklearn's LogisticRegression applies an L2 penalty by default at C=1.0, where C is the
# INVERSE penalty strength. A very large C makes the penalty negligible, which is what we
# want here. Expressed as a large C rather than penalty=None because the keyword spelling
# is not portable across the sklearn versions this project supports ('none' in 1.0-1.3,
# None from 1.2). Verified equivalent: C=1e6 and penalty='none' agree to 2e-7 on every
# coefficient and every fitted score.
NO_PENALTY = 1e6


def fit_propensity(X: pd.DataFrame, t: np.ndarray, model: str = "logit",
                   C: float = NO_PENALTY, out_of_fold: bool = False, seed: int = 0) -> np.ndarray:
    """Return P(T=1|X). model in {'logit','gbm'}. out_of_fold=True gives
    cross-fitted scores (5-fold), which avoids in-sample overfit for flexible models.

    ``C`` defaults to :data:`NO_PENALTY`, i.e. an unpenalized logistic regression.

    This is deliberate, and it is where a propensity model parts company with a prediction
    model. Regularization exists to stop coefficients chasing noise so a model generalizes
    to unseen data. Nothing here is ever applied to unseen data: the score's only job is to
    produce a number such that, conditioning on it, the covariates come out balanced.
    Shrinking the coefficients toward zero makes the score weigh each covariate less, which
    is precisely what makes it less able to equalize them. On this data, balance after
    matching degrades monotonically as the penalty is strengthened.

    The criterion for choosing ``C`` is therefore balance, never accuracy, and never AUC.
    """
    if model == "logit":
        clf = LogisticRegression(C=C, max_iter=2000)
        Xm = StandardScaler().fit_transform(X)
    elif model == "gbm":
        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                             max_iter=300, random_state=seed)
        Xm = X.values
    else:
        raise ValueError(model)
    if out_of_fold:
        return cross_val_predict(clf, Xm, t, cv=5, method="predict_proba")[:, 1]
    return clf.fit(Xm, t).predict_proba(Xm)[:, 1]


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------ matching
def nn_match(ps: np.ndarray, t: np.ndarray, caliper: float, scale: str = "logit",
             replace: bool = False, seed: int = 0) -> pd.DataFrame:
    """Greedy 1:1 nearest-neighbour matching on the propensity score.

    scale='logit': distance measured on logit(ps), caliper in logit-SD units
                   (caliper * pooled SD of logit ps), the Rosenbaum-Rubin convention.
    scale='raw'  : distance and caliper on the raw probability scale.
    Treated units are visited in random order; without replacement each control
    is used at most once. Returns one row per matched pair with an explicit pair_id.
    """
    rng = np.random.default_rng(seed)
    score = logit(ps) if scale == "logit" else ps.copy()
    cal = caliper * score.std() if scale == "logit" else caliper
    t_idx = np.flatnonzero(t == 1)
    c_idx = np.flatnonzero(t == 0)
    order = np.argsort(score[c_idx])
    c_sorted_idx = c_idx[order]
    c_sorted = score[c_sorted_idx]
    used = np.zeros(len(c_sorted), dtype=bool)
    rows = []
    for k, i in enumerate(rng.permutation(t_idx)):
        s = score[i]
        j = bisect_left(c_sorted, s)
        # walk outward from insertion point to nearest available control
        lo, hi, best, bestd = j - 1, j, None, np.inf
        while lo >= 0 or hi < len(c_sorted):
            dlo = s - c_sorted[lo] if lo >= 0 else np.inf
            dhi = c_sorted[hi] - s if hi < len(c_sorted) else np.inf
            if dlo <= dhi:
                cand, d, lo = lo, dlo, lo - 1
            else:
                cand, d, hi = hi, dhi, hi + 1
            if d > cal or d >= bestd:
                break
            if replace or not used[cand]:
                best, bestd = cand, d
                break
        if best is None:
            continue
        if not replace:
            used[best] = True
        rows.append(dict(pair_id=k, treated=i, control=c_sorted_idx[best],
                         ps_t=ps[i], ps_c=ps[c_sorted_idx[best]], dist=bestd))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- balance
def reference_sd(X: pd.DataFrame, t: np.ndarray, kind: str = "treated") -> pd.Series:
    """The denominator for standardized mean differences, computed once and reused.

    ``treated`` uses the treated group's standard deviation alone. This is the
    convention for ATT, and it is what this project uses, for two reasons.

    First, it keeps comparisons honest across samples. The control group's own
    dispersion is a property of which pool you happened to draw from, not of the
    imbalance being measured, so letting it into the denominator lets a wide control
    pool disguise a large gap. On this data the CPS and PSID pools look equally
    imbalanced on 1975 earnings under a pooled denominator (1.75 against 1.77) and
    very differently imbalanced under a fixed one (3.76 against 5.45), which is the
    honest reading: the PSID gap really is 45% larger.

    Second, it keeps before and after comparable. A denominator recomputed after
    matching moves under your feet, so balance can appear to improve purely because
    the matched controls are less dispersed than the original pool.

    ``pooled`` is the ATE convention and is offered for completeness.
    """
    if kind == "treated":
        return X[t == 1].std(ddof=1)
    if kind == "pooled":
        s1, s0 = X[t == 1].std(ddof=1), X[t == 0].std(ddof=1)
        return np.sqrt((s1 ** 2 + s0 ** 2) / 2)
    raise ValueError(f"kind must be 'treated' or 'pooled', got {kind!r}")


def smd(X: pd.DataFrame, t: np.ndarray, w: np.ndarray | None = None,
        ref_sd: pd.Series | None = None) -> pd.Series:
    """Standardized mean difference per covariate.

    Pass ``ref_sd`` to hold the denominator fixed across a set of comparisons; see
    :func:`reference_sd`. When omitted it defaults to the treated group's SD, matching
    the ATT estimand this project targets.
    """
    w = np.ones(len(t)) if w is None else w
    m1 = np.average(X[t == 1], axis=0, weights=w[t == 1])
    m0 = np.average(X[t == 0], axis=0, weights=w[t == 0])
    if ref_sd is None:
        ref_sd = reference_sd(X, t)
    return pd.Series((m1 - m0) / ref_sd.values, index=X.columns)


def balance_table(X: pd.DataFrame, t: np.ndarray, pairs: pd.DataFrame,
                  w: np.ndarray | None = None, sd_kind: str = "treated") -> pd.DataFrame:
    """SMD before matching, after matching, and optionally under IPW weights.

    The denominator is computed once from the pre-match sample and reused for every
    column, so the three are directly comparable.
    """
    ref = reference_sd(X, t, sd_kind)
    out = pd.DataFrame({"before": smd(X, t, ref_sd=ref)})
    if len(pairs):
        idx = np.r_[pairs.treated.values, pairs.control.values]
        out["matched"] = smd(X.iloc[idx], t[idx], ref_sd=ref)
    if w is not None:
        out["ipw"] = smd(X, t, w=w, ref_sd=ref)
    return out


# Plotting lives in src/figures.py so this module stays pure computation.
# Use figures.love_plot(balance_table(...), name) and figures.overlap_plot(ps, t, name).


# ---------------------------------------------------------------- estimators
def att_matched(y: np.ndarray, pairs: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """ATT from matched pairs: paired t-test CI plus a pair-level bootstrap CI."""
    d = y[pairs.treated.values] - y[pairs.control.values]
    est = d.mean()
    tt = stats.ttest_1samp(d, 0)
    lo, hi = tt.confidence_interval(0.95)
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n_boot)])
    return dict(est=est, se=d.std(ddof=1) / np.sqrt(len(d)), ci_lo=lo, ci_hi=hi,
                boot_lo=np.percentile(boots, 2.5), boot_hi=np.percentile(boots, 97.5),
                p=tt.pvalue, n_pairs=len(d))


def ipw_weights(ps: np.ndarray, t: np.ndarray, estimand: str = "ATT",
                trim: tuple = (0.0, 1.0), stabilized: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return (weights, keep_mask). ATT: 1 for treated, ps/(1-ps) for control.
    ATE: 1/ps and 1/(1-ps), optionally stabilized by marginal P(T)."""
    keep = (ps > trim[0]) & (ps < trim[1])
    pt = t.mean()
    if estimand == "ATT":
        w = np.where(t == 1, 1.0, ps / (1 - ps))
    elif estimand == "ATE":
        w = np.where(t == 1, 1 / ps, 1 / (1 - ps))
        if stabilized:
            w = w * np.where(t == 1, pt, 1 - pt)
    else:
        raise ValueError(estimand)
    return w, keep


def ipw_estimate(y: np.ndarray, t: np.ndarray, ps: np.ndarray, estimand: str = "ATT",
                 trim: tuple = (0.0, 1.0)) -> dict:
    """Weighted regression of y on t with HC1 robust SE."""
    w, keep = ipw_weights(ps, t, estimand, trim)
    X = sm.add_constant(t[keep].astype(float))
    fit = sm.WLS(y[keep], X, weights=w[keep]).fit(cov_type="HC1")
    ci = np.asarray(fit.conf_int())
    return dict(est=float(fit.params[1]), se=float(fit.bse[1]), ci_lo=ci[1, 0], ci_hi=ci[1, 1],
                p=float(fit.pvalues[1]), n_used=int(keep.sum()))


def aipw_att(y: np.ndarray, t: np.ndarray, ps: np.ndarray, X: pd.DataFrame,
             trim: tuple = (0.0, 1.0), n_boot: int = 500, seed: int = 0) -> dict:
    """Doubly-robust ATT: outcome model for controls (OLS) + IPW correction."""
    keep = (ps > trim[0]) & (ps < trim[1])
    y, t, ps, X = y[keep], t[keep], ps[keep], X[keep]
    Xc = sm.add_constant(X.values.astype(float))

    def est_fn(idx):
        yy, tt, pp, xx = y[idx], t[idx], ps[idx], Xc[idx]
        mu0 = sm.OLS(yy[tt == 0], xx[tt == 0]).fit().predict(xx)
        n1 = (tt == 1).sum()
        term_t = ((yy - mu0)[tt == 1]).sum()
        w0 = pp / (1 - pp)
        term_c = (w0[tt == 0] * (yy - mu0)[tt == 0]).sum()
        return (term_t - term_c) / n1

    idx_all = np.arange(len(y))
    est = est_fn(idx_all)
    rng = np.random.default_rng(seed)
    boots = np.array([est_fn(rng.choice(idx_all, len(idx_all), replace=True)) for _ in range(n_boot)])
    return dict(est=est, se=boots.std(ddof=1), ci_lo=np.percentile(boots, 2.5),
                ci_hi=np.percentile(boots, 97.5), n_used=int(keep.sum()))
