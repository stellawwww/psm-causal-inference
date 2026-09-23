"""Loading and shaping the LaLonde / Dehejia-Wahba data.

Skeleton only. Fill these in as you work through ``notebooks/01_data_prep.ipynb``;
a working reference implementation of the same steps already exists inline in
``scripts/run_all.py`` if you get stuck.

Background: four files, ten columns each, no header, whitespace separated.

    treat age educ black hisp married nodegree re74 re75 re78

``nswre74_treated`` and ``nswre74_control`` are the two arms of a real randomized
experiment, so comparing them gives the unbiased benchmark. An observational study
is built by discarding the randomized controls and substituting survey respondents
from ``cps_controls`` or ``psid_controls``, who were never part of the experiment.
"""
from __future__ import annotations

import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "lalonde"
OUT_DIR = ROOT / "outputs" / "data"

COLUMNS = ["treat", "age", "educ", "black", "hisp", "married",
           "nodegree", "re74", "re75", "re78"]

FILES = {
    "nsw_treated": "nswre74_treated.txt",
    "nsw_control": "nswre74_control.txt",
    "cps": "cps_controls.txt",
    "psid": "psid_controls.txt",
}

TREATMENT = "treat"
OUTCOME = "re78"


def load_raw(which: str) -> pd.DataFrame:
    """Read one of the four source files by key: nsw_treated, nsw_control, cps, psid.

    The files have no header row and are whitespace separated, so the column names
    have to be supplied. Every value is numeric.
    """
    if which not in FILES:
        raise ValueError(f"unknown file {which!r}; expected one of {list(FILES)}")
    return pd.read_csv(RAW_DIR / FILES[which], sep=r"\s+", header=None, names=COLUMNS)


def load_lalonde() -> dict:
    """Read all four files and return them keyed the same way as FILES."""
    return {k: load_raw(k) for k in FILES}


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add the columns every downstream stage expects.

    ``u74`` / ``u75`` flag zero earnings in the two pre-treatment years. They matter
    more than they look: a large share of program participants earned nothing at all,
    while almost all survey respondents earned something, so the indicator carries
    information the continuous earnings variable alone does not.

    ``employed78`` is a binary version of the outcome, useful for showing that the
    pipeline works on incidence metrics and not only on continuous means.
    """
    df = df.copy()
    df["u74"] = (df["re74"] == 0).astype(int)
    df["u75"] = (df["re75"] == 0).astype(int)
    df["employed78"] = (df["re78"] > 0).astype(int)
    return df


def add_dw_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Add the squared and interaction terms used by the Dehejia-Wahba specification."""
    df = df.copy()
    df["age2"] = df["age"] ** 2
    df["age3"] = df["age"] ** 3
    df["educ2"] = df["educ"] ** 2
    df["re74_2"] = df["re74"] ** 2
    df["re75_2"] = df["re75"] ** 2
    df["educ_re74"] = df["educ"] * df["re74"]
    return df


def build_experimental() -> pd.DataFrame:
    """Stack the two randomized NSW arms.

    Assignment here was by lottery, so a plain difference in ``re78`` between the arms
    is already an unbiased estimate. This frame is what produces the benchmark that
    everything else is scored against.
    """
    raw = load_lalonde()
    df = pd.concat([raw["nsw_treated"], raw["nsw_control"]], ignore_index=True)
    df["pool"] = "nsw_experimental"
    return add_dw_terms(add_derived(df))


def build_observational(pool: str) -> pd.DataFrame:
    """Stack the NSW treated arm on top of a survey control pool ('cps' or 'psid').

    This is the step that manufactures the confounding. The treated units are real
    program participants; the controls are ordinary survey respondents who are older,
    better educated and far higher earning, and who were never part of the experiment.
    Everything downstream exists to undo the resulting bias.

    The two pools are kept separate on purpose and must never be concatenated. They
    are two independent replications of the same test at different difficulty levels:
    CPS offers 15,992 candidates and PSID only 2,490, and PSID respondents sit further
    from the treated group on every covariate. Pooling them would let the matcher draw
    from whichever pool is easier and would hide exactly the overlap problem the
    project is meant to expose.
    """
    if pool not in ("cps", "psid"):
        raise ValueError(f"pool must be 'cps' or 'psid', got {pool!r}")
    raw = load_lalonde()
    df = pd.concat([raw["nsw_treated"], raw[pool]], ignore_index=True)
    df["pool"] = pool
    return add_dw_terms(add_derived(df))


_SPECS = {
    "demographics": ["age", "educ", "black", "hisp", "married", "nodegree"],
    "with_earnings": ["age", "educ", "black", "hisp", "married", "nodegree",
                      "re74", "re75", "u74", "u75"],
    "dw": ["age", "educ", "black", "hisp", "married", "nodegree",
           "re74", "re75", "u74", "u75",
           "age2", "age3", "educ2", "re74_2", "re75_2", "educ_re74"],
}


def covariate_spec(name: str) -> list:
    """Return the covariate column names for one modelling specification.

    Three specifications worth comparing, because the choice changes the answer:

    ``demographics``
        Age, education, race, marital status, degree. Deliberately omits pre-treatment
        earnings. Running the whole pipeline on this specification is what demonstrates
        that an unmeasured confounder flips the sign of the conclusion.
    ``with_earnings``
        Adds 1974 and 1975 earnings plus the zero-earnings indicators.
    ``dw``
        The Dehejia-Wahba specification: adds squared and interaction terms so the
        propensity model can bend rather than only tilt.
    """
    if name not in _SPECS:
        raise ValueError(f"unknown spec {name!r}; expected one of {list(_SPECS)}")
    return list(_SPECS[name])


def save_stage(df: pd.DataFrame, name: str) -> pathlib.Path:
    """Write a notebook's output to outputs/data/<name>.csv and return the path.

    Notebooks hand data to each other through files, never through memory, so each
    one can be re-run on its own. CSV rather than parquet so the intermediate
    products are previewable on GitHub.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def load_stage(name: str) -> pd.DataFrame:
    """Read back what a previous notebook wrote with :func:`save_stage`."""
    return pd.read_csv(OUT_DIR / f"{name}.csv")
