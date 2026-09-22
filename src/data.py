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

    TODO(stella): pd.read_csv with sep=r"\\s+", header=None, names=COLUMNS.
    """
    raise NotImplementedError


def load_lalonde() -> dict[str, pd.DataFrame]:
    """Read all four files and return them keyed the same way as FILES."""
    raise NotImplementedError


def build_observational(pool: str) -> pd.DataFrame:
    """Stack the NSW treated arm on top of a survey control pool ('cps' or 'psid').

    This is the step that manufactures the confounding: the treated units are real
    program participants, the controls are ordinary survey respondents who are older,
    better educated and far higher earning. Everything downstream exists to undo it.

    TODO(stella): concat, reset index, and add any derived columns you want here
    (for example the zero-earnings indicators u74 and u75, which turn out to matter).
    """
    raise NotImplementedError


def build_experimental() -> pd.DataFrame:
    """Stack the two randomized NSW arms. Use this to compute the benchmark."""
    raise NotImplementedError


def covariate_spec(name: str) -> list[str]:
    """Return the covariate column names for one modelling specification.

    Three specifications worth comparing, because the choice changes the answer:

    ``demographics``
        age, educ, black, hisp, married, nodegree. Deliberately omits pre-treatment
        earnings, which is what makes it fail.
    ``with_earnings``
        adds re74, re75 and the zero-earnings indicators u74, u75.
    ``dw``
        the Dehejia-Wahba specification: adds squared and interaction terms.

    TODO(stella): return the list for each name, raise ValueError otherwise.
    """
    raise NotImplementedError


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
