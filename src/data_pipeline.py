from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import CLEAN_ACCIDENT_FILE


def _find_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _read_any_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")
    except Exception:
        # German open data often uses semicolon separators
        return pd.read_csv(path, sep=";")


def load_raw_accidents(path: str | Path) -> pd.DataFrame:
    return _read_any_csv(path)


def standardize_unfallatlas_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise either raw Unfallatlas columns or already-cleaned EDA columns.

    Output columns:
    - year
    - month
    - hour
    - day_of_week
    - accident_severity
    - latitude
    - longitude
    - district_code
    - is_bicycle
    - severity_label
    """

    out = df.copy()

    col_map = {
        "year": ["year", "UJAHR"],
        "month": ["month", "UMONAT"],
        "hour": ["hour", "USTUNDE"],
        "day_of_week": ["day_of_week", "weekday", "UWOCHENTAG"],
        "accident_severity": ["accident_severity", "UKATEGORIE"],
        "latitude": ["latitude", "lat", "YGCSWGS84", "Y"],
        "longitude": ["longitude", "lon", "lng", "XGCSWGS84", "X"],
        "district_code": ["district_code", "ULAND", "UREGBEZ", "UKREIS"],
        "is_bicycle": ["is_bicycle", "IstRad", "istrad"],
    }

    rename = {}
    for target, candidates in col_map.items():
        found = _find_first_existing_column(out, candidates)
        if found is not None:
            rename[found] = target

    out = out.rename(columns=rename)

    required = ["year", "month", "hour", "day_of_week", "accident_severity", "latitude", "longitude"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(
            f"Missing required columns after standardisation: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Convert decimal comma coordinates if necessary.
    for col in ["latitude", "longitude"]:
        out[col] = (
            out[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .replace({"nan": np.nan, "None": np.nan})
            .astype(float)
        )

    for col in ["year", "month", "hour", "day_of_week", "accident_severity"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    if "is_bicycle" not in out.columns:
        out["is_bicycle"] = 1
    else:
        out["is_bicycle"] = pd.to_numeric(out["is_bicycle"], errors="coerce").fillna(0).astype(int)

    if "district_code" not in out.columns:
        out["district_code"] = -1

    # Filter Berlin if ULAND is available and contains German federal state code.
    # Berlin = 11.
    if "ULAND" in df.columns:
        out = out[out["district_code"].astype(str).str.startswith("11") | (out["district_code"] == 11)]

    # Filter bicycle accidents if raw Unfallatlas flag exists.
    out = out[out["is_bicycle"] == 1].copy()

    # Drop unusable coordinates.
    out = out.dropna(subset=["latitude", "longitude", "year", "month", "hour", "day_of_week", "accident_severity"])

    # Severity: Unfallatlas UKATEGORIE: 1=fatal, 2=serious injury, 3=minor injury.
    severity_map = {
        1: "fatal",
        2: "serious_injury",
        3: "minor_injury",
    }
    out["severity_label"] = out["accident_severity"].astype(int).map(severity_map).fillna("unknown")

    out["serious_or_fatal"] = out["accident_severity"].isin([1, 2]).astype(int)

    # KSI-like weights, consistent with Person B's stronger risk design.
    severity_weight_map = {
        "minor_injury": 1.0,
        "serious_injury": 8.0,
        "fatal": 30.0,
        "unknown": 1.0,
    }
    out["severity_weight"] = out["severity_label"].map(severity_weight_map).fillna(1.0)

    out["is_weekend"] = out["day_of_week"].isin([1, 7]).astype(int)
    out["is_rush_hour"] = (
        out["hour"].between(7, 9) | out["hour"].between(16, 18)
    ).astype(int)
    out["is_night"] = (
        out["hour"].between(20, 23) | out["hour"].between(0, 5)
    ).astype(int)
    out["season"] = out["month"].apply(get_season)

    keep = [
        "year", "month", "hour", "day_of_week", "accident_severity",
        "severity_label", "severity_weight", "serious_or_fatal",
        "latitude", "longitude", "district_code",
        "is_weekend", "is_rush_hour", "is_night", "season",
    ]
    keep = [c for c in keep if c in out.columns]
    return out[keep].reset_index(drop=True)


def get_season(month: int) -> str:
    try:
        month = int(month)
    except Exception:
        return "unknown"

    if month in [12, 1, 2]:
        return "winter"
    if month in [3, 4, 5]:
        return "spring"
    if month in [6, 7, 8]:
        return "summer"
    return "autumn"


def prepare_clean_accidents(
    accident_file: str | Path | None = None,
    use_clean_data: bool = False,
    output_file: str | Path = CLEAN_ACCIDENT_FILE,
) -> pd.DataFrame:
    """Load and clean Unfallatlas accident data.

    If use_clean_data=True, reads data/processed/berlin_bike_accidents_clean.csv.
    Otherwise, reads accident_file and standardises it.
    """
    output_file = Path(output_file)

    if use_clean_data:
        if not output_file.exists():
            raise FileNotFoundError(
                f"Clean data file not found: {output_file}. "
                "Save df_berlin_rad_final there or pass --accident-file."
            )
        df = pd.read_csv(output_file)
        clean = standardize_unfallatlas_columns(df)
    else:
        if accident_file is None:
            if output_file.exists():
                df = pd.read_csv(output_file)
                clean = standardize_unfallatlas_columns(df)
            else:
                raise FileNotFoundError(
                    "No accident file provided and clean data file does not exist."
                )
        else:
            raw = load_raw_accidents(accident_file)
            clean = standardize_unfallatlas_columns(raw)

    clean.to_csv(output_file, index=False)
    print(f"Saved clean accidents: {output_file} ({clean.shape[0]:,} rows × {clean.shape[1]} columns)")
    return clean
