from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import CLEAN_ACCIDENT_FILE, RAW_DIR


COLUMN_MAP = {
    # Harmonise column drift across Unfallatlas releases.
    "IstStrasse": "STRZUSTAND",
    "IstStrassenzustand": "STRZUSTAND",
    "LICHT": "ULICHTVERH",
    "IstSonstig": "IstSonstige",
    "UIDENTSTLAE": "UIDENTSTLA",
    "XGCSWGS84": "longitude",
    "YGCSWGS84": "latitude",
    "UJAHR": "year",
    "UMONAT": "month",
    "USTUNDE": "hour",
    "UWOCHENTAG": "day_of_week",
    "UKATEGORIE": "accident_severity",
    "ULAND": "state_code",
}


LIGHT_LABELS = {
    0: "daylight",
    1: "twilight",
    2: "dark_lit",
    3: "dark_unlit_old_code_check",
}

SURFACE_LABELS = {
    0: "dry",
    1: "wet_or_slippery",
    2: "wintery",
}


def get_season(month: int) -> str:
    month = int(month)
    if month in [12, 1, 2]:
        return "winter"
    if month in [3, 4, 5]:
        return "spring"
    if month in [6, 7, 8]:
        return "summer"
    return "autumn"


def _read_accident_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    # Unfallatlas files are usually semicolon-separated. Some newer files are CSV
    # but still use semicolons; using sep=";" is safe for the official files.
    return pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False)


def read_raw_unfallatlas(raw_dir: str | Path = RAW_DIR) -> pd.DataFrame:
    """Read all Unfallatlas files in data/raw or data.

    Skips LinRef files.
    """
    raw_dir = Path(raw_dir)
    patterns = [
        str(raw_dir / "Unfallorte*.txt"),
        str(raw_dir / "Unfallorte*.csv"),
        str(raw_dir.parent / "Unfallorte*.txt"),
        str(raw_dir.parent / "Unfallorte*.csv"),
    ]

    paths: list[str] = []
    for pat in patterns:
        paths.extend(glob.glob(pat))

    paths = sorted(set(p for p in paths if "LinRef" not in os.path.basename(p)))

    if not paths:
        raise FileNotFoundError(
            "No raw Unfallatlas files found. Put Unfallorte*.csv/txt in data/raw/ "
            "or place an already-cleaned berlin_bike_2018_2025.csv in data/processed/."
        )

    frames = []
    for p in paths:
        d = _read_accident_file(p).rename(columns=COLUMN_MAP)
        d["source_file"] = os.path.basename(p)
        frames.append(d)
        print(f"loaded {os.path.basename(p):30s} {len(d):>8,}")

    return pd.concat(frames, ignore_index=True)


def _ensure_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def prepare_berlin_bicycle_accidents(
    raw_file: str | Path | None = None,
    output_file: str | Path = CLEAN_ACCIDENT_FILE,
    use_existing_clean: bool = False,
) -> pd.DataFrame:
    """Prepare Berlin bicycle Unfallatlas data for 2018-2025.

    The output keeps raw fields needed for the severity model and robustness checks:
    ULICHTVERH, STRZUSTAND, IstGkfz, IstSonstige, IstPKW, UTYP1, UKATEGORIE and ULAND.

    This function intentionally does not create synthetic negatives. That belongs
    in model_training.py.
    """
    output_file = Path(output_file)

    if use_existing_clean and output_file.exists():
        df = pd.read_csv(output_file)
        # Canary for the corrected light mapping.
        if "light_label" in df.columns:
            assert "dark_unlit" not in set(df["light_label"].astype(str)), (
                "Stale light labels detected. Regenerate berlin_bike_2018_2025.csv."
            )
        return df

    if raw_file is not None:
        df = _read_accident_file(raw_file).rename(columns=COLUMN_MAP)
    else:
        df = read_raw_unfallatlas()

    df = df.rename(columns=COLUMN_MAP)
    df = _ensure_numeric(
        df,
        [
            "year",
            "month",
            "hour",
            "day_of_week",
            "accident_severity",
            "state_code",
            "latitude",
            "longitude",
            "IstRad",
            "IstPKW",
            "IstGkfz",
            "IstSonstige",
            "ULICHTVERH",
            "STRZUSTAND",
            "UTYP1",
        ],
    )

    required = ["year", "month", "hour", "day_of_week", "accident_severity", "state_code", "IstRad", "latitude", "longitude"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required Unfallatlas columns after harmonisation: {missing}")

    # Berlin = state code 11. Bicycle accident = IstRad == 1.
    out = df[
        (df["state_code"] == 11)
        & (df["IstRad"] == 1)
        & df["year"].between(2018, 2025)
    ].copy()

    out = out.dropna(subset=["latitude", "longitude", "year", "month", "hour", "day_of_week", "accident_severity"])

    out["year"] = out["year"].astype(int)
    out["month"] = out["month"].astype(int)
    out["hour"] = out["hour"].astype(int)
    out["day_of_week"] = out["day_of_week"].astype(int)
    out["accident_severity"] = out["accident_severity"].astype(int)

    # Standard labels.
    sev_map = {1: "fatal", 2: "serious_injury", 3: "minor_injury"}
    out["severity_label"] = out["accident_severity"].map(sev_map).fillna("unknown")
    out["is_fatal"] = (out["accident_severity"] == 1).astype(int)
    out["is_ksi"] = out["accident_severity"].isin([1, 2]).astype(int)
    out["serious_or_fatal"] = out["is_ksi"]

    out["is_weekend"] = out["day_of_week"].isin([1, 7]).astype(int)
    out["is_rush_hour"] = (out["hour"].between(7, 9) | out["hour"].between(16, 18)).astype(int)
    out["is_night"] = ((out["hour"] >= 22) | (out["hour"] <= 4)).astype(int)
    out["season"] = out["month"].apply(get_season)

    for src, dst in [
        ("IstPKW", "is_car"),
        ("IstGkfz", "is_truck"),
        ("IstSonstige", "is_other_participant"),
    ]:
        if src in out.columns:
            out[dst] = pd.to_numeric(out[src], errors="coerce").fillna(0).astype(int)
        else:
            out[dst] = 0

    if "ULICHTVERH" in out.columns:
        out["light_label"] = pd.to_numeric(out["ULICHTVERH"], errors="coerce").map(LIGHT_LABELS).fillna("unknown")
    else:
        out["ULICHTVERH"] = np.nan
        out["light_label"] = "unknown"

    if "STRZUSTAND" in out.columns:
        out["surface_label"] = pd.to_numeric(out["STRZUSTAND"], errors="coerce").map(SURFACE_LABELS).fillna("unknown")
    else:
        out["STRZUSTAND"] = np.nan
        out["surface_label"] = "unknown"

    # Preserve columns useful downstream.
    keep = [
        "year", "month", "hour", "day_of_week", "season",
        "latitude", "longitude", "state_code",
        "accident_severity", "severity_label", "is_fatal", "is_ksi", "serious_or_fatal",
        "is_weekend", "is_rush_hour", "is_night",
        "IstRad", "IstPKW", "IstGkfz", "IstSonstige", "UTYP1",
        "ULICHTVERH", "STRZUSTAND", "light_label", "surface_label",
        "is_car", "is_truck", "is_other_participant",
    ]
    keep = [c for c in keep if c in out.columns]
    out = out[keep].reset_index(drop=True)

    # Canary copied from the improved notebook.
    assert "dark_unlit" not in set(out.get("light_label", pd.Series(dtype=str)).astype(str)), (
        "Stale light labels detected. Regenerate berlin_bike_2018_2025.csv."
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_file, index=False)
    print(f"saved {output_file} | {len(out):,} rows × {out.shape[1]} columns")
    return out
