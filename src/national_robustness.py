from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path

import pandas as pd

from .data_pipeline import COLUMN_MAP


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (centre - margin, centre + margin)


def run_national_robustness(raw_dir: str | Path = "data/raw", output_file: str | Path = "reports/national_robustness_summary.json") -> dict:
    """Optional read-only national robustness check.

    This is not part of the default app pipeline because it needs national raw
    Unfallatlas files. It supports the presentation claim that the truck-severity
    mechanism is not a Berlin small-sample artefact.
    """
    raw_dir = Path(raw_dir)
    paths = sorted(
        p for p in glob.glob(str(raw_dir / "Unfallorte*.txt")) + glob.glob(str(raw_dir / "Unfallorte*.csv"))
        if "LinRef" not in os.path.basename(p)
    )
    if not paths:
        raise FileNotFoundError("No national Unfallatlas files found in data/raw/.")

    frames = []
    for p in paths:
        d = pd.read_csv(p, sep=";", encoding="utf-8-sig", low_memory=False).rename(columns=COLUMN_MAP)
        frames.append(d)

    df = pd.concat(frames, ignore_index=True)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    nat = df[
        (pd.to_numeric(df["IstRad"], errors="coerce") == 1)
        & df["year"].between(2019, 2025)
        & df["IstGkfz"].notna()
    ].copy()

    nat["is_ksi"] = pd.to_numeric(nat["accident_severity"], errors="coerce").isin([1, 2]).astype(int)
    nat["is_fatal"] = (pd.to_numeric(nat["accident_severity"], errors="coerce") == 1).astype(int)

    truck = nat[pd.to_numeric(nat["IstGkfz"], errors="coerce") == 1]
    car = nat[(pd.to_numeric(nat["IstPKW"], errors="coerce") == 1) & (pd.to_numeric(nat["IstGkfz"], errors="coerce") == 0)]

    def summarise(label, frame):
        n = len(frame)
        ksi = int(frame["is_ksi"].sum())
        fatal = int(frame["is_fatal"].sum())
        ksi_lo, ksi_hi = wilson_ci(ksi, n)
        fat_lo, fat_hi = wilson_ci(fatal, n)
        return {
            "group": label,
            "crashes": n,
            "ksi": ksi,
            "ksi_rate": ksi / n,
            "ksi_ci": [ksi_lo, ksi_hi],
            "fatal": fatal,
            "fatal_rate": fatal / n,
            "fatal_ci": [fat_lo, fat_hi],
        }

    result = {
        "window": "2019-2025",
        "national_bicycle_crashes": int(len(nat)),
        "truck": summarise("truck", truck),
        "car_only": summarise("car_only", car),
        "note": (
            "Every comparison is conditional on a crash having occurred. It measures severity, "
            "not collision probability; no cycling exposure data is used."
        ),
    }
    result["ksi_risk_ratio_truck_vs_car"] = result["truck"]["ksi_rate"] / result["car_only"]["ksi_rate"]
    result["fatality_risk_ratio_truck_vs_car"] = result["truck"]["fatal_rate"] / result["car_only"]["fatal_rate"]

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
