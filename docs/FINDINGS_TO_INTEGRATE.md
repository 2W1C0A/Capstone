## src/data_pipeline.py — LIGHT_LABELS contradicts notebook 01

**Found by:** Yasin (owner of `01_eda_and_analysis`)
**File:** `src/data_pipeline.py`
**Owner to apply:** Elena
**Bundle with:** the Step 1 graph regeneration (the clean CSV must be rebuilt anyway)

### Problem

`LIGHT_LABELS` maps `2 -> "dark_lit"` and defines a code `3 -> "dark_unlit_old_code_check"`.
Both are wrong. `ULICHTVERH` in the Unfallatlas has only three valid codes (0, 1, 2)
and carries **no lit/unlit distinction**. Code 2 means darkness, full stop. This is
verified in `01_eda_and_analysis` cell [11] against the official Unfallatlas
codebook (DSB), which is the source of truth for this field.

### Replace

    LIGHT_LABELS = {
        0: "daylight",
        1: "twilight",
        2: "darkness",
    }

Drop the `3:` entry entirely — the code does not exist, and `.fillna("unknown")`
already handles anything unexpected.

### The canary is dead — fix it too

Two sites assert `"dark_unlit" not in set(...)`. That is an exact set-membership
test, so the current wrong label `"dark_unlit_old_code_check"` passes it. The guard
written to catch this bug walks straight past it. Replace both sites with a
substring check:

    _BAD_LIGHT_LABELS = ("dark_lit", "dark_unlit")

    def _assert_light_labels(labels) -> None:
        bad = {s for s in set(map(str, labels))
               if any(b in s for b in _BAD_LIGHT_LABELS)}
        assert not bad, (
            f"Stale light labels {sorted(bad)}. ULICHTVERH has only codes 0/1/2 "
            "and carries no lit/unlit distinction. "
            "Regenerate berlin_bike_2018_2025.csv."
        )

### After the fix

`data/processed/berlin_bike_2018_2025.csv` has the wrong labels baked in. It must
be regenerated (`prepare_berlin_bicycle_accidents` without `use_existing_clean`).

### Why it matters

No model currently consumes `light_label`, so runtime impact today is zero. But:

1. A reader comparing notebook 01 with `src/` sees the team contradicting itself
   on its own data dictionary.
2. It blocks adding `light_label` / `surface_label` to the severity model, which
   is the only remaining legitimate improvement there (notebook 04 limitation 2
   is also stale: `data_pipeline.py` *does* retain `ULICHTVERH` and `STRZUSTAND`).