from __future__ import annotations

from typing import List, Optional

import numpy as np


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if np.isnan(parsed):
        return None
    return float(parsed)


def parse_int(value: object) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and not np.isnan(float(v))]
    if not clean:
        return None
    return float(np.mean(np.asarray(clean, dtype=np.float64)))


def std_or_none(values: List[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and not np.isnan(float(v))]
    if not clean:
        return None
    return float(np.std(np.asarray(clean, dtype=np.float64)))

