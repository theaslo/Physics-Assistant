"""Validate numerical givens without supplying an unstated physical setup."""

import json
import math


def data_object(value):
    data = json.loads(value) if isinstance(value, str) else value
    if not isinstance(data, dict) or not data:
        raise ValueError("Provide a JSON object containing the actual known quantities")

    def check(item):
        if isinstance(item, dict):
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        elif isinstance(item, (int, float)) and not math.isfinite(item):
            raise ValueError("All numerical quantities must be finite")
    check(data)
    return data


def number(data, *keys, positive=False, nonnegative=False):
    raw = next((data[key] for key in keys if data.get(key) is not None), None)
    if raw is None or isinstance(raw, bool):
        raise ValueError("Provide a numerical value for " + keys[0])
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(keys[0] + " must be finite")
    if positive and value <= 0:
        raise ValueError(keys[0] + " must be positive")
    if nonnegative and value < 0:
        raise ValueError(keys[0] + " must be nonnegative")
    return value
