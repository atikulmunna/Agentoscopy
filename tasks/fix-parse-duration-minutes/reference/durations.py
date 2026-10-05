import re

UNITS = {"h": 3600, "m": 60, "s": 1}


def parse_duration(text):
    if not re.fullmatch(r"(\d+[hms])+", text):
        raise ValueError(f"not a duration: {text!r}")
    return sum(int(amount) * UNITS[unit] for amount, unit in re.findall(r"(\d+)([hms])", text))
