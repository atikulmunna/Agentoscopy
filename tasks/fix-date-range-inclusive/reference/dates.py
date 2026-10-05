from datetime import timedelta


def date_range(start, end):
    days = (end - start).days
    if days < 0:
        raise ValueError("end is before start")
    return [start + timedelta(days=offset) for offset in range(days + 1)]
