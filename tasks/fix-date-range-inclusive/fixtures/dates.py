from datetime import timedelta


def date_range(start, end):
    days = (end - start).days
    return [start + timedelta(days=offset) for offset in range(days)]
