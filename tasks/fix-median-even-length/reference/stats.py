def median(values):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("median of empty data")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2
