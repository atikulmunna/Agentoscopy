def mean(values):
    return sum(values) / len(values)


def moving_average(values, window):
    if window < 1:
        raise ValueError("window must be at least 1")
    return [mean(values[i:i + window]) for i in range(len(values) - window + 1)]
