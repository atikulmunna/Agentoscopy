def deep_merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def flatten_keys(config, prefix=""):
    keys = []
    for key, value in config.items():
        path = f"{prefix}{key}"
        keys += flatten_keys(value, path + ".") if isinstance(value, dict) else [path]
    return keys
