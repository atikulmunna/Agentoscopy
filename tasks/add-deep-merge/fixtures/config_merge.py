def deep_merge(base, override):
    raise NotImplementedError


def flatten_keys(config, prefix=""):
    keys = []
    for key, value in config.items():
        path = f"{prefix}{key}"
        keys += flatten_keys(value, path + ".") if isinstance(value, dict) else [path]
    return keys
