def parse(version):
    return [int(part) for part in version.split(".")]


def compare_versions(a, b):
    raise NotImplementedError
