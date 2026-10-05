from itertools import zip_longest


def parse(version):
    return [int(part) for part in version.split(".")]


def compare_versions(a, b):
    for left, right in zip_longest(parse(a), parse(b), fillvalue=0):
        if left != right:
            return -1 if left < right else 1
    return 0
