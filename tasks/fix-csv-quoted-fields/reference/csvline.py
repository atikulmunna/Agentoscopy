import csv


def parse_line(line):
    return next(csv.reader([line]))
