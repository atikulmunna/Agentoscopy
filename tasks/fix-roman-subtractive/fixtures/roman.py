NUMERALS = [(1000, "M"), (500, "D"), (100, "C"), (50, "L"), (10, "X"), (5, "V"), (1, "I")]


def to_roman(number):
    result = ""
    for value, symbol in NUMERALS:
        count, number = divmod(number, value)
        result += symbol * count
    return result
