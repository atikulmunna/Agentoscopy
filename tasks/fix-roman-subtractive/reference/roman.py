NUMERALS = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
    (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]


def to_roman(number):
    if not 1 <= number <= 3999:
        raise ValueError("roman numerals cover 1 to 3999")
    result = ""
    for value, symbol in NUMERALS:
        count, number = divmod(number, value)
        result += symbol * count
    return result
