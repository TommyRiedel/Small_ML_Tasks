# ============================================
# Arabische Zahlen → Römische Zahlen
# ============================================

def arabic_to_roman(num):
    """
    Konvertiert arabische Zahlen (1-3999) in römische Zahlen
    Akzeptiert einzelne Zahl oder Liste von Zahlen
    
    Beispiele:
    - 37 → XXXVII
    - [37, 58] → ['XXXVII', 'LVIII']
    - 1994 → MCMXCIV
    """
    # Wenn Liste übergeben wird, rekursiv für jede Zahl aufrufen
    if isinstance(num, list):
        return [arabic_to_roman(n) for n in num]
    
    # Fehlerbehandlung für einzelne Zahl
    if not isinstance(num, int):
        raise TypeError("Nur ganze Zahlen erlaubt")
    if num <= 0 or num >= 4000:
        raise ValueError("Zahl muss zwischen 1 und 3999 liegen")
    
    # Mapping von groß nach klein (inkl. Subtraktionsregeln)
    values = [
        (1000, 'M'),
        (900, 'CM'),
        (500, 'D'),
        (400, 'CD'),
        (100, 'C'),
        (90, 'XC'),
        (50, 'L'),
        (40, 'XL'),
        (10, 'X'),
        (9, 'IX'),
        (5, 'V'),
        (4, 'IV'),
        (1, 'I')
    ]
    
    result = ''
    for value, numeral in values:
        count = num // value
        if count:
            result += numeral * count
            num -= value * count
    
    return result


def arabic_to_roman_v2(num):
    """Alternative kompakte Version"""
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
    
    result = ''
    for i, v in enumerate(val):
        count = num // v
        result += syms[i] * count
        num -= v * count
    
    return result


def roman_to_arabic(roman):
    """Bonus: Römische → Arabische Zahlen"""
    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    
    result = 0
    prev_value = 0
    
    for char in reversed(roman):
        value = values[char]
        if value < prev_value:
            result -= value
        else:
            result += value
        prev_value = value
    
    return result


# Test
if __name__ == "__main__":
    print("=== Einzelne Zahlen ===")
    print(f"37 → {arabic_to_roman(37)}")
    print(f"1994 → {arabic_to_roman(1994)}")
    
    print("\n=== Listen von Zahlen ===")
    numbers = [1, 4, 9, 37, 58, 99, 444, 1994, 2024, 3999]
    romans = arabic_to_roman(numbers)
    for num, roman in zip(numbers, romans):
        print(f"{num:4d} → {roman:15s} (zurück: {roman_to_arabic(roman)})")
    
    print("\n=== Arabisch → Römisch ===")
    test_cases = [1, 4, 9, 37, 58, 99, 444, 1994, 2024, 3999]
    
    for num in test_cases:
        roman = arabic_to_roman(num)
        print(f"{num:4d} → {roman:15s} (zurück: {roman_to_arabic(roman)})")
    
    print("\n=== Spezielle Beispiele ===")
    examples = {
        37: "XXXVII (30 + 5 + 2)",
        1994: "MCMXCIV (1000 + 900 + 90 + 4)",
        444: "CDXLIV (400 + 40 + 4)",
        2024: "MMXXIV (2000 + 20 + 4)"
    }
    
    for num, explanation in examples.items():
        print(f"{num} → {arabic_to_roman(num):10s} = {explanation}")
