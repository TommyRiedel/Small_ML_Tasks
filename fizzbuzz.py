# ============================================
# FizzBuzz Problem
# ============================================

# Klassische Lösung
def fizzbuzz_classic(n):
    """Gibt FizzBuzz für Zahlen von 1 bis n aus"""
    for i in range(1, n + 1):
        if i % 15 == 0:
            print("FizzBuzz")
        elif i % 3 == 0:
            print("Fizz")
        elif i % 5 == 0:
            print("Buzz")
        else:
            print(i)

# Kompakte Lösung
def fizzbuzz_compact(n):
    """Kompakte Version mit String-Konkatenation"""
    for i in range(1, n + 1):
        output = ("Fizz" * (i % 3 == 0)) + ("Buzz" * (i % 5 == 0))
        print(output or i)

# List Comprehension
def fizzbuzz_list(n):
    """Gibt Liste mit FizzBuzz-Werten zurück"""
    return [
        "FizzBuzz" if i % 15 == 0 else
        "Fizz" if i % 3 == 0 else
        "Buzz" if i % 5 == 0 else
        i
        for i in range(1, n + 1)
    ]

# Test
if __name__ == "__main__":
    print("=== Klassische Lösung ===")
    fizzbuzz_classic(15)
    
    print("\n=== Kompakte Lösung ===")
    fizzbuzz_compact(15)
    
    print("\n=== List Comprehension ===")
    print(fizzbuzz_list(15))
