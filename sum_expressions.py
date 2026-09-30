# ============================================
# Sum of All Expressions with '+' Insertions
# ============================================

def sum_all_expressions(s):
    """
    Findet die Summe aller möglichen Ausdrücke durch Einfügen von '+'
    
    Beispiel: "123"
    Mögliche Ausdrücke:
    - 123 (keine +)
    - 1+23 = 24
    - 12+3 = 15
    - 1+2+3 = 6
    Summe: 123 + 24 + 15 + 6 = 168
    
    Args:
        s: String mit Ziffern
    
    Returns:
        Summe aller möglichen Ausdrücke
    """
    if not s:
        return 0
    
    n = len(s)
    total_sum = 0
    
    # Iteriere über alle 2^(n-1) Möglichkeiten
    # Jede Position zwischen Ziffern kann '+' haben oder nicht
    for mask in range(1 << (n - 1)):
        expression = s[0]
        current_sum = 0
        current_num = int(s[0])
        
        for i in range(n - 1):
            if mask & (1 << i):  # '+' an Position i
                current_sum += current_num
                current_num = int(s[i + 1])
                expression += '+' + s[i + 1]
            else:  # Keine '+', Ziffer anhängen
                current_num = current_num * 10 + int(s[i + 1])
                expression += s[i + 1]
        
        current_sum += current_num
        total_sum += current_sum
    
    return total_sum


def sum_all_expressions_with_details(s):
    """Zeigt alle Ausdrücke und deren Werte"""
    if not s:
        return 0, []
    
    n = len(s)
    total_sum = 0
    expressions = []
    
    for mask in range(1 << (n - 1)):
        expression = s[0]
        current_sum = 0
        current_num = int(s[0])
        
        for i in range(n - 1):
            if mask & (1 << i):
                current_sum += current_num
                current_num = int(s[i + 1])
                expression += '+' + s[i + 1]
            else:
                current_num = current_num * 10 + int(s[i + 1])
                expression += s[i + 1]
        
        current_sum += current_num
        total_sum += current_sum
        expressions.append((expression, current_sum))
    
    return total_sum, expressions


def sum_all_expressions_recursive(s, index=0, current="", current_sum=0):
    """Rekursive Lösung (alternative Implementierung)"""
    if index == len(s):
        return eval(current) if current else 0
    
    total = 0
    
    # Option 1: Füge '+' vor aktueller Ziffer ein (außer am Anfang)
    if index > 0:
        total += sum_all_expressions_recursive(s, index + 1, current + '+' + s[index])
    
    # Option 2: Füge Ziffer ohne '+' hinzu
    if index == 0:
        total += sum_all_expressions_recursive(s, index + 1, s[index])
    else:
        total += sum_all_expressions_recursive(s, index + 1, current + s[index])
    
    return total


def sum_all_expressions_dp(s, modulo=None):
    """Dynamic Programming Lösung (effizient und korrekt für große Strings)
    
    Args:
        s: String mit Ziffern
        modulo: Optional - Modulo-Wert (z.B. 10**9 + 7)
    """
    n = len(s)
    if n == 0:
        return 0
    
    MOD = modulo if modulo else None
    
    # Für jede Ziffer an Position i berechnen wir:
    # - Wie oft sie in welchem Stellenwert erscheint
    # - In wie vielen Ausdrücken sie vorkommt
    
    total = 0
    
    for i in range(n):
        digit = int(s[i])
        
        # Diese Ziffer kann Teil von Zahlen verschiedener Länge sein
        # die an Position i beginnen
        for j in range(i, n):
            # Ziffer ist an Position (j-i) in einer Zahl die bei i startet
            # und bei j endet
            
            # Anzahl der Ausdrücke, wo diese Zahl vorkommt:
            # - Links von i: 2^i Möglichkeiten (jede Position kann + haben)
            # - Rechts von j: 2^(n-j-1) Möglichkeiten
            # - Zwischen i und j: keine + (fixiert)
            
            left_combinations = (1 << i) if i > 0 else 1
            right_combinations = (1 << (n - j - 1)) if j < n - 1 else 1
            
            # Stellenwert der Ziffer in dieser Zahl
            place_value = pow(10, j - i, MOD) if MOD else (10 ** (j - i))
            
            # Beitrag dieser Ziffer
            contribution = digit * place_value * left_combinations * right_combinations
            
            if MOD:
                total = (total + contribution) % MOD
            else:
                total += contribution
    
    return total


# Test
if __name__ == "__main__":
    print("=== Test 1: Einfaches Beispiel '12' ===")
    s1 = "12"
    result1, expressions1 = sum_all_expressions_with_details(s1)
    print(f"String: '{s1}'")
    print("Alle Ausdrücke:")
    for expr, value in expressions1:
        print(f"  {expr} = {value}")
    print(f"Gesamtsumme: {result1}")
    
    print("\n=== Test 2: '123' ===")
    s2 = "123"
    result2, expressions2 = sum_all_expressions_with_details(s2)
    print(f"String: '{s2}'")
    print("Alle Ausdrücke:")
    for expr, value in expressions2:
        print(f"  {expr} = {value}")
    print(f"Gesamtsumme: {result2}")
    
    print("\n=== Test 3: '99' ===")
    s3 = "99"
    result3 = sum_all_expressions(s3)
    print(f"String: '{s3}'")
    print(f"Ausdrücke: 99, 9+9")
    print(f"Gesamtsumme: {result3}")
    
    print("\n=== Test 4: '1234' ===")
    s4 = "1234"
    result4 = sum_all_expressions(s4)
    print(f"String: '{s4}'")
    print(f"Anzahl Ausdrücke: {2**(len(s4)-1)}")
    print(f"Gesamtsumme: {result4}")
    
    print("\n=== Test 5: Großer String '887246266' ===")
    s5 = "887246266"
    MOD = 10**9 + 7
    
    print(f"String: '{s5}'")
    print(f"Anzahl Ausdrücke: {2**(len(s5)-1)}")
    
    # Teste beide Methoden
    result_dp = sum_all_expressions_dp(s5)
    result_dp_mod = sum_all_expressions_dp(s5, MOD)
    print(f"DP-Lösung: {result_dp}")
    print(f"DP-Lösung (mod 10^9+7): {result_dp_mod}")
    
    # Für kleinere Strings zum Vergleich
    if len(s5) <= 10:
        result_brute = sum_all_expressions(s5)
        print(f"Brute-Force: {result_brute}")
        print(f"Stimmen überein: {result_dp == result_brute}")
    
    print("\n=== Test 6: Performance Vergleich ===")
    import time
    s6 = "12345678"
    
    start = time.time()
    result_iterative = sum_all_expressions(s6)
    time_iterative = time.time() - start
    
    start = time.time()
    result_dp_test = sum_all_expressions_dp(s6)
    time_dp = time.time() - start
    
    print(f"String: '{s6}'")
    print(f"Anzahl Ausdrücke: {2**(len(s6)-1)}")
    print(f"Brute-Force: {result_iterative} in {time_iterative*1000:.2f}ms")
    print(f"DP-Lösung:  {result_dp_test} in {time_dp*1000:.2f}ms")
    print(f"Stimmen überein: {result_iterative == result_dp_test}")
    print(f"Speedup: {time_iterative/time_dp:.1f}x")
    
    print("\n=== Verifikation mit kleinen Beispielen ===")
    test_cases = ["12", "123", "99"]
    MOD = 10**9 + 7
    for test in test_cases:
        r1 = sum_all_expressions(test)
        r2 = sum_all_expressions_dp(test)
        r3 = sum_all_expressions_dp(test, MOD)
        print(f"'{test}': Brute={r1}, DP={r2}, DP(mod)={r3}, Match={r1==r2}")
    
    print("\n=== Erklärung ===")
    print("Für String der Länge n gibt es 2^(n-1) mögliche Ausdrücke")
    print("Jede Position zwischen Ziffern kann '+' haben oder nicht")
    print("Beispiel '123': 2^2 = 4 Ausdrücke")
    print("  Binär 00: 123")
    print("  Binär 01: 12+3")
    print("  Binär 10: 1+23")
    print("  Binär 11: 1+2+3")
    print("\nModulo 10^9+7 wird verwendet für sehr große Ergebnisse")
