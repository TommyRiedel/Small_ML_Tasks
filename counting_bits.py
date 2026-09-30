# ============================================
# Counting Bits - Positionen der gesetzten Bits
# ============================================

def decimal_to_binary(n):
    """Konvertiert Dezimalzahl in Binärdarstellung"""
    return bin(n)[2:]  # Entfernt '0b' Prefix


def decimal_to_binary_manual(n):
    """Manuelle Konvertierung ohne bin()"""
    if n == 0:
        return '0'
    
    binary = ''
    while n > 0:
        binary = str(n % 2) + binary
        n //= 2
    return binary


def decimal_to_binary_list(n):
    """Gibt Liste von Bits zurück [1, 0, 0, 1, 0, 1]"""
    return [int(bit) for bit in bin(n)[2:]]


def count_bits_positions(n):
    """
    Gibt [Anzahl, pos1, pos2, ...] zurück
    Positionen von rechts nach links (0-indiziert)
    
    Beispiel: 37 = 0b100101
    - Bit 0 (2^0 = 1): gesetzt
    - Bit 2 (2^2 = 4): gesetzt  
    - Bit 5 (2^5 = 32): gesetzt
    → [3, 0, 2, 5]
    """
    binary = decimal_to_binary(n)
    count = binary.count('1')
    
    # Finde Positionen von rechts nach links
    positions = []
    for i, bit in enumerate(reversed(binary)):
        if bit == '1':
            positions.append(i)
    
    return [count] + positions


def count_bits_positions_v2(n):
    """Alternative: Bit-Manipulation"""
    positions = []
    pos = 0
    temp = n
    
    while temp > 0:
        if temp & 1:  # Prüfe ob letztes Bit gesetzt
            positions.append(pos)
        temp >>= 1  # Shift nach rechts
        pos += 1
    
    return [len(positions)] + positions


def count_bits_positions_v3(n):
    """Kompakte Version mit List Comprehension"""
    binary = decimal_to_binary(n)
    positions = [i for i, bit in enumerate(reversed(binary)) if bit == '1']
    return [len(positions)] + positions


# Test
if __name__ == "__main__":
    print("=== Dezimal zu Binär Konvertierung ===")
    test_nums = [37, 5, 15, 255]
    for num in test_nums:
        print(f"{num:3d} → {decimal_to_binary(num):>8s} (builtin)")
        print(f"     → {decimal_to_binary_manual(num):>8s} (manual)")
        print(f"     → {decimal_to_binary_list(num)} (list)")
        print()
    
    print("\n=== Bit-Positionen finden ===")
    test_cases = [37, 5, 15, 1, 0, 255]
    
    for num in test_cases:
        result = count_bits_positions(num)
        print(f"{num:3d} = {bin(num):>10s} → {result}")
    
    print("\n=== Beispiel 37 im Detail ===")
    n = 37
    print(f"Dezimal: {n}")
    print(f"Binär:   {bin(n)}")
    print(f"Ergebnis: {count_bits_positions(n)}")
    print(f"Bedeutung: {n} = 2^5 + 2^2 + 2^0 = 32 + 4 + 1")
