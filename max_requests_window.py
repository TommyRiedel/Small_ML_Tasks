# ============================================
# Max Requests in Time Window (Sliding Window)
# ============================================

def max_requests_in_window(timestamps, window):
    """
    Findet die maximale Anzahl von Requests in einem Zeitfenster
    
    Args:
        timestamps: Liste von Zeitstempeln (sortiert)
        window: Größe des Zeitfensters
    
    Returns:
        Maximale Anzahl von Requests in [x, x+window-1]
    
    Beispiel:
        timestamps = [1, 2, 3, 9, 10, 11]
        window = 5
        → 4 (Fenster [1, 5] enthält 1,2,3 und [9,13] enthält 9,10,11)
    """
    if not timestamps:
        return 0
    
    max_count = 0
    left = 0
    
    # Sliding Window mit zwei Pointern
    for right in range(len(timestamps)):
        # Verschiebe linken Pointer, bis Fenster gültig ist
        while timestamps[right] - timestamps[left] >= window:
            left += 1
        
        # Zähle Requests im aktuellen Fenster
        max_count = max(max_count, right - left + 1)
    
    return max_count


def max_requests_in_window_v2(timestamps, window):
    """Alternative: Brute Force (langsamer, aber einfacher zu verstehen)"""
    if not timestamps:
        return 0
    
    max_count = 0
    
    for i in range(len(timestamps)):
        count = 0
        window_end = timestamps[i] + window - 1
        
        for j in range(i, len(timestamps)):
            if timestamps[j] <= window_end:
                count += 1
            else:
                break
        
        max_count = max(max_count, count)
    
    return max_count


def max_requests_with_details(timestamps, window):
    """Gibt auch das beste Fenster zurück"""
    if not timestamps:
        return 0, None
    
    max_count = 0
    best_window = None
    left = 0
    
    for right in range(len(timestamps)):
        while timestamps[right] - timestamps[left] >= window:
            left += 1
        
        count = right - left + 1
        if count > max_count:
            max_count = count
            best_window = (timestamps[left], timestamps[left] + window - 1)
    
    return max_count, best_window


# Test
if __name__ == "__main__":
    print("=== Test 1: Einfaches Beispiel ===")
    timestamps1 = [1, 2, 3, 9, 10, 11]
    window1 = 5
    result1 = max_requests_in_window(timestamps1, window1)
    count, best = max_requests_with_details(timestamps1, window1)
    print(f"Timestamps: {timestamps1}")
    print(f"Window: {window1}")
    print(f"Max Requests: {result1}")
    print(f"Bestes Fenster: [{best[0]}, {best[1]}]")
    
    print("\n=== Test 2: Alle Requests im Fenster ===")
    timestamps2 = [1, 2, 3, 4, 5]
    window2 = 10
    result2 = max_requests_in_window(timestamps2, window2)
    print(f"Timestamps: {timestamps2}")
    print(f"Window: {window2}")
    print(f"Max Requests: {result2}")
    
    print("\n=== Test 3: Keine überlappenden Fenster ===")
    timestamps3 = [1, 10, 20, 30]
    window3 = 5
    result3 = max_requests_in_window(timestamps3, window3)
    print(f"Timestamps: {timestamps3}")
    print(f"Window: {window3}")
    print(f"Max Requests: {result3}")
    
    print("\n=== Test 4: Großes Beispiel ===")
    timestamps4 = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    window4 = 6
    result4 = max_requests_in_window(timestamps4, window4)
    count4, best4 = max_requests_with_details(timestamps4, window4)
    print(f"Timestamps: {timestamps4}")
    print(f"Window: {window4}")
    print(f"Max Requests: {result4}")
    print(f"Bestes Fenster: [{best4[0]}, {best4[1]}]")
    
    print("\n=== Vergleich: Sliding Window vs Brute Force ===")
    import time
    timestamps5 = list(range(0, 1000, 2))  # [0, 2, 4, ..., 998]
    window5 = 100
    
    start = time.time()
    result_sliding = max_requests_in_window(timestamps5, window5)
    time_sliding = time.time() - start
    
    start = time.time()
    result_brute = max_requests_in_window_v2(timestamps5, window5)
    time_brute = time.time() - start
    
    print(f"Sliding Window: {result_sliding} in {time_sliding*1000:.2f}ms")
    print(f"Brute Force:    {result_brute} in {time_brute*1000:.2f}ms")
    print(f"Speedup: {time_brute/time_sliding:.1f}x")
