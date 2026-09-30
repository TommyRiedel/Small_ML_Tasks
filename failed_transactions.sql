-- ============================================
-- Failed Transactions Report
-- ============================================
-- Identifiziert User mit >= 5 fehlgeschlagenen Transaktionen
-- und zeigt die Anzahl unterschiedlicher Zahlungsmethoden

-- Annahme: Tabelle 'transactions' mit Spalten:
-- - user_id
-- - transaction_id
-- - status ('failed', 'success', etc.)
-- - payment_method ('credit_card', 'paypal', 'bank_transfer', etc.)

-- ============================================
-- Lösung 1: Mit CTE (Common Table Expression)
-- ============================================
WITH failed_transactions AS (
    SELECT 
        user_id,
        COUNT(*) AS failed_transactions,
        COUNT(DISTINCT payment_method) AS distinct_payment_methods
    FROM transactions
    WHERE status = 'failed'
    GROUP BY user_id
)
SELECT 
    user_id,
    failed_transactions,
    distinct_payment_methods
FROM failed_transactions
WHERE failed_transactions >= 5
ORDER BY failed_transactions DESC;


-- ============================================
-- Lösung 2: Direkt mit HAVING (kompakter)
-- ============================================
SELECT 
    user_id,
    COUNT(*) AS failed_transactions,
    COUNT(DISTINCT payment_method) AS distinct_payment_methods
FROM transactions
WHERE status = 'failed'
GROUP BY user_id
HAVING COUNT(*) >= 5
ORDER BY failed_transactions DESC;


-- ============================================
-- Lösung 3: Mit zusätzlichen Details
-- ============================================
SELECT 
    user_id,
    COUNT(*) AS failed_transactions,
    COUNT(DISTINCT payment_method) AS distinct_payment_methods,
    GROUP_CONCAT(DISTINCT payment_method) AS payment_methods_used,
    MIN(transaction_date) AS first_failure,
    MAX(transaction_date) AS last_failure
FROM transactions
WHERE status = 'failed'
GROUP BY user_id
HAVING COUNT(*) >= 5
ORDER BY failed_transactions DESC;


-- ============================================
-- Lösung 4: Mit Ranking (Top 10 User)
-- ============================================
WITH failed_summary AS (
    SELECT 
        user_id,
        COUNT(*) AS failed_transactions,
        COUNT(DISTINCT payment_method) AS distinct_payment_methods
    FROM transactions
    WHERE status = 'failed'
    GROUP BY user_id
    HAVING COUNT(*) >= 5
)
SELECT 
    user_id,
    failed_transactions,
    distinct_payment_methods,
    RANK() OVER (ORDER BY failed_transactions DESC) AS rank
FROM failed_summary
ORDER BY failed_transactions DESC
LIMIT 10;


-- ============================================
-- Test-Daten zum Ausprobieren
-- ============================================
/*
CREATE TABLE transactions (
    transaction_id INT PRIMARY KEY,
    user_id INT,
    status VARCHAR(20),
    payment_method VARCHAR(50),
    transaction_date DATE
);

INSERT INTO transactions VALUES
(1, 101, 'failed', 'credit_card', '2024-01-01'),
(2, 101, 'failed', 'paypal', '2024-01-02'),
(3, 101, 'failed', 'credit_card', '2024-01-03'),
(4, 101, 'failed', 'bank_transfer', '2024-01-04'),
(5, 101, 'failed', 'paypal', '2024-01-05'),
(6, 101, 'success', 'credit_card', '2024-01-06'),
(7, 102, 'failed', 'credit_card', '2024-01-01'),
(8, 102, 'failed', 'credit_card', '2024-01-02'),
(9, 102, 'failed', 'paypal', '2024-01-03'),
(10, 103, 'failed', 'paypal', '2024-01-01'),
(11, 103, 'failed', 'paypal', '2024-01-02');

-- Erwartetes Ergebnis:
-- user_id | failed_transactions | distinct_payment_methods
-- --------|---------------------|-------------------------
-- 101     | 5                   | 3
*/


-- ============================================
-- Erklärung der Lösung
-- ============================================
/*
1. WHERE status = 'failed'
   → Filtert nur fehlgeschlagene Transaktionen

2. GROUP BY user_id
   → Gruppiert nach User

3. COUNT(*) AS failed_transactions
   → Zählt alle fehlgeschlagenen Transaktionen pro User

4. COUNT(DISTINCT payment_method) AS distinct_payment_methods
   → Zählt unterschiedliche Zahlungsmethoden

5. HAVING COUNT(*) >= 5
   → Filtert nur User mit mindestens 5 Fehlern

6. ORDER BY failed_transactions DESC
   → Sortiert nach Anzahl der Fehler (absteigend)
*/
