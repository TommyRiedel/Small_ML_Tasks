-- ============================================
-- Employee Skills and Certifications Distribution
-- Basierend auf tatsächlicher Tabellenstruktur
-- ============================================

-- Tabellen:
-- skills: employee_id (INT), name (VARCHAR)
-- certifications: employee_id (INT), name (VARCHAR)

-- Prüfe ob Daten in beiden Tabellen sind:
SELECT 'skills' AS table_name, COUNT(*) AS row_count FROM skills
UNION ALL
SELECT 'certifications', COUNT(*) FROM certifications;

-- Zeige Beispieldaten:
SELECT 'skills' AS source, employee_id, name FROM skills LIMIT 5
UNION ALL
SELECT 'certifications', employee_id, name FROM certifications LIMIT 5;

-- Hauptquery:
SELECT 
    qualification_type,
    qualification_name,
    COUNT(DISTINCT employee_id) AS total_unique_employees
FROM (
    SELECT 
        'Skill' AS qualification_type,
        name AS qualification_name,
        employee_id
    FROM skills
    
    UNION ALL
    
    SELECT 
        'Certification' AS qualification_type,
        name AS qualification_name,
        employee_id
    FROM certifications
) AS combined_qualifications
GROUP BY qualification_type, qualification_name
ORDER BY qualification_type ASC, qualification_name ASC;
