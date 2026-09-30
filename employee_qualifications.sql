-- ============================================
-- Employee Skills and Certifications Distribution
-- ============================================

-- Annahme: Zwei Tabellen
-- 1. employee_skills (employee_id, skill_name)
-- 2. employee_certifications (employee_id, certification_name)

-- ============================================
-- Lösung 1: UNION ALL mit Aggregation
-- ============================================
SELECT 
    qualification_type,
    qualification_name,
    COUNT(DISTINCT employee_id) AS total_unique_employees
FROM (
    -- Skills
    SELECT 
        'Skill' AS qualification_type,
        skill_name AS qualification_name,
        employee_id
    FROM employee_skills
    
    UNION ALL
    
    -- Certifications
    SELECT 
        'Certification' AS qualification_type,
        certification_name AS qualification_name,
        employee_id
    FROM employee_certifications
) AS combined_qualifications
GROUP BY qualification_type, qualification_name
ORDER BY qualification_type ASC, qualification_name ASC;


-- ============================================
-- Lösung 2: Separate CTEs (übersichtlicher)
-- ============================================
WITH skills AS (
    SELECT 
        'Skill' AS qualification_type,
        skill_name AS qualification_name,
        COUNT(DISTINCT employee_id) AS total_unique_employees
    FROM employee_skills
    GROUP BY skill_name
),
certifications AS (
    SELECT 
        'Certification' AS qualification_type,
        certification_name AS qualification_name,
        COUNT(DISTINCT employee_id) AS total_unique_employees
    FROM employee_certifications
    GROUP BY certification_name
)
SELECT 
    qualification_type,
    qualification_name,
    total_unique_employees
FROM (
    SELECT * FROM skills
    UNION ALL
    SELECT * FROM certifications
) AS combined
ORDER BY qualification_type ASC, qualification_name ASC;


-- ============================================
-- Lösung 3: Mit zusätzlichen Statistiken
-- ============================================
SELECT 
    qualification_type,
    qualification_name,
    COUNT(DISTINCT employee_id) AS total_unique_employees,
    ROUND(COUNT(DISTINCT employee_id) * 100.0 / 
          (SELECT COUNT(DISTINCT employee_id) FROM (
              SELECT employee_id FROM employee_skills
              UNION
              SELECT employee_id FROM employee_certifications
          ) AS all_employees), 2) AS percentage_of_employees
FROM (
    SELECT 
        'Skill' AS qualification_type,
        skill_name AS qualification_name,
        employee_id
    FROM employee_skills
    
    UNION ALL
    
    SELECT 
        'Certification' AS qualification_type,
        certification_name AS qualification_name,
        employee_id
    FROM employee_certifications
) AS combined_qualifications
GROUP BY qualification_type, qualification_name
ORDER BY qualification_type ASC, qualification_name ASC;


-- ============================================
-- Test-Daten
-- ============================================
/*
CREATE TABLE employee_skills (
    employee_id INT,
    skill_name VARCHAR(100)
);

CREATE TABLE employee_certifications (
    employee_id INT,
    certification_name VARCHAR(100)
);

INSERT INTO employee_skills VALUES
(1, 'Python'),
(1, 'SQL'),
(2, 'Python'),
(2, 'Java'),
(3, 'SQL'),
(4, 'Python');

INSERT INTO employee_certifications VALUES
(1, 'AWS Certified'),
(2, 'AWS Certified'),
(2, 'Azure Certified'),
(3, 'AWS Certified'),
(5, 'PMP');

-- Erwartetes Ergebnis:
-- qualification_type | qualification_name | total_unique_employees
-- -------------------|--------------------|-----------------------
-- Certification      | AWS Certified      | 3
-- Certification      | Azure Certified    | 1
-- Certification      | PMP                | 1
-- Skill              | Java               | 1
-- Skill              | Python             | 3
-- Skill              | SQL                | 2
*/


-- ============================================
-- Erklärung
-- ============================================
/*
1. UNION ALL kombiniert Skills und Certifications
   - 'Skill' als qualification_type für Skills
   - 'Certification' als qualification_type für Certifications

2. GROUP BY qualification_type, qualification_name
   - Gruppiert nach Typ und Name

3. COUNT(DISTINCT employee_id)
   - Zählt eindeutige Mitarbeiter pro Qualifikation

4. ORDER BY qualification_type ASC, qualification_name ASC
   - Sortiert erst nach Typ (Certification vor Skill alphabetisch)
   - Dann nach Name innerhalb des Typs

WICHTIG: UNION ALL (nicht UNION) verwenden, um Duplikate zu behalten
         für korrekte Zählung mit COUNT(DISTINCT)
*/
