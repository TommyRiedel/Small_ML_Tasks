-- ============================================
-- Employee Qualifications - Flexible Version
-- ============================================

-- VERSION 1: Wenn Spalten 'name' heißen
-- ============================================
SELECT 
    qualification_type,
    qualification_name,
    COUNT(DISTINCT employee_id) AS total_unique_employees
FROM (
    SELECT 
        'Skill' AS qualification_type,
        name AS qualification_name,
        employee_id
    FROM employee_skills
    
    UNION ALL
    
    SELECT 
        'Certification' AS qualification_type,
        name AS qualification_name,
        employee_id
    FROM employee_certifications
) AS combined
GROUP BY qualification_type, qualification_name
ORDER BY qualification_type ASC, qualification_name ASC;


-- VERSION 2: Wenn Spalten 'skill_name' und 'certification_name' heißen
-- ============================================
SELECT 
    qualification_type,
    qualification_name,
    COUNT(DISTINCT employee_id) AS total_unique_employees
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
) AS combined
GROUP BY qualification_type, qualification_name
ORDER BY qualification_type ASC, qualification_name ASC;


-- VERSION 3: Wenn Spalten 'skill' und 'certificate' heißen
-- ============================================
SELECT 
    qualification_type,
    qualification_name,
    COUNT(DISTINCT employee_id) AS total_unique_employees
FROM (
    SELECT 
        'Skill' AS qualification_type,
        skill AS qualification_name,
        employee_id
    FROM employee_skills
    
    UNION ALL
    
    SELECT 
        'Certification' AS qualification_type,
        certificate AS qualification_name,
        employee_id
    FROM employee_certifications
) AS combined
GROUP BY qualification_type, qualification_name
ORDER BY qualification_type ASC, qualification_name ASC;


-- ============================================
-- Finde heraus, welche Spalten existieren:
-- ============================================
SHOW COLUMNS FROM employee_skills;
SHOW COLUMNS FROM employee_certifications;

-- Oder:
DESCRIBE employee_skills;
DESCRIBE employee_certifications;

-- Oder:
SELECT * FROM employee_skills LIMIT 1;
SELECT * FROM employee_certifications LIMIT 1;
