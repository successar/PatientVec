DROP VIEW IF EXISTS mimiciii.mortality CASCADE;
CREATE VIEW mimiciii.mortality as
    SELECT pat.subject_id, adm.hadm_id, 
            CASE WHEN dod < dischtime + interval '30' day THEN 1 ELSE 0 END as Mortality_30day,
            CASE WHEN dod < dischtime + interval '1' year THEN 1 ELSE 0 END as Mortality_1yr,
            CASE WHEN adm.deathtime IS NOT NULL THEN 1 ELSE 0 END as Mortality_In_Hospital,
            ROW_NUMBER() over (partition by pat.subject_id order by admittime) as RowNum
    FROM mimiciii.admissions adm
    INNER JOIN mimiciii.patients pat
    ON adm.subject_id = pat.subject_id
    WHERE lower(diagnosis) NOT LIKE '%organ donor%'
	AND date_part('year', age(adm.admittime::date, pat.dob::date)) > 18
    AND HAS_CHARTEVENTS_DATA = 1;
    
SELECT * from mimiciii.mortality;