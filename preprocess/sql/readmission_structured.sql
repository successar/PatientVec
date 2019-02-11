DROP VIEW IF EXISTS mimiciii.structured_readmission;
CREATE VIEW mimiciii.structured_readmission as 
select icu.subject_id, icu.hadm_id, icu.icustay_id, icu.readmission
from mimiciii.readmissions as icu
where readmission != 2
order by subject_id, hadm_id, icustay_id;

SELECT * FROM mimiciii.structured_readmission;