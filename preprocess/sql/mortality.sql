DROP VIEW IF EXISTS mimiciii.mortality CASCADE;
CREATE VIEW mimiciii.mortality as
    SELECT pat.subject_id, adm.hadm_id, adm.admittime, adm.dischtime, 
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
 
DROP TABLE IF EXISTS mimiciii.structured_mortality CASCADE;
CREATE TABLE mimiciii.structured_mortality as 
WITH firstrow as (
SELECT * from mimiciii.mortality where RowNum = 1
),
sapsvals as (
select sapsii.*, intime, outtime, ROW_NUMBER() over (partition by sapsii.subject_id, sapsii.hadm_id order by intime) as RowNum
from mimiciii.sapsii as sapsii, mimiciii.icustays as icu
where sapsii.icustay_id = icu.icustay_id
),
saps_range as (
select subject_id, hadm_id, min(sapsii) as sapsii_min, max(sapsii) as sapsii_max
from sapsvals
group by subject_id, hadm_id
),
saps_admit as (
select subject_id, hadm_id, sapsii as sapsii_admit 
from sapsvals
where RowNum = 1
)
(select elix.*, sapsii_min, sapsii_max, sapsii_admit, firstrow.Mortality_30day, firstrow.admittime, firstrow.dischtime, firstrow.Mortality_1yr, firstrow.Mortality_In_Hospital
from mimiciii.elixhauser_ahrq_no_drg_all_icd as elix, firstrow, saps_range, saps_admit
where saps_range.subject_id = elix.subject_id and saps_range.hadm_id = elix.hadm_id 
 and saps_admit.subject_id = elix.subject_id and saps_admit.hadm_id = elix.hadm_id 
and firstrow.hadm_id = elix.hadm_id);