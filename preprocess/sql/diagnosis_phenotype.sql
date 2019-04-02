WITH tmp as (
SELECT * from mimiciii.icustays as icu
where icu.first_wardid = icu.last_wardid and icu.first_careunit = icu.last_careunit
),
grouped_hadm_id as (
SELECT hadm_id from tmp group by hadm_id having count(*) = 1 
)
SELECT tmp.*
from tmp, mimiciii.patients as pat
where pat.subject_id = tmp.subject_id
and date_part('year', age(tmp.intime::date, pat.dob::date)) >= 18
and hadm_id in (SELECT hadm_id from grouped_hadm_id)
order by subject_id, hadm_id, icustay_id;