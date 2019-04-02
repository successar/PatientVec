SELECT adm.subject_id, adm.hadm_id, date_part('year', age(adm.admittime::date, pat.dob::date)) as age, pat.gender, adm.ethnicity
from mimiciii.admissions as adm , mimiciii.patients as pat 
where 
    adm.subject_id = pat.subject_id 
	AND lower(diagnosis) NOT LIKE '%organ donor%' 
	AND date_part('year', age(adm.admittime::date, pat.dob::date)) >= 1
order by adm.subject_id, adm.hadm_id;