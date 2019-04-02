WITH needed_icu as (
	SELECT subject_id, hadm_id, icustay_id
	from mimiciii.first_icu_notes
	group by subject_id, hadm_id, icustay_id
)
select det.subject_id, det.hadm_id, det.icustay_id, det.gender, det.ethnicity, det.admission_age, det.hospital_expire_flag, 
	icu.los, icu.first_careunit, icu.last_careunit, icu.first_wardid, icu.last_wardid, 
	adm.admission_type,admission_location,discharge_location,insurance, language, marital_status,diagnosis
from mimiciii.icustay_detail as det, mimiciii.icustays as icu, needed_icu as need, mimiciii.admissions as adm
where 
det.subject_id = icu.subject_id and det.hadm_id = icu.hadm_id and det.icustay_id = icu.icustay_id
and det.subject_id = need.subject_id and det.hadm_id = need.hadm_id and det.icustay_id = need.icustay_id 
and det.subject_id = adm.subject_id and det.hadm_id = adm.hadm_id;