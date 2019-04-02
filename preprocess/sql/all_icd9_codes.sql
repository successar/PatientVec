SELECT subject_id, hadm_id, string_agg(icd9_code, ';' order by seq_num) as icd9_codes 
from mimiciii.diagnoses_icd 
group by subject_id, hadm_id 
order by subject_id, hadm_id;