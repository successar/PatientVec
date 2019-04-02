CREATE VIEW mimiciii.first_icu_query AS
select distinct i.subject_id, i.hadm_id,
i.icustay_id, i.intime, i.outtime, i.admittime, i.dischtime
  FROM mimiciii.icustay_detail i
  LEFT JOIN mimiciii.icustays s ON i.icustay_id = s.icustay_id
  WHERE s.first_careunit NOT like 'NICU'
  and i.hospstay_seq = 1
  and i.icustay_seq = 1
  and i.admission_age >= 15
  and i.los_icu >= 0.5
  and i.admittime <= i.intime
  and i.intime <= i.outtime
  and i.outtime <= i.dischtime;