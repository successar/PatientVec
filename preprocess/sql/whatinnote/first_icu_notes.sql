drop view if exists mimiciii.first_icu_notes ;
create view mimiciii.first_icu_notes as 
SELECT first_icu.*, notes.charttime, notes.category, notes.text
FROM mimiciii.first_icu_query as first_icu, mimiciii.notes as notes
where first_icu.subject_id = notes.subject_id 
and first_icu.hadm_id = notes.hadm_id
and first_icu.intime <= notes.charttime
and first_icu.outtime >= notes.charttime
and TRIM(notes.category) IN ('Radiology' , 'Nursing', 'Physician', 'Nursing/other');