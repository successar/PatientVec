
create view mimiciii.notes as 
select n.subject_id,n.hadm_id,n.charttime,n.category,n.text
from mimiciii.noteevents n
where iserror IS NULL
and category != 'Discharge summary'
and hadm_id IS NOT NULL
and charttime IS NOT NULL;