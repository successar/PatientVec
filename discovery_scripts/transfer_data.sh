for d in Mortality 
do
    scp -p -r preprocess/$d/Split_Structured_Final.msg jain.sar@xfer.discovery.neu.edu:/scratch/jain.sar/PatientVec/preprocess/$d/Split_Structured_Final.msg
    scp -p -r preprocess/$d/combined_notes_sequences.p jain.sar@xfer.discovery.neu.edu:/scratch/jain.sar/PatientVec/preprocess/$d/combined_notes_sequences.p
    scp -p -r preprocess/$d/combined_notes_vecs.p jain.sar@xfer.discovery.neu.edu:/scratch/jain.sar/PatientVec/preprocess/$d/combined_notes_vecs.p
    scp -p -r preprocess/$d/vocabulary.p jain.sar@xfer.discovery.neu.edu:/scratch/jain.sar/PatientVec/preprocess/$d/vocabulary.p
    scp -p -r preprocess/$d/embedding* jain.sar@xfer.discovery.neu.edu:/scratch/jain.sar/PatientVec/preprocess/$d
done