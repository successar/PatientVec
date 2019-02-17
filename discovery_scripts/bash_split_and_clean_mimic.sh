for i in {1..45}
do
    iter=$i sbatch /home/jain.sar/PatientVec/discovery_scripts/split_and_clean_notes.sh
done