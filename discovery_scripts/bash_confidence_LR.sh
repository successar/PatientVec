for dataset in readmission mortality_1yr mortality_30day hip_1yr knee_1yr diagnosis 
    do
        dataset=$dataset sbatch ~/PatientVec/discovery_scripts/confidence_intervals_LR.sh
    done