#!/bin/bash

#SBATCH --job-name=split_and_clean_mimic                    
#SBATCH --nodes=1                                 
#SBATCH --tasks-per-node=1                        
#SBATCH --cpus-per-task=1                         
#SBATCH --mem=20Gb                               
#SBATCH --output=split_and_clean_mimic_result.%j.out               
#SBATCH --error=split_and_clean_mimic_result.%j.err            
#SBATCH --mail-type=ALL    
#SBATCH --mail-user=successar@gmail.com

srun python /home/jain.sar/PatientVec/preprocess/common_data/split_and_clean_notes.py --notes_file="/scratch/jain.sar/PatientVec/NOTEEVENTS.csv" --output_dir="/scratch/jain.sar/PatientVec/noteevents_cleaned" --iter=$iter --chunk=50000
