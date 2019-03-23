#!/bin/bash

#SBATCH --job-name=confidence_LR                   
#SBATCH --nodes=1                                 
#SBATCH --tasks-per-node=1                        
#SBATCH --cpus-per-task=8                         
#SBATCH --mem=60Gb     
#SBATCH --output=confidence_LR.%j.out               
#SBATCH --error=confidence_LR.%j.err                
#SBATCH --partition=general

srun python /home/jain.sar/PatientVec/"Discovery Experiments"/confidence_interval_LR.py --data_dir="/scratch/jain.sar/PatientVec" --output_dir="/scratch/jain.sar/PatientVec/outputs" 
