#!/bin/bash

#SBATCH --job-name=run_model_lda                   
#SBATCH --nodes=1                                 
#SBATCH --tasks-per-node=1                        
#SBATCH --cpus-per-task=4                         
#SBATCH --mem=40Gb                               
#SBATCH --output=run_model_lda_result.%j.out               
#SBATCH --error=run_model_lda_result.%j.err
#SBATCH --mail-type=ALL    
#SBATCH --mail-user=successar@gmail.com
#SBATCH --partition=general

srun python /home/jain.sar/PatientVec/"Discovery Experiments"/LDA.py --dataset $dataset --data_dir="/scratch/jain.sar/PatientVec" --output_dir="/scratch/jain.sar/PatientVec/outputs" 
