#!/bin/bash

#SBATCH --job-name=readmission_attention                    
#SBATCH --nodes=1                                 
#SBATCH --tasks-per-node=1                        
#SBATCH --cpus-per-task=1                         
#SBATCH --mem=40Gb                               
#SBATCH --output=readmission_attention_result.%j.out               
#SBATCH --error=readmission_attention_result.%j.err            
#SBATCH --mail-type=ALL    
#SBATCH --mail-user=successar@gmail.com
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k40m:1

srun python /home/jain.sar/PatientVec/Readmission_Attention_hyperparams.py --data_dir="/scratch/jain.sar/PatientVec" --output_dir="/scratch/jain.sar/PatientVec/outputs" --exps=$exps
