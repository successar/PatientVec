#!/bin/bash

#SBATCH --job-name=diagnosis_hyperparam                    
#SBATCH --nodes=1                                 
#SBATCH --tasks-per-node=1                        
#SBATCH --cpus-per-task=1                         
#SBATCH --mem=40Gb                               
#SBATCH --time=23:59:00                            
#SBATCH --output=diagnosis_hyperparam_result.%j.out               
#SBATCH --error=diagnosis_hyperparam_result.%j.err            
#SBATCH --mail-type=ALL    
#SBATCH --mail-user=successar@gmail.com
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k40m:1

srun python /home/jain.sar/PatientVec/Diagnosis_hyperparams.py --data_dir="/scratch/jain.sar/PatientVec" --output_dir="/scratch/jain.sar/PatientVec/outputs" --lr=$lr --reg=$reg --exps=$exps
