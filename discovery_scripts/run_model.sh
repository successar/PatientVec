#!/bin/bash

#SBATCH --job-name=run_model                   
#SBATCH --nodes=1                                 
#SBATCH --tasks-per-node=1                        
#SBATCH --cpus-per-task=1                         
#SBATCH --mem=40Gb                               
#SBATCH --output=run_model_result.%j.out               
#SBATCH --error=run_model_result.%j.err                
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="E5-2690v3@2.60GHz"

srun python /home/jain.sar/PatientVec/"Discovery Experiments"/run_models.py --dataset $dataset --exp_types $exp_types --data_dir="/scratch/jain.sar/PatientVec" --output_dir="/scratch/jain.sar/PatientVec/outputs" --structured 
