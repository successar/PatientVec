#!/bin/bash

#SBATCH --job-name=confidence                   
#SBATCH --nodes=1                                 
#SBATCH --tasks-per-node=1                        
#SBATCH --cpus-per-task=1                         
#SBATCH --mem=40Gb     
#SBATCH --time=10:00:00
#SBATCH --output=confidence.%j.out               
#SBATCH --error=confidence.%j.err                
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="E5-2690v3@2.60GHz"

srun python /home/jain.sar/PatientVec/"Discovery Experiments"/confidence_interval.py --data_dir="/scratch/jain.sar/PatientVec" --output_dir="/scratch/jain.sar/PatientVec/outputs" 
