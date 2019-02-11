#!/bin/bash

#SBATCH --job-name=diagnosis                    
#SBATCH --nodes=1                                 
#SBATCH --tasks-per-node=1                        
#SBATCH --cpus-per-task=1                         
#SBATCH --mem=40Gb                               
#SBATCH --time=01:00:00                            
#SBATCH --output=diagnosis.%j.out               
#SBATCH --error=diagnosis.%j.err                
#SBATCH --partition=general
#No USE SBATCH --gres=gpu:1
#No USE SBATCH --constraint="E5-2690v3@2.60GHz"

srun python Diagnosis.py --data_dir="/scratch/jain.sar/PatientVec"
