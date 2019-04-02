#!/bin/bash

#SBATCH --job-name=n_times                   
#SBATCH --nodes=1                                 
#SBATCH --tasks-per-node=1                        
#SBATCH --cpus-per-task=1                         
#SBATCH --mem=40Gb                               
#SBATCH --output=n_times_result.%j.out               
#SBATCH --error=n_times_result.%j.err                
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL    
#SBATCH --mail-user=successar@gmail.com
#SBATCH --constraint="E5-2690v3@2.60GHz"

srun python /home/jain.sar/PatientVec/"Discovery Experiments"/n_times.py --data_dir="/scratch/jain.sar/PatientVec" --output_dir="/scratch/jain.sar/PatientVec/outputs" --n=$n 
