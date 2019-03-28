#!/bin/bash

#SBATCH --job-name=constant_LR                   
#SBATCH --nodes=1                                 
#SBATCH --tasks-per-node=1                        
#SBATCH --cpus-per-task=8                         
#SBATCH --mem=60Gb     
#SBATCH --output=constant_LR.%j.out               
#SBATCH --error=constant_LR.%j.err                
#SBATCH --partition=infiniband
#SBATCH --mail-type=ALL    
#SBATCH --mail-user=successar@gmail.com

srun python /home/jain.sar/PatientVec/"Discovery Experiments"/constant_LR.py --data_dir="/scratch/jain.sar/PatientVec" --output_dir="/scratch/jain.sar/PatientVec/outputs" --constant=$constant
