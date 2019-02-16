#!/bin/bash

for exps in vanilla attention structured hierarchical 
do
    for reg in 0.001 0.0001 0.000001
    do
        lr=0.001 reg=$reg exps=$exps sbatch /home/jain.sar/PatientVec/discovery_scripts/readmission_hyperparam.sh 
    done
done