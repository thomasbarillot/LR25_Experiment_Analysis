#!/bin/bash
for ((c=$1; c<=$2; c++))
do
  bsub -q psanaq -n $3 -o ./Log_XTCExporter/job%J_run$1.log mpirun python XTCExporter.py exp=amolr2516:run=$c
done
