#!/bin/bash
for ((c=$1; c<=$2; c++))
do
  bsub -q psanaq -n 128 -o %J_run$1.log mpirun python mpi_driver.py exp=amolr2516:run=$c
done
