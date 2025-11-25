#!/bin/bash
#SBATCH --job-name=fea_solverAndPC
#SBATCH --account=ners570f25_class
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --output=../logs/fea_out_%j.txt
#SBATCH --error=../logs/fea_err_%j.txt

# Load MPI
module purge
module load gcc openmpi lapack openblas

# PETSC environment
export PETSC_DIR=/home/kwyi/petsc
export PETSC_ARCH=arch-linux-c-opt
export PATH=$PETSC_DIR/$PETSC_ARCH/bin:$PATH
export LD_LIBRARY_PATH=$PETSC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH

# Compile
make

# Run over geometry folder
mpirun -np 1 ./fea_petsc_solverAndPC.exe ../results/sim_20251117_181147_cpp
