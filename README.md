module load gcc/10.3.0
module load python/3.10.4
module load numpy/1.22.3
module load scipy/1.8.0
module load pandas/1.4.2
module load matplotlib/3.5.3
module load lapack
module load openblas
module load openmpi

module load gcc
module load python
module load lapack
module load openblas
module load openmpi


mpirun -np 4 ./fea_petsc /home/kwyi/ners570_project/mycelium-fea-project/results/sim_20251122_155110
