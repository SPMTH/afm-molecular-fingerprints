#!/bin/bash
#SBATCH --job-name=relax_ce_clean_wo_water
#SBATCH --chdir=.
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=32
#SBATCH -N 1
###SBATCH --nodelist=node-05
#SBATCH -p spmth
#SBATCH --time=3-00:00:00
#SBATCH --mem-per-cpu=3936
#SBATCH --get-user-env

source ~/.bashrc

export VASP_PP_PATH='/apps/exported/installed/software/vasp/PP/'


module load vasp/6.3.0 intel/oneAPI/conda

date > time.log
srun vasp_std  >> vasp.out
while [[ `tail -1 vasp.out` == '     to POSCAR and continue' ]]; do
  \cp CONTCAR POSCAR
  \mv CONTCAR CONTCAR-$(date "+%s")
   srun vasp_std  >> vasp.out
 done
 date >> time.log
