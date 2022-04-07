#!/bin/bash -l
#SBATCH -p cascade.p
#SBATCH -J 5Magb_q0.25
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elias.olofsson@h-its.org
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --nodes=20
#SBATCH --gres=cpuonly
#SBATCH -B 1:10:1
#SBATCH --threads-per-core=1
#SBATCH --time 24:00:00


source /etc/profile.d/z00_lmod.sh  
module use /hits/sw/pso/modules/
module use /hits/sw/tap/modules/
 
module load GCC
module load OpenMPI
module load pso_gsl
module load pso_gmp
module load pso_hdf5
module load pso_hwloc
module load pso_fftw

#mpirun --bind-to core ../arepo-cascade-2/Arepo params 2 151
if [ -d "output/restartfiles" ];
then
  # restart Arepo
  mpirun --bind-to core ../arepo/Arepo params 1
else
  mpirun --bind-to core ../arepo/Arepo params
fi

# restart runs
if [ -f "output/cont" ];
then
  sbatch 5Magb-run.cmd
fi
