#!/bin/bash
#SBATCH -p debug
#SBATCH -n 1
#SBATCH-t :60
#SBATCH --mem 1000

.~/.profile
cd /lustre/zachg
module load python/3.2.3
python FockStateMinDep.py  -t .01 -dt 1e-6 -name ${NAME} -q ${Q} -n 4000
