#!/bin/bash
#SBATCH -t 0-08:00:00                    # time limit set 
#SBATCH --mem=64G                         # reserve memory
#SBATCH -J MCF_LARGE_DS_MERGE                    # the job name
#SBATCH --mail-type=END,FAIL,TIME_LIMIT  # send notification emails
#SBATCH -n 1                             # use 1 tasks
#SBATCH --cpus-per-task=2                # use 8 thread per taks
#SBATCH -N 1                             # request slots on 1 node
#SBATCH --partition=cpuidle
#SBATCH --output=test_%j_out.txt         # capture output
#SBATCH --error=test_%j_err.txt          # and error streams

module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh

conda activate cuda
cd src/torch_code
python torch_merger.py $SCRATCH/large_ds_mcf/
conda deactivate
