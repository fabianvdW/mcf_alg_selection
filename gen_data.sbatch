#!/bin/bash
#SBATCH -t 0-12:00:00                    # time limit set 
#SBATCH --array=0-99
#SBATCH --mem=8G                         # reserve memory
#SBATCH -J Data_Generation_Array         # the job name
#SBATCH --mail-type=END,FAIL,TIME_LIMIT  # send notification emails
#SBATCH -n 1                             # use 1 tasks
#SBATCH --cpus-per-task=8                # use 8 thread per taks
#SBATCH -N 1                             # request slots on 1 node
#SBATCH --partition=cpuidle
#SBATCH --output=/scratch/vdwarth/gen_data/run%a/test_%j_%a.out        # capture output
#SBATCH --error=/scratch/vdwarth/gen_data/run%a/test%j_%a.err          # and error streams

module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh

conda activate cuda
mkdir $SCRATCH/gen_data/run$SLURM_ARRAY_TASK_ID
cp -r cs2 $SCRATCH/gen_data/run$SLURM_ARRAY_TASK_ID
cd $SCRATCH/gen_data/run$SLURM_ARRAY_TASK_ID/cs2
make
cd ..
touch data_commands.csv 
touch features.csv
touch runtimes.csv
cd ~/mcf_alg_selection/src/gen_data
python generate_features_runtimes_parallelized.py -n 8 -dsroot $SCRATCH/gen_data/run$SLURM_ARRAY_TASK_ID -cs2path $SCRATCH/gen_data/run$SLURM_ARRAY_TASK_ID/cs2
conda deactivate
