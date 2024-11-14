#!/bin/bash
#SBATCH -t 3:00:00                    # time limit set
#SBATCH --array=1-47
#SBATCH --mem=76G                         # reserve memory
#SBATCH -J MCF_SMALL_DS_GNN                      # the job name
#SBATCH --mail-type=END,FAIL,TIME_LIMIT  # send notification emails
#SBATCH -n 1                             # use 1 tasks
#SBATCH --cpus-per-task=2                # use 8 thread per taks
#SBATCH -N 1                             # request slots on 1 node
#SBATCH --gpus=V100:1                    # request 1 Volta V100 GPU
#SBATCH --partition=gpuidle
#SBATCH --output=/scratch/vdwarth/large_ds_mcf/train/result/skip_f_loss_mix_post_hpo_%a/posthpo_%j_out.txt         # capture output
#SBATCH --error=/scratch/vdwarth/large_ds_mcf/train/result/skip_f_loss_mix_post_hpo_%a/posthpo_%j_err.txt          # and error streams

module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh

conda activate cuda
cd src/torch_code
python torch_post_hpo_trainer.py -dsroottrain $SCRATCH/large_ds_mcf/train -dsroottest $SCRATCH/large_ds_mcf/test -cuda 0 -compile_model True -experiment_name skip_f_loss_mix_post_hpo_$SLURM_ARRAY_TASK_ID -ntrain $SLURM_ARRAY_TASK_ID "$@"
conda deactivate
