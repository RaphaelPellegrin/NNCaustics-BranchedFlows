#!/bin/bash
#SBATCH -J cluster_Raphael
#SBATCH --gres=gpu:1
#SBATCH -p gpu_test                                     # partition name
#SBATCH -N 1                                        # node count required for job
#SBATCH --mem 96000                                 # memory request per node (MB)
#SBATCH -t 0-08:00                                  # time request (D-HH:MM)
#SBATCH --open-mode=append                          # append when writing files
#SBATCH -o logs_%j.out                              # standard output file with job ID
#SBATCH -e logs_%j.err                              # standard error file with job ID
#SBATCH --mail-type=END                             # email when job is finished
#SBATCH --mail-user=raphaelpellegrin@g.harvard.edu  # send to this email
module load gcc/10.2.0-fasrc01                      # something about torch plotting
module load Anaconda3/2020.11
source activate env_raphael
pip3 install torch
module load cuda/11.1.0-fasrc01
srun -c 1 --gres=gpu:1 python3 Cluster.py 



