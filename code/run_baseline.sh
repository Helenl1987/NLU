#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -t10:00:00
#SBATCH --mem=20GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jl10005@nyu.edu
#SBATCH --job-name=run-baseline
#SBATCH --gres=gpu:p40:1
#SBATCH --output=slurm_%j.out

module purge
module load python3/intel/3.6.3 
source ~/pyenv/py3.6.3/bin/activate

#python train_man_exp3.py --dataset amazon-lang --model lstm --max_epoch 20
python train_man_baselines.py --dataset amazon-lang --model dan --max_epoch 2 --model_save_file ./save/train_baselines_shared --topic_domain all --domains de en fr --no_wgan_trick --sum_pooling --shared
python train_man_baselines.py --dataset amazon-lang --model dan --max_epoch 2 --model_save_file ./save/train_baselines_shared_man --topic_domain all --domains de en fr --no_wgan_trick --sum_pooling --shared_man
python train_man_baselines.py --dataset amazon-lang --model dan --max_epoch 2 --model_save_file ./save/train_baselines_private --topic_domain all --domains de en fr --no_wgan_trick --sum_pooling --private
python train_man_baselines.py --dataset amazon-lang --model dan --max_epoch 2 --model_save_file ./save/train_baselines_shared_private_man --topic_domain all --domains de en fr --no_wgan_trick --sum_pooling --shared_private_man
