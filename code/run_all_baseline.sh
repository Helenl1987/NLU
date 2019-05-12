#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -t10:00:00
#SBATCH --mem=20GB
#SBATCH --mail-type=END
#SBATCH --mail-user=zl2521@nyu.edu
#SBATCH --job-name=dan_10epochs
#SBATCH --gres=gpu:p40:1
#SBATCH --output=slurm_%j.out

module purge
module load python3/intel/3.6.3 
source ~/pyenv/py3.6.3/bin/activate

#python train_man_exp3.py --dataset amazon-lang --model lstm --max_epoch 20

python train_shared.py --dataset amazon-lang --model dan --max_epoch 20 --model_save_file ./save/all_shared_dan --topic_domain all --domains fr en de

python train_shared_man.py --dataset amazon-lang --model dan --max_epoch 20 --model_save_file ./save/all_shared_man_dan --topic_domain all --domains fr en de

python train_private.py --dataset amazon-lang --model dan --max_epoch 20 --model_save_file ./save/all_private_dan --topic_domain all --domains fr en de

python train_man_exp3.py --dataset amazon-lang --model dnn --max_epoch 20 --model_save_file ./save/all_dan --topic_domain all --domains fr en de

