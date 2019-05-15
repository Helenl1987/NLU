#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -t50:00:00
#SBATCH --mem=20GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jl10005@nyu.edu
#SBATCH --job-name=run-bert
#SBATCH --gres=gpu:p40:1
#SBATCH --output=slurm_%j.out

module purge
module load python3/intel/3.6.3 
source ~/pyenv/py3.6.3/bin/activate

#python train_man_exp3.py --dataset amazon-lang --model lstm --max_epoch 20
# python train_man_bert.py --dataset amazon-lang --model dan --max_epoch 14 --model_save_file ./save/train_bert_shared --topic_domain all --domains de en fr --no_wgan_trick --sum_pooling --shared_hidden_size 768 --shared --batch_size 16
# python train_man_bert.py --dataset amazon-lang --model dan --max_epoch 14 --model_save_file ./save/train_bert_shared_man --topic_domain all --domains de en fr --no_wgan_trick --sum_pooling --shared_hidden_size 768 --shared_man --batch_size 16
# python train_man_bert.py --dataset amazon-lang --model dan --max_epoch 14 --model_save_file ./save/train_bert_private --topic_domain all --domains de en fr --no_wgan_trick --sum_pooling --shared_hidden_size 768 --private --batch_size 16
python train_man_bert.py --dataset amazon-lang --model dan --max_epoch 14 --model_save_file ./save/train_bert_shared_private_man --topic_domain all --domains de en fr --no_wgan_trick --sum_pooling --shared_hidden_size 768 --shared_private_man --batch_size 16
