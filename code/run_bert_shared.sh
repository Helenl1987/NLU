#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -t10:00:00
#SBATCH --mem=20GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zl2521@nyu.edu
#SBATCH --job-name=test_bert
#SBATCH --gres=gpu:p40:1
#SBATCH --output=slurm_%j.out

module purge
module load python3/intel/3.6.3 
source ~/pyenv/py3.6.3/bin/activate

#python train_man_exp3.py --dataset amazon-lang --model lstm --max_epoch 20
python train_man_bert.py --dataset amazon-lang --model dan --max_epoch 20 --model_save_file ./save/test_bert --topic_domain music --domains de en --unlabeled_domains fr --dev_domains fr --no_wgan_trick --sum_pooling --shared_hidden_size 768
