#!/bin/bash

##SBATCH -p cnode01
#SBATCH -p gnode04
#SBATCH -n 1
#SBATCH -J ChemTSv2
#SBATCH -o log/stout.%J
#SBATCH --gres=gpu:1

module load slurm cuda/11.8
#export CUDA_VISIBLE_DEVICES=0,1,2,3

start_time=`date "+%Y-%m-%d %H:%M:%S"`

#python make_feature.py -c config/setting_feature.yaml
python make_model.py -c config/setting_model.yaml
#chemtsv2 -c config/setting_protacts.yaml --gpu 0 --use_gpu_only_reward

end_time=`date "+%Y-%m-%d %H:%M:%S"`

echo "start" $start_time
echo "end" $end_time
