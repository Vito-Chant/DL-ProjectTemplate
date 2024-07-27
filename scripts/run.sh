#!/bin/bash
fold_num=4
dataset_name="eyediap"
data_format="cont"
model="mobilegazenet_v2"
epoch=30
batch_size=64
seed=42
name=${dataset_name}"/"${model}"_bs_"${batch_size}
optim="adam_multistep
dataset="gazedataset"
better="small"
metric="angle_error"
disable_engin_plugin="all"
parallel_mode="dp"
#gpu id need to fix in the following part
for i in $(seq 0 $fold_num)
do
  python main.py --name ${name} --model ${model} --epoch ${epoch} --optim ${optim} \
  --dataset ${dataset} --disable_wandb --better ${better} --metric ${metric} \
  --disable_engine_plugin ${disable_engin_plugin} --parallel_mode ${parallel_mode} \
  --batch_size ${batch_size} --gpu_id 0 1 2 --dataset_fold $i --dataset_name ${dataset_name}\
  --seed ${seed}
done