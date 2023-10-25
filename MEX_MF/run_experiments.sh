#!/bin/bash

# Script to reproduce results

policy="IO"
comment="IO"
#envs=(
#	"Ant-v3"
#  "HalfCheetah-v3"
#  "Walker2d-v3"
#  "Hopper-v3"
#)
envs=(
"walker-vel-sparse"
  "cheetah-vel-sparse"
 "walker-vel-sparse"
  "hopper-vel-sparse"
)
gpus=(0 1 2)
for ((j=0;j<${#envs[@]};j+=1))
do
  export CUDA_VISIBLE_DEVICES=${gpus[j]}
  env=${envs[j]}
  for ((i=0;i<5;i+=1))
  do
    nohup python main.py \
    --policy ${policy} \
    --env ${env} \
    --comment ${comment} \
    --eta 0.005 \
    --use_baseline \
    --sparse \
    --seed $i > nohup_logs/"${env}"_"${policy}"_"${comment}"_"${i}".out &
  done
done
