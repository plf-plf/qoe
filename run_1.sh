#!/bin/bash
GPUS=${1}
CUDA_VISIBLE_DEVICES=${GPUS}
export CUBLAS_WORKSPACE_CONFIG=:16:8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
job_name="online"
current_datetime=$(date +"%Y-%m-%d_%H-%M-%S")
log_dir="/opt/data/private/ljx/plf/qos_mi/log/out" 
log_file=${log_dir}/${current_datetime}_${job_name}.log
nohup python run_online_test.py > ${log_file} 2>&1 &