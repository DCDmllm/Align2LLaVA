#!/bin/bash

main_process_port=16828
batch_size=5
keep_ratio=0.3

repeat_times=1
image_folder=./datasets/coco_train2017
run_name=filtered_0.3_0.3
task_name=conversation  # [complex_reasoning, detail]
scoring_output_dir=./scoring_outputs
result_output_dir=./filtered_instruction

data_path=${result_output_dir}/${run_name}/stage_1/${task_name}.json

export CUDA_VISIBLE_DEVICES=0,1


accelerate launch --main_process_port ${main_process_port} eval.py \
    --model-path ./checkpoints/llava-v1.5-7b-lora-answer \
    --model-base liuhaotian/llava-v1.5-7b \
    --data-path ${data_path} \
    --data-type answer \
    --image-folder ${image_folder} \
    --output-path ${scoring_output_dir}/${run_name}_answer/${task_name}.jsonl \
    --batch-size ${batch_size} \
    --inference-only

python post_process/rank_answer.py \
    --score-path ${scoring_output_dir}/${run_name}_answer/${task_name}.jsonl \
    --ann-path ${data_path} \
    --output-path ${result_output_dir}/${run_name}/stage_2/${task_name}.json \
    --keep-ratio ${keep_ratio}