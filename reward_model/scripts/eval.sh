#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# question prediction
# accelerate launch --main_process_port 12464 eval.py \
#     --model-path ./checkpoints/llava-v1.5-7b-lora-question \
#     --model-base liuhaotian/llava-v1.5-7b \
#     --data-path ./datasets/preference/question/test.json \
#     --data-type question \
#     --image-folder ./datasets/coco_train2017 \
#     --output-path ./eval_outputs/scoring_question.jsonl \
#     --batch-size 6 

# answer prediction
accelerate launch --main_process_port 16399 eval.py \
    --model-path ./checkpoints/llava-v1.5-7b-lora-answer \
    --model-base liuhaotian/llava-v1.5-7b \
    --data-path ./datasets/preference/answer/test.json \
    --data-type answer \
    --image-folder ./datasets/coco_train2017 \
    --output-path ./eval_outputs/scoring_answer.jsonl \
    --batch-size 5