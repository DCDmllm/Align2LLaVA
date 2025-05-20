import os
import json
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--result-dir', type=str, required=True)
parser.add_argument('--run-name', type=str, required=True)
parser.add_argument('--output-path', type=str, default=None)
args = parser.parse_args()

task_names = ['complex_reasoning', 'conversation', 'detail']
result_paths = [os.path.join(args.result_dir, args.run_name, 'stage_2', task)
                for task in task_names]
if args.output_path is None:
    args.output_path = os.path.join(args.result_dir, args.run_name, 'merged.json')

random.seed(1234)
merged_anns = []
for path in result_paths:
    anns = json.load(open(path, 'r'))
    print(path, len(anns))
    merged_anns += anns
for example in merged_anns:
    if 'score' in example:
        example.pop('score')

os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
with open(args.output_path, 'w') as f:
    json.dump(merged_anns, f, indent=2)
