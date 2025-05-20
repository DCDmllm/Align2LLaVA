import os
import json
import argparse
from copy import deepcopy
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--score-path', type=str, required=True)
parser.add_argument('--ann-path', type=str, required=True)
parser.add_argument('--output-path', type=str, required=True)
parser.add_argument('--repeat-times', type=int, required=True)
parser.add_argument('--keep-ratio', type=float, default=0.3)
args = parser.parse_args()

os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

scores = []
with open(args.score_path, 'r') as f:
    for i, line in enumerate(f):
        scores.append(json.loads(line))
while len(scores) > 0:
    repeat = False
    group_index = -1
    while scores[group_index - 1]['image'] == scores[-1]['image']:
        group_index -= 1
    for data in scores[:group_index]:
        if scores[-1]['image'] == data['image']:
            repeat = True
            break
    if repeat:
        scores = scores[:group_index]
    else:
        break
print('Valid scores:', len(scores))

anns = json.load(open(args.ann_path))
conv_dict = {ann['image']: ann['conversations'] for ann in anns}

keep_anns = []
start = 0
pbar = tqdm(range(len(anns)))
while start < len(scores):
    pbar.set_postfix(start=start)
    image = scores[start]['image']
    conv = conv_dict[image]
    best_example = {
        'id': image.replace('.jpg', ''),
        'image': image,
        'conversations': deepcopy(conv),
        'score': 0.,
    }
    turn_num = len(conv) // 2
    for i in range(turn_num):
        turn_scores = []
        for j in range(args.repeat_times):
            scored_qa = scores[start + i * args.repeat_times + j]
            assert scored_qa['image'] == image
            assert scored_qa['question'] == conv[i*2]['value']
            turn_scores.append(scored_qa)
        turn_scores.sort(key=lambda x: x['score'], reverse=True)
        best_example['conversations'][i*2+1]['value'] = turn_scores[0]['answer']
        best_example['score'] += turn_scores[0]['score']
    best_example['score'] /= turn_num
    keep_anns.append(best_example)

    start += turn_num * args.repeat_times
    pbar.update()
pbar.close()

keep_anns.sort(key=lambda x: x['score'], reverse=True)
keep_anns = keep_anns[:int(len(keep_anns)*args.keep_ratio)]
with open(args.output_path, 'w') as f:
    json.dump(keep_anns, f, indent=2)