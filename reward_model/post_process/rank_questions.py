import os
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--score-path', type=str, required=True)
parser.add_argument('--ann-path', type=str, required=True)
parser.add_argument('--output-path', type=str, required=True)
parser.add_argument('--keep-ratio', type=float, default=0.3)
args = parser.parse_args()

os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

scores = []
with open(args.score_path, 'r') as f:
    for line in f:
        example = json.loads(line)
        example.pop('questions')
        scores.append(example)

while len(scores) > 0:
    repeat = False
    for data in scores[:-1]:
        if scores[-1]['image'] == data['image']:
            repeat = True
            break
    if repeat:
        scores.pop()
    else:
        break
print('Valid scores:', len(scores))
scores.sort(key=lambda x: x['score'], reverse=True)
keep_images = {example['image']: example['score'] for example in scores[:int(args.keep_ratio*len(scores))]}

anns = json.load(open(args.ann_path, 'r'))
anns_dict = {}
for example in anns:
    if example['image'] in anns_dict:
        print(example['id'], example['image'])
    else:
        anns_dict[example['image']] = example
keep_anns = []
for image, score in tqdm(keep_images.items()):
    keep_anns.append(anns_dict[image])
    keep_anns[-1]['score'] = score

print(len(keep_anns))
with open(args.output_path, 'w') as f:
    json.dump(keep_anns, f, indent=2)