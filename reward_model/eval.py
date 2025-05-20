import argparse
import os
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import gather_object

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.data import make_eval_data_module


def eval_model(args):
    # Model
    disable_torch_init()
    args.is_multimodal = True
    accelerator = Accelerator()
    device = accelerator.device

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device=device)
    args.image_processor = image_processor

    if accelerator.is_main_process:
        output_path = os.path.expanduser(args.output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_file = open(output_path, "w")

    dataset, data_collator = make_eval_data_module(tokenizer, args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=data_collator)
    dataloader = accelerator.prepare(dataloader)
    data_iter = tqdm(dataloader) if accelerator.is_main_process else dataloader
    all_scores = []
    for ids, inputs in data_iter:
        def to_device_and_dtype(data):
            for k in data:
                data[k] = data[k].to(device)
                if data[k].dtype == torch.float32:
                    data[k] = data[k].half()
            return data
        inputs = to_device_and_dtype(inputs)
        with torch.inference_mode():
            outputs = model(**inputs)
        
        scores = outputs.logits.view(-1).cpu().tolist()
        gathered_scores = gather_object(list(zip(ids, scores)))
        if accelerator.is_main_process:
            for id, score in gathered_scores:
                data = dataset.data[id]
                if args.data_type == 'answer':
                    output_file.write(json.dumps({
                        "image": data['image'],
                        "question": data['question'],
                        "answer": data['answer'],
                        "score": score,
                    }) + "\n")
                elif args.data_type == 'question':
                    output_file.write(json.dumps({
                        "image": data['image'],
                        "questions": data['question'],
                        "score": score,
                    }) + "\n")
                else:
                    raise NotImplementedError()
            output_file.flush()
            if not args.inference_only:
                all_scores += gathered_scores
    
    if accelerator.is_main_process:
        output_file.close()

        if not args.inference_only:
            total, correct = 0, 0
            all_scores.sort(key= lambda x: x[0])
            for group_index in range(len(dataset.group_start_pos) - 1):
                start_pos = dataset.group_start_pos[group_index]
                end_pos = dataset.group_start_pos[group_index+1]
                for i in range(start_pos, end_pos):
                    for j in range(start_pos, end_pos):
                        if dataset.data[i]['score'] > dataset.data[j]['score']:
                            total += 1
                            id1, score1 = all_scores[i]
                            id2, score2 = all_scores[j]
                            assert i == id1 and j == id2
                            if score1 > score2:
                                correct += 1
            
            print(f'acc: {correct/total}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--data-type", type=str, default="answer", choices=["answer", "question"])
    parser.add_argument("--inference-only", action='store_true')
    parser.add_argument("--image-aspect-ratio", type=str, default='square')
    parser.add_argument("--mm-use-im-start-end", type=bool, default=False)
    parser.add_argument("--output-path", type=str, default="score.jsonl")
    parser.add_argument("--conv-mode", type=str, default="plain")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
