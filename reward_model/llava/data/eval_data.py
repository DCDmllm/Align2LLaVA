# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import random
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple, Union
from PIL import Image

import torch
from torch.utils.data import Dataset
import transformers
import tokenizers

from llava import conversation as conversation_lib
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token
from .utils import *


def make_eval_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args):
    if data_args.data_type == "answer":
        dataset = AnswerEvalDataset(tokenizer=tokenizer,
                                    data_path=data_args.data_path,
                                    data_args=data_args)
    elif data_args.data_type == "question":
        dataset = QuestionEvalDataset(tokenizer=tokenizer,
                                      data_path=data_args.data_path,
                                      data_args=data_args)
    else:
        raise NotImplementedError()
    data_collator = DataCollatorForEvalDataset(tokenizer=tokenizer)
    return dataset, data_collator


class AnswerEvalDataset(Dataset):
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        super(AnswerEvalDataset, self).__init__()
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        self.data = []
        self.group_start_pos = []
        for ann in raw_data:
            assert len(ann['conversations']) % 2 == 0, "numbers of questions and answers not equal"
            for turn_index in range(0, len(ann['conversations']), 2):
                qst_turn = ann['conversations'][turn_index]
                ans_turn = ann['conversations'][turn_index+1]
                assert qst_turn['from'] == 'human' and ans_turn['from'] == 'gpt', "questions and answers do not match"

                question = qst_turn['value']
                answers = ans_turn['value']
                if not isinstance(answers, list):
                    answers = [answers]
                if 'score' in ans_turn:
                    scores = ans_turn['score']
                    assert len(answers) == len(scores), "numbers of answers and scores do not match"
                n_ans = len(answers)
                self.group_start_pos.append(len(self.data))
                for i in range(n_ans):
                    new_qa = {
                        'id': len(self.data),
                        'question': question,
                        'answer': answers[i],
                    }
                    self.data.append(new_qa)
                    if 'image' in ann:
                        self.data[-1]['image'] = ann['image']
                    if 'score' in ans_turn:
                        self.data[-1]['score'] = scores[i]
        self.group_start_pos.append(len(self.data))
        
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        source = self.data[i]
        if 'image' in source:
            image_file = source['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        input_ids = preprocess_qa(self.tokenizer, self.data_args,
                                  source['question'], source['answer'])
        data_dict = dict(id=source['id'],
                         input_ids=input_ids)

        # image exist in the data
        if 'image' in source:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


class QuestionEvalDataset(Dataset):
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        super(QuestionEvalDataset, self).__init__()
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        self.data = []
        self.group_start_pos = []
        for ann in raw_data:
            assert ('questions' in ann) ^ ('conversations' in ann)
            if 'questions' in ann:
                questions = ann['questions']
            elif 'conversations' in ann:  # for llava-format data (single group for each image)
                questions = [[turn['value'] for turn in ann['conversations'] if turn['from'] == 'human']]
            if 'score' in ann:
                scores = ann['score']
                assert len(questions) == len(scores), "numbers of questions and scores do not match"
            n_qst = len(questions)
            self.group_start_pos.append(len(self.data))
            for i in range(n_qst):
                new_qst = {
                    'id': len(self.data),
                    'question': questions[i],
                }
                self.data.append(new_qst)
                if 'image' in ann:
                    self.data[-1]['image'] = ann['image']
                if 'score' in ann:
                    self.data[-1]['score'] = scores[i]
        self.group_start_pos.append(len(self.data))
        
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        source = self.data[i]
        if 'image' in source:
            image_file = source['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        input_ids = preprocess_qa(self.tokenizer, self.data_args, source['question'])
        data_dict = dict(id=source['id'],
                         input_ids=input_ids)

        # image exist in the data
        if 'image' in source:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


class DataCollatorForEvalDataset(DataCollatorMeta):
    def __call__(self, instances: Sequence[Dict]):
        ids, input_ids = tuple([instance[key] for instance in instances]
                                for key in ("id", "input_ids"))
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
        else:
            images = None
        
        batch = self._batchify(input_ids, images)

        return ids, batch