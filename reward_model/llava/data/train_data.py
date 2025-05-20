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
from typing import Dict, Optional, Sequence, List, Union, Tuple
from PIL import Image

import torch
from torch.utils.data import Dataset
import transformers
import tokenizers

from llava import conversation as conversation_lib
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token
from .utils import *


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args):
    if data_args.data_type == "answer":
        dataset = AnswerDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    elif data_args.data_type == "question":
        dataset = QuestionDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    else:
        raise NotImplementedError()
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dataset, data_collator


class AnswerDataset(Dataset):
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        super(AnswerDataset, self).__init__()
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        self.data_pairs = []
        for ann in raw_data:
            assert len(ann['conversations']) % 2 == 0, "numbers of questions and answers not equal"
            for turn_index in range(0, len(ann['conversations']), 2):
                question = ann['conversations'][turn_index]['value']
                answers = ann['conversations'][turn_index+1]['value']
                scores = ann['conversations'][turn_index+1]['score']
                assert ann['conversations'][turn_index]['from'] == 'human' and ann['conversations'][turn_index+1]['from'] == 'gpt', "questions and answers do not match"
                assert len(answers) == len(scores), "numbers of answers and scores do not match"
                n_ans = len(answers)
                for i in range(n_ans):
                    for j in range(n_ans):
                        if scores[i] > scores[j]:
                            self.data_pairs.append({
                                'question': question,
                                'answer1': answers[i],
                                'answer2': answers[j],
                            })
                            if 'image' in ann:
                                self.data_pairs[-1]['image'] = ann['image']
                    
                    if random.random() < getattr(data_args, 'augment_rate', 0.):
                        self.data_pairs.append({
                            'question': question,
                            'answer1': answers[i],
                            'answer2': random_augmentation(answers[i]),
                        })
                        if 'image' in ann:
                            self.data_pairs[-1]['image'] = ann['image']
        
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.data_pairs)
    
    @property
    def lengths(self):
        length_list = []
        for pair in self.data_pairs:
            img_tokens = 128 if 'image' in pair else 0
            length_list.append(sum(len(v.split()) for k, v in pair.items() if k != 'image') + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for pair in self.data_pairs:
            cur_len = sum(len(v.split()) for k, v in pair.items() if k != 'image')
            cur_len = cur_len if 'image' in pair else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        source = self.data_pairs[i]
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
        
        input_ids_1 = preprocess_qa(self.tokenizer, self.data_args,
                                    source['question'], source['answer1'])
        input_ids_2 = preprocess_qa(self.tokenizer, self.data_args,
                                    source['question'], source['answer2'])
        data_dict = dict(input_ids_1=input_ids_1,
                         input_ids_2=input_ids_2)

        # image exist in the data
        if 'image' in source:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


class QuestionDataset(Dataset):
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        super(QuestionDataset, self).__init__()
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        self.data_pairs = []
        for ann in raw_data:
            questions = ann['questions']
            scores = ann['score']
            assert len(questions) == len(scores), "numbers of questions and scores do not match"
            n_qst = len(questions)
            for i in range(n_qst):
                for j in range(n_qst):
                    if scores[i] > scores[j]:
                        self.data_pairs.append({
                            'question1': questions[i],
                            'question2': questions[j],
                        })
                        if 'image' in ann:
                            self.data_pairs[-1]['image'] = ann['image']
                
                if random.random() < getattr(data_args, 'augment_rate', 0.):
                    self.data_pairs.append({
                        'question1': questions[i],
                        'question2': question_random_augmentation(questions[i]),
                    })
                    if 'image' in ann:
                        self.data_pairs[-1]['image'] = ann['image']
        
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.data_pairs)
    
    @property
    def lengths(self):
        length_list = []
        for pair in self.data_pairs:
            img_tokens = 128 if 'image' in pair else 0
            length_list.append(sum(sum(len(q.split()) for q in v) for k, v in pair.items() if k != 'image') + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for pair in self.data_pairs:
            cur_len = sum(sum(len(q.split()) for q in v) for k, v in pair.items() if k != 'image')
            cur_len = cur_len if 'image' in pair else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        source = self.data_pairs[i]
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
        
        input_ids_1 = preprocess_qa(self.tokenizer, self.data_args, source['question1'])
        input_ids_2 = preprocess_qa(self.tokenizer, self.data_args, source['question2'])
        data_dict = dict(input_ids_1=input_ids_1,
                         input_ids_2=input_ids_2)

        # image exist in the data
        if 'image' in source:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


class DataCollatorForSupervisedDataset(DataCollatorMeta):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_1, input_ids_2 = tuple([instance[key] for instance in instances]
                                         for key in ("input_ids_1", "input_ids_2"))
        input_ids = input_ids_1 + input_ids_2

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances] * 2
        else:
            images = None
        
        return self._batchify(input_ids, images)