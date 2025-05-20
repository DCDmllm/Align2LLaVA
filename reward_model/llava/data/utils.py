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
from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, List, Union
from PIL import Image

import torch
from torch.utils.data import Dataset
import transformers
import tokenizers

from llava import conversation as conversation_lib
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token


@dataclass
class DataArguments:
    data_type: str = field(default="answer",
                           metadata={"help": '"answer" or "question"'})
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    augment_rate: float = field(default=0.,
                                metadata={"help": "Possibility of adding an augmentation pair for each candidate. Range in [0, 1]."})


def preprocess_qa(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    question: Union[str, List[str]],
    answer: Optional[str] = None,
) -> torch.Tensor:
    is_multimodal = data_args.is_multimodal
    
    if answer:  # for answer dataset
        assert isinstance(question, str), "Multiple questions in one turn"
        question = question.replace(DEFAULT_IMAGE_TOKEN, '').strip()
        prompt = f"Question: {question}\n"
        answer = answer.replace(DEFAULT_IMAGE_TOKEN, '').strip()
        prompt += f"Answer: {answer}\n"
    else:  # for question dataset
        if isinstance(question, str):
            question = [question]
        for i in range(len(question)):
            question[i] = question[i].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        prompt = '\n'.join(question)
    if not is_multimodal:
        return prompt

    prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    prompt = prompt.strip()
    if "mmtag" in conversation_lib.default_conversation.version:
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
    replace_token = DEFAULT_IMAGE_TOKEN
    if data_args.mm_use_im_start_end:
        replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
    prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    # tokenize conversations
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors='pt')

    return input_ids


def random_augmentation(text, aug_mode=None):
    words = text.strip().split(' ')
    if aug_mode is None:
        aug_mode = random.randint(1, 2)
    if aug_mode == 1:  # random permutation
        random.shuffle(words)
    elif aug_mode == 2:  # random cropping
        index = random.sample(list(range(len(words))), random.randint(1, len(words)))
        index.sort()
        words = [words[i] for i in index]
    return ' '.join(words)


def question_random_augmentation(questions, aug_mode=None):
    res = []
    if aug_mode is None:
        aug_mode = random.randint(1, 3)

    if aug_mode == 3:  # duplication
        duplicate_index = random.sample(list(range(len(questions))), random.randint(1, len(questions)))
        total_len = len(questions) + len(duplicate_index)
        insert_pos = random.sample(list(range(total_len)), len(duplicate_index))
        insert_pos.sort()
        duplicate_num = 0
        for i in range(total_len):
            if duplicate_num < len(insert_pos) and i == insert_pos[duplicate_num]:
                res.append(questions[duplicate_index[duplicate_num]])
                duplicate_num += 1
            else:
                res.append(questions[i-duplicate_num])
    else:
        for text in questions:
            aug_text = random_augmentation(text, aug_mode)
            res.append(' '.join(aug_text))

    return res


@dataclass
class DataCollatorMeta(ABC):
    tokenizer: transformers.PreTrainedTokenizer

    def _batchify(self, input_ids: List[torch.Tensor],
                  images: List[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if images:
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        
        return batch
    
    @abstractmethod
    def __call__(self, instances):
        pass