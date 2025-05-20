# Reward Model

### Data Preparation

We have provided the human preference annotations at `datasets/preference`. To train the reward model, you need to download the [COCO 2017 train images](http://images.cocodataset.org/zips/train2017.zip), and create a link to the image directory at `datasets/coco_train2017`:

```
ln -s /path/to/train2017 datasets/coco_train2017
```

Additionally, to run reward model inference, you also need to prepare the instruction `.json` file organized as follows:

```
[
    {
        "id": "000000215677",
        "image": "000000215677.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "What skill set might someone need to perform such a frisbee trick?"
            },
            {
                "from": "gpt",
                "value": [
                    "To perform the frisbee trick shown in the image, a person would ...",
                    "To perform such a frisbee trick, an individual would ...",
                    "To perform such a frisbee trick, someone would ..."
                ]  # if there is only one answer, the value can also be a single string
            }
        ]
    },
    ...
]
```

By default, we recommend to organize the three types of instructions (complex reasoning, conversation, detail description) as three distinct files (`complex_reasoning.json`, `conversation.json`, `detail.json`).

### Training

For quick starting, run the scripts:

```
bash ./scripts/train_lora_question.sh   # train question reward model
bash ./scripts/train_lora_answer.sh  	# train answer reward model
```

The resulting checkpoints are saved under `checkpoints` by default.

### Model Zoo

We also provide trained reward models at [this link](https://huggingface.co/Huanghz/Align2LLaVA_RM).

### Evaluation

To run direct evaluation on human preference prediction, run the script:

```
bash ./scripts/eval.sh
```

Modify the script to switch between question prediction and answer prediction.

### Inference

To run reward model inference to filter instructions, first specify the path to the instruction file in [rank_questions.sh](scripts/rank_questions.sh#L8) with the `data_path` variable. Afterwards, run the two scripts in sequence:

```
bash ./scripts/rank_questions.sh   	# stage 1 filtration (question)
bash ./scripts/rank_answers.sh  	# stage 2 filtration (answer)
```

If you have run inference for all three instruction types (complex reasoning, conversation, detail description), we have also provided a script to merge them:

```
bash ./scripts/merge_final.sh
```
