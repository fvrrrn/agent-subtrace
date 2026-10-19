---
library_name: transformers
license: apache-2.0
base_model: answerdotai/ModernBERT-base
base_model_relation: finetune
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: ModernBERT-base-zeroshot-v2.0
  results: []
---


# ModernBERT-base-zeroshot-v2.0

## Model description

This model is [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) 
fine-tuned on the same dataset mix as the `zeroshot-v2.0` models in the [Zeroshot Classifiers Collection](https://huggingface.co/collections/MoritzLaurer/zeroshot-classifiers-6548b4ff407bb19ff5c3ad6f).


## General takeaways: 
- The model is very fast and memory efficient. It's multiple times faster and consumes multiple times less memory than DeBERTav3.
The memory efficiency enables larger batch sizes. I got a ~2x speed increase by enabling bf16 (instead of fp16).
- It performs slightly worse then DeBERTav3 on average on the tasks tested below.
- I'm in the process of preparing a newer version trained on better synthetic data to make full use of the 8k context window
and to update the training mix of the older `zeroshot-v2.0` models.


## Training results

Per-dataset breakdown:

|Datasets|Mean|Mean w/o NLI|mnli_m|mnli_mm|fevernli|anli_r1|anli_r2|anli_r3|wanli|lingnli|wellformedquery|rottentomatoes|amazonpolarity|imdb|yelpreviews|hatexplain|massive|banking77|emotiondair|emocontext|empathetic|agnews|yahootopics|biasframes_sex|biasframes_offensive|biasframes_intent|financialphrasebank|appreviews|hateoffensive|trueteacher|spam|wikitoxic_toxicaggregated|wikitoxic_obscene|wikitoxic_identityhate|wikitoxic_threat|wikitoxic_insult|manifesto|capsotu|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|Accuracy|0.831|0.835|0.932|0.936|0.884|0.763|0.647|0.657|0.823|0.889|0.753|0.864|0.949|0.935|0.974|0.798|0.788|0.727|0.789|0.793|0.489|0.893|0.717|0.927|0.851|0.859|0.907|0.952|0.926|0.726|0.978|0.912|0.914|0.93|0.951|0.906|0.476|0.708|
|F1 macro|0.813|0.818|0.925|0.93|0.872|0.74|0.61|0.611|0.81|0.874|0.751|0.864|0.949|0.935|0.974|0.751|0.738|0.746|0.733|0.798|0.475|0.893|0.712|0.919|0.851|0.859|0.892|0.952|0.847|0.721|0.966|0.912|0.914|0.93|0.942|0.906|0.329|0.637|
|Inference text/sec (A100 40GB GPU, batch=128)|3472.0|3474.0|2338.0|4416.0|2993.0|2959.0|2904.0|3003.0|4647.0|4486.0|5032.0|4354.0|2466.0|1140.0|1582.0|4392.0|5446.0|5296.0|4904.0|4787.0|2251.0|4042.0|1884.0|4048.0|4032.0|4121.0|4275.0|3746.0|4485.0|1114.0|4322.0|2260.0|2274.0|2189.0|2085.0|2410.0|3933.0|4388.0|


## Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 32
- eval_batch_size: 128
- seed: 42
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.06
- num_epochs: 2


## Framework versions

- Transformers 4.48.0.dev0
- Pytorch 2.5.1+cu124
- Datasets 3.2.0
- Tokenizers 0.21.0
