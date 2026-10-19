---
library_name: transformers
license: apache-2.0
base_model: answerdotai/ModernBERT-large
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: ModernBERT-large-zeroshot-v2.0
  results: []
---

# ModernBERT-base-zeroshot-v2.0

## Model description

This model is [answerdotai/ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large) 
fine-tuned on the same dataset mix as the `zeroshot-v2.0` models in the [Zeroshot Classifiers Collection](https://huggingface.co/collections/MoritzLaurer/zeroshot-classifiers-6548b4ff407bb19ff5c3ad6f).


## General takeaways: 
- The model is very fast and memory efficient. It's multiple times faster and consumes multiple times less memory than DeBERTav3.
The memory efficiency enables larger batch sizes. I got a ~2x speed increase by enabling bf16 (instead of fp16).
- It performs slightly worse then DeBERTav3 on average on the tasks tested below.
- I'm in the process of preparing a newer version trained on better synthetic data to make full use of the 8k context window
and to update the training mix of the older `zeroshot-v2.0` models.


### Training results

|Datasets|Mean|Mean w/o NLI|mnli_m|mnli_mm|fevernli|anli_r1|anli_r2|anli_r3|wanli|lingnli|wellformedquery|rottentomatoes|amazonpolarity|imdb|yelpreviews|hatexplain|massive|banking77|emotiondair|emocontext|empathetic|agnews|yahootopics|biasframes_sex|biasframes_offensive|biasframes_intent|financialphrasebank|appreviews|hateoffensive|trueteacher|spam|wikitoxic_toxicaggregated|wikitoxic_obscene|wikitoxic_identityhate|wikitoxic_threat|wikitoxic_insult|manifesto|capsotu|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|Accuracy|0.85|0.851|0.942|0.944|0.894|0.812|0.717|0.716|0.836|0.909|0.815|0.899|0.964|0.951|0.984|0.814|0.8|0.744|0.752|0.802|0.544|0.899|0.735|0.934|0.864|0.877|0.913|0.953|0.921|0.821|0.989|0.901|0.927|0.931|0.959|0.911|0.497|0.73|
|F1 macro|0.834|0.835|0.935|0.938|0.882|0.795|0.688|0.676|0.823|0.898|0.814|0.899|0.964|0.951|0.984|0.77|0.753|0.763|0.69|0.805|0.533|0.899|0.729|0.925|0.864|0.877|0.901|0.953|0.855|0.821|0.983|0.901|0.927|0.931|0.952|0.911|0.362|0.662|
|Inference text/sec (A100 40GB GPU, batch=32)|1116.0|1104.0|1039.0|1241.0|1138.0|1102.0|1124.0|1133.0|1251.0|1240.0|1263.0|1231.0|1054.0|559.0|795.0|1238.0|1312.0|1285.0|1273.0|1268.0|992.0|1222.0|894.0|1176.0|1194.0|1197.0|1206.0|1166.0|1227.0|541.0|1199.0|1045.0|1054.0|1020.0|1005.0|1063.0|1214.0|1220.0|


### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 9e-06
- train_batch_size: 16
- eval_batch_size: 32
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 32
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.06
- num_epochs: 2


### Framework versions

- Transformers 4.48.0.dev0
- Pytorch 2.5.1+cu124
- Datasets 3.2.0
- Tokenizers 0.21.0
