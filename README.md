<div align='center'>
<h2>Embedding Enhancement via Fine-Tuned Language Models for Learner-Item Cognitive Modeling, WWW 2026</h2>
<a>Yuanhao Liu</a>,
<a>Zihan Zhou</a>,
<a>Kaiying Wu</a>,
<a>Shuo Liu</a>,
<a>Yiyang Huang</a>,
<a>Jiajun Guo</a>,
<a>Aimin Zhou</a> and
<a>Hong Qian*</a> (*Correspondence)

<a>East China Normal University</a><br>
<a>Shanghai Innovation Institute</a>

<a href='https://github.com/BW297/EduEmbed'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://github.com/BW297/EduEmbed/tree/master/paper/EduEmbed_WWW26.pdf'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a>


<br/>
<img src="figure/frame.png" width="700" alt="Framework Image" />

</div>

------

**EduEmbed** employs a **two-stage framework** to achieve embedding enhancement for Learner-Item Cognitive Modeling.


## 📌 Stage 1: Role-aware Interaction Fine-tuning (RaIF)

RaIF aims to fine-tune a language model with role-specific descriptions and generate textual embeddings for students, exercises, and concepts.

### 🛠 Implementation Workflow

Please navigate to the following directory:

```
cd stage1
```

------

#### 1. Preprocess Data

To generate personalized profiles for students, exercises, and concepts:

```bash
bash data_gene.sh
```

This step processes the dataset and creates the input for fine-tuning.

#### 2. Fine-tune the Language Model

To fine-tune the language model:

```bash
bash main.sh
```

#### 3. Generate Semantic Embeddings

To obtain textual embeddings for students, exercises, and concepts:

```bash
bash infer.sh
```

------

### ⚙️ Configuration Guide

Before running the scripts, please configure the following variables:

| Variable                 | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| `$Path_to_dataset$`      | Path to the processed dataset                                |
| `$Abbreviation_of_LM$`   | Abbreviation for the base model (e.g., `Qwen2.5-3B`)         |
| `$Path_to_base_LM$`      | Path to the downloaded base language model                   |
| `$Path_to_finetuned_LM$` | Path to the fine-tuned language model to generate textual embeddings |

Make sure these values are correctly set in your shell environment or within the corresponding scripts before execution.

> 🔎 **Note**:
>
> * The fine-tuned language model will be saved to: `checkpoint/...` 
> *  The generated embeddings will be stored under: ``$Path_to_dataset$/$model_type$`, `$model_type$` indicates the type of textual embeddings for the dataset, which are used for various Cognitive Diagnosis (CD) tasks.

### 🧾 Example

In our example, the dataset is located at: `stage2/datasets`. `stage2/datasets/SLP-Math-induct` is used in inductive CD, where the ratio of existing students to new students is 1:1.

## 📌 Stage 2: Adapter-aware Representation Integration (AaRI)

AaRI leverages a textual adapter to extract task-relevant semantic and integrates them with existing modeling paradigms to improve generalization across diverse CD tasks.

The proposed EduEmbed can be evaluated on four CD tasks and Computerized Adaptive Testing (CAT) task.

### 🛠 Implementation Workflow

Please navigate to the following directory:

```
cd stage2
```

------

#### 1. Transductive CD

Please navigate to the following directory:

```bash
cd transductiveCD
```

Run the training command:

```bash
python main.py --method=orcdf --train_file=../datasets/SLP-Math --test_file=../datasets/SLP-Math --seed=0 --batch_size=256 --device=cuda:0 --epoch=20 --lr=1e-3 --latent_dim=64 --inter=kancd --ssl_temp=0.5 --ssl_weight=1e-3 --flip_ratio=0.15 --gcn_layers=3 --keep_prob=1.0 --weight_decay=0 --model_type=$model_type$ --lamda 0.75 --alpha 0.01
```

---

#### 2. Inductive CD

Please navigate to the following directory:

```bash
cd inductiveCD
```

Run the training command:

```bash
python main.py --method=ncd --train_file=../datasets/SLP-MAT-induct --test_file=../datasets/SLP-MAT-induct --seed=$i --batch_size=256 --device=cuda:0 --epoch=100 --lr=5e-4 --inter=ncd --model_type=$model_type$
```

---

#### 3. Cross CD

Please navigate to the following directory:

```bash
cd crossCD
```

Run the training command:

```bash
python main.py --method=orcdf --train_file=../datasets/SLP-Chi --test_file=../datasets/SLP-Math --seed=0 --batch_size=256 --device=cuda:0 --epoch=50 --lr=1e-4 --latent_dim=64 --model_type=$model_type$
```

---

#### 4. Computerized Adaptive Testing (CAT)

Please navigate to the following directory:

```bash
cd CAT/scripts
```

Run the training shell script:

```bash
device="cuda:0"
for dataset in SLP-MAT; do
    for strategy in BECAT; do
        if [ "$strategy" = "BOBCAT" ]; then
            python bobcat_train.py --model biirt-biased --dataset $dataset --seed 0 --device $device
            python bobcat_train.py --model binn-biased --dataset $dataset --seed 0 --device $device
        fi
        python train.py --dataset $dataset --cdm irt --device $device --lr 1e-2 --batch_size 32 --num_epochs 5 --decoder irt --seed 0 --test_size 0.8 --LLM_model_name $model_type$ --lamda 0.75 --alpha 0.01
        python test.py --dataset $dataset --strategy $strategy --cdm irt --device $device --lr 1e-2 --batch_size 32 --num_epochs 2 --decoder irt --seed 0 --test_size 0.8 --wandb --LLM_model_name $model_type$ --test_length 15 --lamda 0.75 --alpha 0.01

        python train.py --dataset $dataset --cdm ncd --device $device --lr 5e-3 --batch_size 128 --num_epochs 5 --decoder ncd --seed 0 --test_size 0.8 --LLM_model_name $model_type$ --lamda 0.75 --alpha 0.01
        python test.py --dataset $dataset --strategy $strategy --cdm ncd --device $device --lr 5e-3 --batch_size 128 --num_epochs 5 --decoder ncd --seed 0 --test_size 0.8 --wandb --LLM_model_name $model_type$ --test_length 15 --lamda 0.75 --alpha 0.01
    done
done
```

---

## 📰 News 
- [x] [2026.1.25] EduEmbed v1.0 is released.

## 📋Requirements

To run the code properly, please install the following dependencies (or install them from `requirements.txt`)

```text
dgl==2.5a241215+cu121
joblib==1.4.2
multiprocess==0.70.16
numpy==1.26.4
pandas==2.2.3
peft==0.14.0
scikit-learn==1.6.0
scipy==1.13.1
safetensors==0.4.5
torch==2.4.0+cu121
torch-geometric==2.6.1
tqdm==4.67.1
transformers==4.47.1
wandb==0.19.1
vegas==6.2.1
```

# Reference :thought_balloon:

Yuanhao Liu, Zihan Zhou, Kaiying Wu, Shuo Liu, Yiyang Huang, Jiajun Guo, Aimin Zhou, and Hong Qian. Embedding Enhancement via Fine-Tuned Language Models for Learner-Item Cognitive Modeling. In Proceedings of the 2026 ACM Web Conference (WWW’26).

## Bibtex

```
@inproceedings{Liu2026EduEmbed,
author = {Yuanhao Liu and Zihan Zhou and Kaiying Wu and Shuo Liu and Yiyang Huang and Jiajun Guo and Aimin Zhou and Hong Qian},
booktitle = {Proceedings of the 2026 ACM Web Conference},
title = {Embedding Enhancement via Fine-Tuned Language Models for Learner-Item Cognitive Modeling},
year = {2026},
address={Dubai, United Arab Emirates}
}
```
