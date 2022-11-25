# SuS-X: Training-Free Name-Only Transfer of Vision-Language Models

Official code for the paper "SuS-X: Training-Free Name-Only Transfer of Vision-Language Models". 

## Introduction
Contrastive Language-Image Pre-training (CLIP) has emerged as a simple yet effective way to train large-scale vision-language models. CLIP demonstrates impressive zero-shot classification and retrieval on diverse downstream tasks. However, to leverage its full potential, fine-tuning still appears to be necessary. Fine-tuning the entire CLIP model can be resource-intensive and unstable. Moreover, recent methods that aim to circumvent this need for
fine-tuning still require access to images from the target distribution. We pursue a different approach and explore the regime of training-free "name-only transfer" in which the only knowledge we possess about the downstream task comprises the names of downstream target categories. We propose a novel method, SuS-X, consisting of two key building blocks: "SuS" and "TIP-X", that requires neither intensive fine-tuning nor costly labelled data. SuS-X achieves state-of-the-art zero-shot classification results on 19 benchmark datasets. We further show the utility of TIP-X in the training-free few-shot setting, where we again achieve state-of-the-art results over strong training-free baselines.

## Getting started
All our code was tested on Python 3.6.8 with Pytorch 1.9.0+cu111. Ideally, our scripts require access to a single GPU (uses `.cuda()` for inference). Inference can also be done on CPUs with minimal changes to the scripts.

#### Setting up environments
We recommend setting up a python virtual environment and installing all the requirements. Please follow these steps to set up the project folder correctly:

```bash
git clone https://github.com/vishaal27/SuS-X.git
cd SuS-X

python3 -m venv ./env
source env/bin/activate

pip install -r requirements.txt
```


The code directory is structured as:
|
|--- sus_construction
|-------------------- laion
|-------------------- stable-diffusion
|------------------------------------- output_jsons
|
|--- tip-x_inference
|-------------------- data
|-------------------- features
|

We enumerate the functionality of the different scripts below:

SuS construction:
1. sus_construction/laion/download_images.py: Script to download the support set images from the downloaded URLs from laion
2. sus_construction/laion/download_urls_lib.py: Script to download the URLs of support set images after obtaining top ranked retrievals for each class prompt from laion
3. sus_construction/stable-diffusion/generate_gpt3_prompts.py: Script to generate the gpt3 prompts for the CuPL prompting strategy
4. sus_construction/stable-diffusion/generate_sd_support_set.py: Script to generate SuS-SD support set samples
5. sus_construction/stable-diffusion/output_jsons: Directory containing all the output prompts generated using GPT-3. These are the prompts used by our CuPL strategy

TIP-X inference:
1. features: Directory containing test and validation set image features, support set features and labels using the SuS-LC-P strategy, and text classifier weights
2. data: Directory containing CIFAR-10 data (directly downloaded using Pytorch dataloader)
3. run_calip_baseline.py: Our re-implementation of the CALIP baseline.
Run using: python run_calip_baseline.py 
4. tipx.py: Script containing implementation of the tipx framework. It contains both the hyperparameter tuned and untuned variants of the code.
Run using: python tipx.py
