We provide implementations of both SuS construction methods and the TIP-X inference procedure. All our
scripts provide examples on the CIFAR-10 dataset as a sample dataset.
All our code was tested on Python 3.6.8 with Pytorch 1.9.0+cu111
Ideally, our script require access to a single GPU (uses .cuda() for inference). Inference can also be done on CPUs with minimal chagnes to the scripts.

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