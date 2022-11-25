# Script to generate text classifier weights using the target category names

import os
import clip
import random
import torch
import argparse
from dataloader import KShotDataLoader
from utils import utils

random.seed(1)
torch.manual_seed(1)

'''
Function to return the L2 normalised mean ensembled text feature embeddings using clip's text encoder
'''
def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True) # L2 normalise text embedding
            class_embedding = class_embeddings.mean(dim=0) # take mean over all text embeddings for all prompts
            class_embedding /= class_embedding.norm() # L2 normalise mean embedding
            zeroshot_weights.append(class_embedding)
        # create shape CxN
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

wp_dict = {
    'ensemble': [
        "itap of a {}.",
        "a bad photo of the {}.",
        "a origami {}.",
        "a photo of the large {}.",
        "a {} in a video game.",
        "art of the {}.",
        "a photo of the small {}.",
    ],
    'itap': [
        "itap of a {}.",
    ],
    'origami': [
        "a origami {}.",
    ],
    'small': [
        "a photo of the small {}.",
    ],
    'class_name': [
        "{}",
    ],
    'lowres': [
        "a low resolution photo of the {}.",
    ],
}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10')
args = parser.parse_args()

# dummy parameters for dataloader
args.k_shot = 2
args.val_batch_size = 64 
args.train_batch_size = 256

req_models = ['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14']
req_prompt_types = ['ensemble', 'small', 'origami', 'lowres', 'class_name', 'itap']

feat_dims = {'RN50': 1024, 'ViT-B/16': 512, 'RN50x16': 768, 'RN101': 512, 'ViT-L/14': 768, 'ViT-B/32': 512}

for model in req_models:

    args.backbone = model

    clip_model, preprocess = clip.load(args.backbone)
    clip_model.cuda()
    clip_model.eval()

    input_resolution = clip_model.visual.input_resolution
    context_length = clip_model.context_length
    vocab_size = clip_model.vocab_size

    dataset = args.dataset
    model = args.backbone

    disp_name = model
    if('/' in model):
        disp_name = model.replace('/', '')

    feat_dim = utils.get_model_feat_dims(model)
    num_classes = utils.get_num_classes(dataset)

    # load few shot dataset
    train_images, train_loader, train_loader_shuffle, val_images, val_loader, test_images, test_loader, num_classes, string_classnames = KShotDataLoader(args, preprocess).load_dataset()
    assert num_classes == utils.get_num_classes(args.dataset), 'Num classes for '+args.dataset+' not correct'

    for prompt_type in req_prompt_types:

        print('Current dataset {}, model {} and prompt type {}'.format(dataset, model, prompt_type))

        wp = './features/{}_zeroshot_text_weights_m{}_pt{}.pt'

        if(os.path.exists(wp.format(dataset, disp_name, prompt_type))):
            load_text = True
        else:
            load_text = False

        if not load_text:
            zeroshot_weights = zeroshot_classifier(string_classnames, wp_dict[prompt_type], clip_model)
            print('Storing zeroshot weights to: '+wp.format(dataset, disp_name, prompt_type))
            torch.save(zeroshot_weights, wp.format(dataset, disp_name, prompt_type))
        else:
            print('Reading zeroshot weights from: '+wp.format(dataset, disp_name, prompt_type))
            zeroshot_weights = torch.load(wp.format(dataset, disp_name, prompt_type))

        print(zeroshot_weights.shape)
        assert zeroshot_weights.shape[0]==feat_dims[args.backbone] and zeroshot_weights.shape[1]==num_classes, 'zeroshot_weights are not of dim CxN'