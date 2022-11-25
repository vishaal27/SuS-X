# CuPL baseline
# Adapted from: https://github.com/sarahpratt/CuPL

import os
import clip
import json
import torch
import argparse
from dataloader import KShotDataLoader
from utils import utils
import random

random.seed(1)
torch.manual_seed(1)

'''
Function to return the L2 normalised mean ensembled text feature embeddings using clip's text encoder
'''
def zeroshot_classifier(classnames, gpt3_templates, model, templates=None):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            if(templates is not None):
                texts = [template.format(classname.replace('_', ' ')) for template in templates]
            else:
                texts = []

            for t in gpt3_templates[classname.replace('_', ' ')]:
                texts.append(t)
            texts = clip.tokenize(texts, truncate=True).cuda()  # tokenize
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
parser.add_argument('--backbone', type=str, default='RN50')
parser.add_argument('--dataset', type=str, default='cifar10')
args = parser.parse_args()

# dummy parameters for dataloader
args.k_shot = 2
args.val_batch_size = 64 
args.train_batch_size = 256

json_root = './gpt3_prompts/CuPL_prompts_{}.json'
features_path = "./features"

clip_model, preprocess = clip.load(args.backbone)
clip_model.cuda()
clip_model.eval()

input_resolution = clip_model.visual.input_resolution
context_length = clip_model.context_length
vocab_size = clip_model.vocab_size

feat_dims = {'RN50': 1024, 'ViT-B/16': 512, 'RN50x16': 768, 'RN101': 512, 'ViT-L/14': 768, 'ViT-B/32': 512}

dataset = args.dataset
model = args.backbone

# Image train and test features from CLIP encoder
disp_name = model
if('/' in model):
    disp_name = model.replace('/', '')

feat_dim = utils.get_model_feat_dims(model)
num_classes = utils.get_num_classes(dataset)

train_images, train_loader, train_loader_shuffle, val_images, val_loader, test_images, test_loader, num_classes, string_classnames = KShotDataLoader(args, preprocess).load_dataset()

test_features_path = features_path+"/{}_f_test_m{}.pt".format(dataset, disp_name)
test_targets_path = features_path+"/{}_t_test_m{}.pt".format(dataset, disp_name)

# dim nxC 
test_features = torch.load(test_features_path)
# dim n
test_labels = torch.load(test_targets_path)

cupl_save_weights_path = './features/{}_zeroshot_text_weights_m{}_ptcupl.pt'
combined_save_weights_path = './features/{}_zeroshot_text_weights_m{}_ptcombined.pt'

if(os.path.exists(cupl_save_weights_path.format(dataset, disp_name))):
    load_cupl_text = True
else:
    load_cupl_text = False

if(os.path.exists(combined_save_weights_path.format(dataset, disp_name))):
    load_combined_text = True
else:
    load_combined_text = False

gpt3_prompts = json.load(open(json_root.format(dataset)))

# for consistency with SD and LC sus construction
if(dataset=='cifar10'):
    string_classnames[0] = 'airplane'

if not load_cupl_text:
    zeroshot_weights = zeroshot_classifier(string_classnames, gpt3_prompts, clip_model)
    torch.save(zeroshot_weights, cupl_save_weights_path.format(dataset, disp_name))
else:
    zeroshot_weights = torch.load(cupl_save_weights_path.format(dataset, disp_name))

assert zeroshot_weights.shape[0]==feat_dims[args.backbone] and zeroshot_weights.shape[1]==num_classes, 'zeroshot_weights are not of dim CxN'

cupl_weights = zeroshot_weights.clone()

if not load_combined_text:
    zeroshot_weights = zeroshot_classifier(string_classnames, gpt3_prompts, clip_model, templates=wp_dict['ensemble'])
    torch.save(zeroshot_weights, combined_save_weights_path.format(dataset, disp_name))
else:
    zeroshot_weights = torch.load(combined_save_weights_path.format(dataset, disp_name))

assert zeroshot_weights.shape[0]==feat_dims[args.backbone] and zeroshot_weights.shape[1]==num_classes, 'zeroshot_weights are not of dim CxN'

combined_weights = zeroshot_weights.clone()

logits = 100. * test_features @ cupl_weights
labels = test_labels
np_preds = torch.argmax(logits, dim=1).cpu().numpy()
np_labels = labels.cpu().numpy()   
cupl_acc = 100*(np_preds == np_labels).sum()/np_labels.shape[0]         
print('CUPL Acc for Dataset: {}, Model: {} == '.format(args.dataset, args.backbone), cupl_acc)

logits = 100. * test_features @ combined_weights
labels = test_labels
np_preds = torch.argmax(logits, dim=1).cpu().numpy()
np_labels = labels.cpu().numpy()   
combined_acc = 100*(np_preds == np_labels).sum()/np_labels.shape[0]
print('CUPL+e Acc for Dataset: {}, Model: {} == '.format(args.dataset, args.backbone), combined_acc)
