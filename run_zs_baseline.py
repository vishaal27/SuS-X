# Zero-shot CLIP baseline
# Adapted from: https://github.com/openai/CLIP

import os
import clip
import random
import torch
import argparse
from utils import utils

random.seed(1)
torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--backbone', type=str, default='RN50')
parser.add_argument('--dataset', type=str, default='cifar10')
args = parser.parse_args()

features_path = "./features/"

clip_model, preprocess = clip.load(args.backbone)
clip_model.cuda()
clip_model.eval()

input_resolution = clip_model.visual.input_resolution
context_length = clip_model.context_length
vocab_size = clip_model.vocab_size

feat_dims = {'RN50': 1024, 'ViT-B/16': 512, 'RN50x16': 768, 'RN101': 512, 'ViT-L/14': 768, 'ViT-B/32': 512}

dataset = args.dataset
model = args.backbone

disp_name = model
if('/' in model):
    disp_name = model.replace('/', '')

feat_dim = utils.get_model_feat_dims(model)
num_classes = utils.get_num_classes(dataset)

test_features_path = features_path+"/{}_f_test_m{}.pt".format(dataset, disp_name)
test_targets_path = features_path+"/{}_t_test_m{}.pt".format(dataset, disp_name)

# dim nxC 
test_features = torch.load(test_features_path)
# dim n
test_labels = torch.load(test_targets_path)

text_classifier_weights_path = os.path.join(features_path, "{}_zeroshot_text_weights_m{}_ptensemble.pt".format(dataset, disp_name))
text_classifier_weights = torch.load(text_classifier_weights_path)

logits = 100. * test_features @ text_classifier_weights
labels = test_labels
np_preds = torch.argmax(logits, dim=1).cpu().numpy()
np_labels = labels.cpu().numpy()   
zs_acc = 100*(np_preds == np_labels).sum()/np_labels.shape[0]         
print('ZS Acc for Dataset: {}, Model: {} == '.format(args.dataset, args.backbone), zs_acc)