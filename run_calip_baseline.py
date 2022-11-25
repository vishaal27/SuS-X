# Our re-implementation of the CALIP baseline
# Followed the model specifications described in https://arxiv.org/pdf/2209.14169.pdf

import os
import numpy as np
import clip
import torch.nn as nn
import random
from tqdm import tqdm
import torch
import argparse
from dataloader import KShotDataLoader
from utils import utils

random.seed(1)
torch.manual_seed(1)

# taken from: https://arxiv.org/pdf/2209.14169.pdf
CALIP_HPARAMS = {
    'imagenet': [1.12, 0.02],
    'caltech101': [5, 0.18],
    'sun397': [0.43, 0.01],
    'food101': [0.6, 0.02],
    'flowers102': [0.5, 0.01],
    'stanfordcars': [2.8, 0.01],
    'fgvcaircraft': [1.3, 0.01],
    'oxfordpets': [0.61, 0.01],
    'dtd': [1.4, 0.01],
    'eurosat': [6.08, 0.06],
    'ucf101': [1.28, 0.01]
}

SEARCH_SPACE = {
    'imagenet-sketch': [0.5, 2, 0, 0.1],
    'imagenet-r': [0.5, 2, 0, 0.1],
    'caltech256': [3, 6, 0, 0.1],
    'cifar10': [0, 3, 0, 0.1],
    'cifar100': [0, 3, 0, 0.1],
    'cub': [0, 3, 0, 0.1],
    'birdsnap': [0, 3, 0, 0.1],
    'country211': [0, 3, 0, 0.1]
}

def hparam_search(dataset, clip_logits, visual_logits, text_logits, test_labels):

    if(dataset in CALIP_HPARAMS):
        logits = clip_logits + CALIP_HPARAMS[dataset][0] * visual_logits + CALIP_HPARAMS[dataset][1] * text_logits

        labels = test_labels
        np_preds = torch.argmax(logits, dim=1).cpu().numpy()
        np_labels = labels.cpu().numpy()            
        return 100*(np_preds == np_labels).sum()/np_labels.shape[0]

    else:
        
        b1_space = np.linspace(SEARCH_SPACE[dataset][0], SEARCH_SPACE[dataset][1], 10)
        b2_space = np.linspace(SEARCH_SPACE[dataset][2], SEARCH_SPACE[dataset][3], 5)

        best_acc = 0
        best_b1 = None
        best_b2 = None

        print('Hparam search for {}'.format(dataset))
        for b1 in tqdm(b1_space):
            for b2 in b2_space:
                    logits = clip_logits + b1 * visual_logits + b2 * text_logits
                    labels = test_labels
                    np_preds = torch.argmax(logits, dim=1).cpu().numpy()
                    np_labels = labels.cpu().numpy()
                    acc = 100*(np_preds == np_labels).sum()/np_labels.shape[0]

                    if(acc>best_acc):
                        best_acc = acc
                        best_b1 = b1
                        best_b2 = b2

        return best_acc

parser = argparse.ArgumentParser()
parser.add_argument('--backbone', type=str, default='RN50')
parser.add_argument('--dataset', type=str, default='cifar10')
args = parser.parse_args()

# dummy parameters for dataloader
args.k_shot = 2
args.val_batch_size = 64 
args.train_batch_size = 256

features_path = "./features/"

# This baseline requires the unoptimised non-JIT model loading from CLIP -- do not load the optimised JIT model
clip_model, preprocess = clip.load(args.backbone, load_spatial=True)
clip_model.cuda()
clip_model.eval()

input_resolution = clip_model.visual.input_resolution
context_length = clip_model.context_length
vocab_size = clip_model.vocab_size

train_images, train_loader, train_loader_shuffle, val_images, val_loader, test_images, test_loader, num_classes, string_classnames = KShotDataLoader(args, preprocess).load_dataset()

feat_dims = {'RN50': 1024, 'ViT-B/16': 512, 'RN50x16': 768, 'RN101': 512, 'ViT-L/14': 768, 'ViT-B/32': 512}

dataset = args.dataset
model = args.backbone

disp_name = model
if('/' in model):
    disp_name = model.replace('/', '')

feat_dim = utils.get_model_feat_dims(model)
num_classes = utils.get_num_classes(dataset)

assert num_classes == utils.get_num_classes(args.dataset), 'Num classes for '+args.dataset+' not correct'

test_features_path = features_path+"/{}_f_test_m{}.pt".format(dataset, disp_name)
test_targets_path = features_path+"/{}_t_test_m{}.pt".format(dataset, disp_name)

# dim nxC 
test_features = torch.load(test_features_path)
# dim n
test_labels = torch.load(test_targets_path)

text_classifier_weights_path = os.path.join(features_path, "{}_zeroshot_text_weights_m{}_ptensemble.pt".format(dataset, disp_name))
text_classifier_weights = torch.load(text_classifier_weights_path)

test_features = []
test_labels = []

test_fs = []
test_fv = []

with torch.no_grad():
    for i, (images, target) in enumerate(tqdm(test_loader)):

        images = images.cuda()
        target = target.cuda()
        # encode image
        image_features = clip_model.encode_image(images)
        # B x HW x C
        f_s = image_features[1].reshape(images.shape[0], -1, feat_dims[model])
        # B x C
        f_v = image_features[0]
        # L2 norm image embedding
        f_s /= f_s.norm(dim=-1, keepdim=True)
        f_v /= f_v.norm(dim=-1, keepdim=True)

        test_fs.append(f_s)
        test_fv.append(f_v)
        test_labels.append(target)
test_fs_ = torch.cat(test_fs).detach().clone()
test_fv_ = torch.cat(test_fv).detach().clone()
test_labels_ = torch.cat(test_labels).clone()

if(dataset in ['imagenet-sketch', 'imagenet']):
    bs = 10000
    num_batches = test_fv_.shape[0]//bs
    if(test_fv_.shape[0]%bs!=0):
        num_batches += 1
else:
    bs = test_fv_.shape[0]
    num_batches = 1

total_zs_acc = 0
total_calip_acc = 0

for batch_num in tqdm(range(num_batches)):

    test_fs = test_fs_[ bs*batch_num : bs*(batch_num+1), ... ]
    test_fv = test_fv_[ bs*batch_num : bs*(batch_num+1), ... ]
    test_labels = test_labels_[ bs*batch_num : bs*(batch_num+1), ... ]

    test_ft = text_classifier_weights.T

    logits = 100. * test_fv @ test_ft.T
    labels = test_labels
    np_preds = torch.argmax(logits, dim=1).cpu().numpy()
    np_labels = labels.cpu().numpy()      
    zs_acc = 100*(np_preds == np_labels).sum()/np_labels.shape[0]

    A = torch.einsum('bhc, kc -> bhk', test_fs, test_ft)
    test_f_s_a = torch.einsum('bhk, kc -> bhc', nn.Softmax(dim=-1)(A/2), test_ft)
    perm_A = torch.permute(A, (0, 2, 1))
    test_f_t_a = torch.bmm(nn.Softmax(dim=-1)(perm_A/2), test_fs)

    # Baseline only implemented with RN50 backbone for comparison
    if(model=='RN50'):
        test_f_s_a_spatial = test_f_s_a.reshape(-1, 2048, 7, 7)
        if(dataset in ['imagenet-r', 'food101']):
            test_fvas = []
            bs = 10000
            num_its = (test_f_s_a_spatial.shape[0]//bs)
            if(test_f_s_a_spatial.shape[0] % bs==0):
                pass
            else:
                num_its += 1
            for i in range(num_its):
                fts = clip_model.visual.attnpool(test_f_s_a_spatial[bs*i : bs*(i+1), ...])
                test_fvas.append(fts)
            test_f_v_a = torch.cat(test_fvas, dim=0)
        else:
            test_f_v_a = clip_model.visual.attnpool(test_f_s_a_spatial)

    test_f_t_a /= test_f_t_a.norm(dim=-1, keepdim=True)
    test_f_v_a /= test_f_v_a.norm(dim=-1, keepdim=True)

    clip_logits = 100. * test_fv @ test_ft.T

    visual_guided_logits = 100. * torch.bmm(test_fv.unsqueeze(1), test_f_t_a.permute(0, 2, 1)).squeeze(1)

    textual_blended_logits = 100. * test_f_v_a @ test_ft.T

    acc = hparam_search(dataset, clip_logits, visual_guided_logits, textual_blended_logits, test_labels)
    total_calip_acc += acc
    total_zs_acc += zs_acc

print('CALIP Acc for Dataset: {}, Model: {} == '.format(args.dataset, args.backbone), total_calip_acc/num_batches)
