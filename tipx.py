# Main implementation of the TIP-X framework
# We use some parts of the TIP-Adapter codebase: https://github.com/gaopengcuhk/Tip-Adapter
# Refer Sec 3.2 of paper

import os
import torch
import clip
import torch.nn as nn
import random
from tqdm import tqdm
import argparse
from utils import utils

random.seed(1)
torch.manual_seed(1)

def compute_image_text_distributions(temp, train_images_features_agg, test_features, val_features, vanilla_zeroshot_weights):
    train_image_class_distribution = train_images_features_agg.T @ vanilla_zeroshot_weights
    train_image_class_distribution = nn.Softmax(dim=-1)(train_image_class_distribution/temp)

    test_image_class_distribution = test_features @ vanilla_zeroshot_weights
    test_image_class_distribution = nn.Softmax(dim=-1)(test_image_class_distribution/temp)

    val_image_class_distribution = val_features @ vanilla_zeroshot_weights
    val_image_class_distribution = nn.Softmax(dim=-1)(val_image_class_distribution/temp)

    return train_image_class_distribution, test_image_class_distribution, val_image_class_distribution

def get_kl_divergence_sims(train_image_class_distribution, test_image_class_distribution):
    bs = 100
    kl_divs_sim = torch.zeros((test_image_class_distribution.shape[0], train_image_class_distribution.shape[0]))

    for i in tqdm(range(test_image_class_distribution.shape[0]//bs)):
        curr_batch = test_image_class_distribution[i*bs : (i+1)*bs]
        repeated_batch = torch.repeat_interleave(curr_batch, train_image_class_distribution.shape[0], dim=0)    
        q = train_image_class_distribution
        q_repeated = torch.cat([q]*bs)
        kl = repeated_batch * (repeated_batch.log() - q_repeated.log())
        kl = kl.sum(dim=-1)
        kl = kl.view(bs, -1)
        kl_divs_sim[ i*bs : (i+1)*bs , : ] = kl  

    return kl_divs_sim

def get_kl_div_sims(args, test_features, val_features, train_features, clip_weights):

    train_image_class_distribution, test_image_class_distribution, val_image_class_distribution = compute_image_text_distributions(args.temperature, train_features, test_features, val_features, clip_weights)

    train_kl_divs_sim = get_kl_divergence_sims(train_image_class_distribution, train_image_class_distribution)
    test_kl_divs_sim = get_kl_divergence_sims(train_image_class_distribution, test_image_class_distribution)
    val_kl_divs_sim = get_kl_divergence_sims(train_image_class_distribution, val_image_class_distribution)

    return train_kl_divs_sim, test_kl_divs_sim, val_kl_divs_sim

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def scale_(x, target):
    
    y = (x - x.min()) / (x.max() - x.min())
    y *= target.max() - target.min()
    y += target.min()
    
    return y

def hparam_search(val_features, val_labels, test_features, test_labels, train_images_features_agg, train_images_targets, zeroshot_weights, val_kl_divs_sim, test_kl_divs_sim):

    search_scale = [50, 50, 30]
    search_step = [200, 20, 50]

    alpha_list = [i * (search_scale[1] - 0.1) / search_step[1] + 0.1 for i in range(search_step[1])]
    beta_list = [i * (search_scale[0] - 1) / search_step[0] + 1 for i in range(search_step[0])]
    gamma_list = [i * (search_scale[2] - 0.1) / search_step[2] + 0.1 for i in range(search_step[2])]

    best_tipx_acc = 0 

    best_gamma_tipx, best_alpha_tipx, best_beta_tipx = 0, 0, 0

    for alpha in alpha_list:
        for beta in beta_list:
            n = 0.
            batch_idx = 0
 
            new_knowledge = val_features @ train_images_features_agg
            cache_logits = ((-1) * (beta - beta * new_knowledge)).exp() @ (train_images_targets)
            clip_logits = 100. * val_features @ zeroshot_weights

            batch_idx += 1
            n += val_features.size(0)

            neg_affs = scale_((val_kl_divs_sim).cuda(), new_knowledge)
            affinities = -neg_affs
            kl_logits = affinities.half() @ train_images_targets

            for gamma in gamma_list:  
                tipx_top1, tipx_top5 = 0., 0.

                tipx_logits = clip_logits + kl_logits * gamma + cache_logits * alpha
                tipx_acc1, tipx_acc5 = accuracy(tipx_logits, val_labels, topk=(1, 5))
                tipx_top1 += tipx_acc1
                tipx_top5 += tipx_acc5
                tipx_top1 = (tipx_top1 / n) * 100
                tipx_top5 = (tipx_top5 / n) * 100

                if tipx_top1 > best_tipx_acc:
                    best_tipx_acc = tipx_top1
                    best_alpha_tipx = alpha
                    best_gamma_tipx = gamma
                    best_beta_tipx = beta

    n = test_features.size(0)

    clip_logits = 100. * test_features @ zeroshot_weights

    neg_affs = scale_((test_kl_divs_sim).cuda(), new_knowledge)
    affinities = -neg_affs
    kl_logits = affinities.half() @ train_images_targets

    tipx_top1, tipx_top5 = 0., 0.

    new_knowledge = test_features @ train_images_features_agg
    cache_logits = ((-1) * (best_beta_tipx - best_beta_tipx * new_knowledge)).exp() @ train_images_targets    
    tipx_logits = clip_logits + kl_logits * best_gamma_tipx + cache_logits * best_alpha_tipx
    tipx_acc1, tipx_acc5 = accuracy(tipx_logits, test_labels, topk=(1, 5))
    tipx_top1 += tipx_acc1
    tipx_top5 += tipx_acc5
    tipx_top1 = (tipx_top1 / n) * 100
    tipx_top5 = (tipx_top5 / n) * 100

    return tipx_top1, best_alpha_tipx, best_beta_tipx, best_gamma_tipx

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='RN50')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--prompt_shorthand', type=str, default='photo')
    parser.add_argument('--sus_type', type=str, default='lc')
    parser.add_argument('--text_prompt_type', type=str, default='combined')
    args = parser.parse_args()

    # dummy parameters for dataloader
    args.k_shot = 2
    args.val_batch_size = 64 
    args.train_batch_size = 256

    feat_dims = {'RN50': 1024, 'ViT-B/16': 512, 'RN50x16': 768, 'RN101': 512, 'ViT-L/14': 768, 'ViT-B/32': 512}

    if(args.backbone=='all'):
        req_models = ['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14']
    else:
        req_models = [args.backbone]
    features_path = "./features"

    for model_ in req_models:

        print('Current model: {}'.format(model_))

        args.backbone = model_

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

        val_features_path = features_path+"/{}_f_val_m{}.pt".format(dataset, disp_name)
        val_targets_path = features_path+"/{}_t_val_m{}.pt".format(dataset, disp_name)

        test_features_path = features_path+"/{}_f_test_m{}.pt".format(dataset, disp_name)
        test_targets_path = features_path+"/{}_t_test_m{}.pt".format(dataset, disp_name)

        support_features_path = os.path.join(features_path, 'sus_{}_{}_{}_f_m{}.pt'.format(args.sus_type, args.prompt_shorthand, dataset, disp_name))
        support_labels_path = os.path.join(features_path, 'sus_{}_{}_{}_t_m{}.pt'.format(args.sus_type, args.prompt_shorthand, dataset, disp_name))

        text_classifier_weights_path = os.path.join(features_path, "{}_zeroshot_text_weights_m{}_pt{}.pt".format(dataset, disp_name, args.text_prompt_type))

        # dim nxC
        val_features = torch.load(val_features_path)
        # dim n
        val_labels = torch.load(val_targets_path)

        # dim nxC 
        test_features = torch.load(test_features_path)
        # dim n
        test_labels = torch.load(test_targets_path)

        # dim nxC
        support_features = torch.load(support_features_path)
        # dim n
        support_labels = torch.load(support_labels_path)

        text_classifier_weights = torch.load(text_classifier_weights_path)

        train_kl_divs_sims, test_kl_divs_sims, val_kl_divs_sims = get_kl_div_sims(args, test_features, val_features, support_features, text_classifier_weights)

        tipx_acc, best_alpha_tipx, best_beta_tipx, best_gamma_tipx = hparam_search(val_features, val_labels, test_features, test_labels, support_features, support_labels, text_classifier_weights, val_kl_divs_sims, test_kl_divs_sims)

        print('--------------------------------------------')
        print('Best for Dataset: {}, Model: {}, SuS Type: {}, Prompting strategy: {}, alpha: {}, beta: {}, gamma: {}, TIP-X Accuracy: {}'.format(args.dataset, args.backbone, args.sus_type, args.prompt_shorthand, best_alpha_tipx, best_beta_tipx, best_gamma_tipx, tipx_acc))
        print('--------------------------------------------')
        print()
        print('----------------------------------------------------------------------------')
