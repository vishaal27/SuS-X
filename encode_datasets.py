# Script to encode the target dataset images using CLIP's image encoders
# We encode the validation and testing splits of each dataset independently
# We also encode few-shot support sets akin to TIP-Adapter for 5 shot configurations (k = 1, 2, 4, 8, 16) [refer Sec. 4.2 of paper]

import os
import numpy as np
import torch
import clip
from tqdm import tqdm
import torch.nn.functional as F
import random
import argparse
from utils import utils
from dataloader import KShotDataLoader

# feature dimensions for each model
feat_dims = {'RN50': 1024, 'ViT-B/16': 512, 'RN50x16': 768, 'RN101': 512, 'ViT-L/14': 768, 'ViT-B/32': 512}

parser = argparse.ArgumentParser()
# number of augmentations to apply for averaging visual features
parser.add_argument('--augment_epoch', type=int, default=10)
parser.add_argument('--dataset', type=str, default='cifar10')
args = parser.parse_args()

# dummy parameters for dataloader
args.val_batch_size = 64
args.train_batch_size = 256

random.seed(1)
torch.manual_seed(1)

req_models = ['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14']
req_shots = [1, 2, 4, 8, 16]

for shot in req_shots:
    for model_name in req_models:

        args.backbone = model_name
        args.k_shot = shot

        name = model_name
        print('Current model: {} and k-shot: {}'.format(model_name, shot))

        disp_name = name
        if('/' in name):
            disp_name = name.replace('/', '')

        model, preprocess = clip.load(name)
        model.eval()

        input_resolution = model.visual.input_resolution
        context_length = model.context_length
        vocab_size = model.vocab_size

        print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
        print("Input resolution:", input_resolution)
        print("Context length:", context_length)
        print("Vocab size:", vocab_size)

        features_path = "./features"

        val_features_path = features_path+"/{}_f_val_m{}.pt".format(args.dataset, disp_name)
        val_targets_path = features_path+"/{}_t_val_m{}.pt".format(args.dataset, disp_name)

        test_features_path = features_path+"/{}_f_test_m{}.pt".format(args.dataset, disp_name)
        test_targets_path = features_path+"/{}_t_test_m{}.pt".format(args.dataset, disp_name)

        train_features_path = features_path+"/{}_f_train_m{}_k{}.pt".format(args.dataset, disp_name, args.k_shot)
        train_targets_path = features_path+"/{}_t_train_m{}_k{}.pt".format(args.dataset, disp_name, args.k_shot)

        if(os.path.exists(train_features_path) and os.path.exists(train_targets_path)):
            load_train = True
        else:
            load_train = False

        if(os.path.exists(test_features_path) and os.path.exists(test_targets_path)):
            load_test = True
        else:
            load_test = False

        if(os.path.exists(val_features_path) and os.path.exists(val_targets_path)):
            load_val = True
        else:
            load_val = False

        # load few shot dataset
        train_images, train_loader, train_loader_shuffle, val_images, val_loader, test_images, test_loader, num_classes, string_classnames = KShotDataLoader(args, preprocess).load_dataset()
        assert num_classes == utils.get_num_classes(args.dataset), 'Num classes for '+args.dataset+' not correct'

        # ------------------------------------------saving val features------------------------------------------
        print('start saving val image features')

        if not load_val:
            val_features = []
            val_labels = []
            with torch.no_grad():
                for i, (images, target) in enumerate(tqdm(val_loader)):

                    images = images.cuda()
                    target = target.cuda()
                    # encode image
                    image_features = model.encode_image(images)
                    # L2 norm image embedding
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    val_features.append(image_features)
                    val_labels.append(target)
            val_features = torch.cat(val_features)
            val_labels = torch.cat(val_labels)

            assert val_features.shape[0]==len(val_images) and val_features.shape[1]==feat_dims[name], 'val_features is not of shape nxC'
            assert val_labels.shape[0]==len(val_images), 'val_labels is not of shape n'

            print('Storing val features to: '+val_features_path+' and '+val_targets_path)

            # dim nxC
            torch.save(val_features, val_features_path)
            # dim n
            torch.save(val_labels, val_targets_path)

        else:
            print('Loading val features from: '+val_features_path+' and '+val_targets_path)

            # dim nxC
            val_features = torch.load(val_features_path)
            # dim n
            val_labels = torch.load(val_targets_path)

            assert val_features.shape[0]==len(val_images) and val_features.shape[1]==feat_dims[name], 'val_features is not of shape nxC'
            assert val_labels.shape[0]==len(val_images), 'val_labels is not of shape n'


        # ------------------------------------------saving test features------------------------------------------
        print('start saving test image features')

        if not load_test:
            test_features = []
            test_labels = []
            with torch.no_grad():
                for i, (images, target) in enumerate(tqdm(test_loader)):

                    images = images.cuda()
                    target = target.cuda()
                    # encode image
                    image_features = model.encode_image(images)
                    # L2 norm image embedding
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    test_features.append(image_features)
                    test_labels.append(target)
            test_features = torch.cat(test_features)
            test_labels = torch.cat(test_labels)

            assert test_features.shape[0]==len(test_images) and test_features.shape[1]==feat_dims[name], 'test_features is not of shape nxC'
            assert test_labels.shape[0]==len(test_images), 'test_labels is not of shape n'

            print('Storing test features to: '+test_features_path+' and '+test_targets_path)

            # dim nxC
            torch.save(test_features, test_features_path)
            # dim n
            torch.save(test_labels, test_targets_path)

        else:
            print('Loading test features from: '+test_features_path+' and '+test_targets_path)

            # dim nxC
            test_features = torch.load(test_features_path)
            # dim n
            test_labels = torch.load(test_targets_path) 

            assert test_features.shape[0]==len(test_images) and test_features.shape[1]==feat_dims[name], 'test_features is not of shape nxC'
            assert test_labels.shape[0]==len(test_images), 'test_labels is not of shape n'

        # imagenet-sketch and imagenet-r do not have train (few-shot) splits
        if(args.dataset=='imagenet-sketch' or args.dataset=='imagenet-r'):
            continue

        # ------------------------------------------saving few-shot support features------------------------------------------
        print('start saving few-shot image features')

        if not load_train:
            
            train_images_targets = []
            train_images_features_agg = []

            # take average of features over multiple augmentations for a more robust feature set
            # similar to averaging done in: https://github.com/gaopengcuhk/Tip-Adapter/blob/fcb06059457a3b74e44ddb0d5c96d2ea7e4c5957/utils.py#L46
            with torch.no_grad():
                for augment_idx in range(args.augment_epoch):
                    train_images_features = []

                    print('Augment time: {:} / {:}'.format(augment_idx, args.augment_epoch))
                    for i, (images, target) in enumerate(tqdm(train_loader)):
                        images = images.cuda()
                        image_features = model.encode_image(images)
                        train_images_features.append(image_features)

                        if augment_idx == 0:
                            target = target.cuda()
                            train_images_targets.append(target)

                    images_features_cat = torch.cat(train_images_features, dim=0).unsqueeze(0)
                    train_images_features_agg.append(images_features_cat)
                
            # concatenate and take mean of features from multiple augment runs
            train_images_features_agg = torch.cat(train_images_features_agg, dim=0).mean(dim=0)
            # L2 normalise image embeddings from few shot dataset -- dim NKxC
            train_images_features_agg /= train_images_features_agg.norm(dim=-1, keepdim=True)
            # dim CxNK
            train_images_features_agg = train_images_features_agg.permute(1, 0)

            # convert all image labels to one hot labels -- dim NKxN
            train_images_targets = F.one_hot(torch.cat(train_images_targets, dim=0)).half()

            assert train_images_features_agg.shape[0]==feat_dims[args.backbone] and train_images_features_agg.shape[1]==args.k_shot*num_classes, 'train_images_features_agg is not of shape CxNK'
            assert train_images_targets.shape[0]==args.k_shot*num_classes and train_images_targets.shape[1]==num_classes, 'train_images_targets is not of shape NKxN'

            print('Storing train features to: '+train_features_path+' and '+train_targets_path)
            # dim CxNK
            torch.save(train_images_features_agg, train_features_path)
            # dim NKxN
            torch.save(train_images_targets, train_targets_path)

        else:
            print('Loading train features from: '+train_features_path+' and '+train_targets_path)
            # dim CxNK
            train_images_features_agg = torch.load(train_features_path)
            # dim NKxN
            train_images_targets = torch.load(train_targets_path)

            assert train_images_features_agg.shape[0]==feat_dims[args.backbone] and train_images_features_agg.shape[1]==args.k_shot*num_classes, 'train_images_features_agg is not of shape CxNK'
            assert train_images_targets.shape[0]==args.k_shot*num_classes and train_images_targets.shape[1]==num_classes, 'train_images_targets is not of shape NKxN'