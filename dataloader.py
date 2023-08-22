# Script for loading all 19 datasets
# We release the exact train and test splits here: https://drive.google.com/drive/folders/1nzRf13Ha1gvKP_n_4a_JreplA0QkHGBh?usp=sharing
# Adapted from: https://github.com/KaiyangZhou/CoOp

import os
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from collections import defaultdict
import argparse
import json
from utils import utils
from torchvision.utils import make_grid
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.utils import save_image
import random
from tqdm import tqdm

DATASET_PATH = './data/{}'
        
class ImageDatasetFromPaths(Dataset):
    def __init__(self, split_entity, transform):
        self.image_paths, self.labels = split_entity.image_paths, split_entity.labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = read_image(img_path)
        except RuntimeError as e:
            # HACK: if the image is corrupted or not readable, then sample a random image
            image_rand = None
            while(image_rand is None):
                rand_ind = random.randint(0, self.__len__())
                try:
                    image_rand = read_image(self.image_paths[rand_ind])
                except RuntimeError as e1:
                    image_rand = None
                    continue
            image = image_rand
            label = self.labels[rand_ind]

        image = transforms.ToPILImage()(image)
        image = image.convert("RGB")

        if(self.transform):
            image = self.transform(image)
        return image, label

class DataEntity():
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

class KShotDataLoader():
    def __init__(self, args, preprocess):
        self.dataset_path = DATASET_PATH.format(args.dataset)
        self.args = args
        # val/test images preprocessing
        self.preprocess = preprocess

    def parse_image_paths(self, dataset_path, splits_paths):
        train_split, val_split, test_split = splits_paths['train'], splits_paths['val'], splits_paths['test']

        train_class_to_images_map = {}
        train_image_paths = []
        train_labels = []
        train_classnames = []

        for ind in train_split:
            train_image_path = ind[0]
            train_label = ind[1]
            train_classname = ind[2]

            if(train_label in train_class_to_images_map):
                train_class_to_images_map[train_label].append(os.path.join(dataset_path, train_image_path))
            else:
                train_class_to_images_map[train_label] = []
                train_class_to_images_map[train_label].append(os.path.join(dataset_path, train_image_path))
            train_image_paths.append(os.path.join(dataset_path, ind[0]))
            train_labels.append(ind[1])
            train_classnames.append((ind[1], ind[2]))

        val_image_paths = [os.path.join(dataset_path, ind[0]) for ind in val_split]
        val_labels = [ind[1] for ind in val_split]
        val_classnames = [(ind[1], ind[2]) for ind in val_split]

        test_image_paths = [os.path.join(dataset_path, ind[0]) for ind in test_split]
        test_labels = [ind[1] for ind in test_split]
        test_classnames = [(ind[1], ind[2]) for ind in test_split]

        unique_classes = list(set(test_labels + train_labels + val_labels))
        unique_classnames = list(set(test_classnames + train_classnames + val_classnames))
        unique_classnames.sort(key=lambda x: x[0])
        unique_classnames = [u[1] for u in unique_classnames]

        assert len(unique_classnames) == utils.get_num_classes(self.args.dataset), 'Total num classes is not correct'
        assert len(unique_classes) == utils.get_num_classes(self.args.dataset), 'Total num classes is not correct'

        return train_class_to_images_map, DataEntity(train_image_paths, train_labels), DataEntity(test_image_paths, test_labels), DataEntity(val_image_paths, val_labels), unique_classnames

    def load_dataset(self):
        if(self.args.dataset == 'imagenet'):
            return self.imagenet_load()
        elif(self.args.dataset == 'imagenet-r'):
            return self.imagenet_r_load()
        elif(self.args.dataset == 'imagenet-sketch'):
            return self.imagenet_sketch_load()
        elif(self.args.dataset == 'stanfordcars'):
            return self.custom_load()
        elif(self.args.dataset == 'ucf101'):
            return self.custom_load()
        elif(self.args.dataset == 'caltech101'):
            return self.custom_load()
        elif(self.args.dataset == 'caltech256'):
            return self.custom_load()
        elif(self.args.dataset == 'cub'):
            return self.custom_load()
        elif(self.args.dataset == 'country211'):
            return self.country_211_load()
        elif(self.args.dataset == 'flowers102'):
            return self.custom_load()
        elif(self.args.dataset == 'sun397'):
            return self.custom_load()
        elif(self.args.dataset == 'dtd'):
            return self.custom_load()
        elif(self.args.dataset == 'eurosat'):
            return self.custom_load()
        elif(self.args.dataset == 'fgvcaircraft'):
            return self.fgvcaircraft_load()
        elif(self.args.dataset == 'oxfordpets'):
            return self.custom_load()
        elif(self.args.dataset == 'food101'):
            return self.custom_load()
        elif(self.args.dataset == 'birdsnap'):
            return self.custom_load()
        elif(self.args.dataset == 'cifar10'):
            return self.cifar10_load()
        elif(self.args.dataset == 'cifar100'):
            return self.cifar100_load()
        else:
            raise ValueError('Dataset not supported')

    def country_211_load(self):

        traindir = os.path.join(self.dataset_path, 'train')
        valdir = os.path.join(self.dataset_path, 'valid')
        testdir = os.path.join(self.dataset_path, 'test')

        val_images = torchvision.datasets.ImageFolder(valdir, transform=self.preprocess)
        test_images = torchvision.datasets.ImageFolder(testdir, transform=self.preprocess)

        val_loader = torch.utils.data.DataLoader(val_images, batch_size=self.args.val_batch_size, num_workers=8, shuffle=False)  
        test_loader = torch.utils.data.DataLoader(test_images, batch_size=self.args.val_batch_size, num_workers=8, shuffle=False)  

        # CLIP-style pre-processing
        train_tranform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

        train_images = torchvision.datasets.ImageFolder(traindir, transform=train_tranform)
        num_classes = len(list(np.unique(train_images.targets)))        

        assert len(list(np.unique(train_images.targets))) == 211, 'train image targets length is not 211'
        split_by_label_dict = defaultdict(list)

        print('Load Country211 data finished.')
        for i in range(len(train_images.imgs)):
            split_by_label_dict[train_images.targets[i]].append(train_images.imgs[i])
        imgs = []
        targets = []

        # randomly sample k-shot images for the few-shot cache training
        # imgs and targets should be of size NK
        for label, items in split_by_label_dict.items():
            imgs = imgs + random.sample(items, self.args.k_shot)
            targets = targets + [label for i in range(self.args.k_shot)]

        assert len(imgs) == self.args.k_shot*num_classes, 'Few-shot training set size is not NK'

        # update few-shot dataloader to only consider few-shot dataset
        train_images.imgs = imgs
        train_images.targets = targets
        train_images.samples = imgs
        train_loader = torch.utils.data.DataLoader(train_images, batch_size=self.args.train_batch_size, num_workers=8, shuffle=False)
        train_loader_shuffle = torch.utils.data.DataLoader(train_images, batch_size=self.args.train_batch_size, num_workers=8, shuffle=True) 

        string_classnames = utils.country211_classes()

        return train_images, train_loader, train_loader_shuffle, val_images, val_loader, test_images, test_loader, num_classes, string_classnames

    def cifar10_load(self):

        # CLIP-style pre-processing
        train_tranform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

        trainset = torchvision.datasets.CIFAR10(root=self.dataset_path, train=True, download=True, transform=train_tranform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.args.train_batch_size, shuffle=True, num_workers=8)
        testset = torchvision.datasets.CIFAR10(root=self.dataset_path, train=False, download=True, transform=self.preprocess)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.args.val_batch_size, shuffle=False, num_workers=8)

        string_classnames = utils.cifar10_clases()
        num_classes = len(string_classnames)

        split_by_label_dict = defaultdict(list)

        print('Load CIFAR-10 data finished.')
        for i in tqdm(range(len(trainset)), ascii=True):
            split_by_label_dict[trainset[i][1]].append(trainset[i][0])
        imgs = []
        targets = []

        # randomly sample k-shot images for the few-shot cache training
        # imgs and targets should be of size NK
        for label, items in split_by_label_dict.items():
            imgs = imgs + random.sample(items, self.args.k_shot)
            targets = targets + [label for i in range(self.args.k_shot)]

        assert len(imgs) == self.args.k_shot*num_classes, 'Few-shot training set size is not NK'

        # update few-shot dataloader to only consider few-shot dataset

        train_images = []
        for (img, target) in zip(imgs, targets):
            train_images.append((img, target))
        train_loader = torch.utils.data.DataLoader(train_images, batch_size=self.args.train_batch_size, num_workers=8, shuffle=False)
        train_loader_shuffle = torch.utils.data.DataLoader(train_images, batch_size=self.args.train_batch_size, num_workers=8, shuffle=True)

        # For CIFAR-10 the test and val sets are the same -- hence returning same sets for both
        return train_images, train_loader, train_loader_shuffle, testset, testloader, testset, testloader, num_classes, string_classnames

    def cifar100_load(self):

        # CLIP-style pre-processing
        train_tranform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

        trainset = torchvision.datasets.CIFAR100(root=self.dataset_path, train=True, download=True, transform=train_tranform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.args.train_batch_size, shuffle=True, num_workers=8)
        testset = torchvision.datasets.CIFAR100(root=self.dataset_path, train=False, download=True, transform=self.preprocess)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.args.val_batch_size, shuffle=False, num_workers=8)

        string_classnames = utils.cifar100_classes()
        num_classes = len(string_classnames)

        split_by_label_dict = defaultdict(list)

        print('Load CIFAR-100 data finished.')
        for i in tqdm(range(len(trainset)), ascii=True):
            split_by_label_dict[trainset[i][1]].append(trainset[i][0])
        imgs = []
        targets = []

        # randomly sample k-shot images for the few-shot cache training
        # imgs and targets should be of size NK
        for label, items in split_by_label_dict.items():
            imgs = imgs + random.sample(items, self.args.k_shot)
            targets = targets + [label for i in range(self.args.k_shot)]

        assert len(imgs) == self.args.k_shot*num_classes, 'Few-shot training set size is not NK'

        # update few-shot dataloader to only consider few-shot dataset

        train_images = []
        for (img, target) in zip(imgs, targets):
            train_images.append((img, target))
        train_loader = torch.utils.data.DataLoader(train_images, batch_size=self.args.train_batch_size, num_workers=8, shuffle=False)
        train_loader_shuffle = torch.utils.data.DataLoader(train_images, batch_size=self.args.train_batch_size, num_workers=8, shuffle=True)

        # For CIFAR-100 the test and val sets are the same -- hence returning same sets for both
        return train_images, train_loader, train_loader_shuffle, testset, testloader, testset, testloader, num_classes, string_classnames

    def imagenet_load(self):  

        traindir = os.path.join(self.dataset_path, 'train')
        valdir = os.path.join(self.dataset_path, 'val')

        val_images = torchvision.datasets.ImageFolder(valdir, transform=self.preprocess)

        val_loader = torch.utils.data.DataLoader(val_images, batch_size=self.args.val_batch_size, num_workers=8, shuffle=False)  

        # CLIP-style pre-processing
        train_tranform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

        train_images = torchvision.datasets.ImageFolder(traindir, transform=train_tranform)
        num_classes = len(list(np.unique(train_images.targets)))        

        assert len(list(np.unique(train_images.targets))) == 1000, 'train image targets length is not 1000'
        split_by_label_dict = defaultdict(list)

        print('Load Imagenet data finished.')
        for i in range(len(train_images.imgs)):
            split_by_label_dict[train_images.targets[i]].append(train_images.imgs[i])
        imgs = []
        targets = []

        # randomly sample k-shot images for the few-shot cache training
        # imgs and targets should be of size NK
        for label, items in split_by_label_dict.items():
            imgs = imgs + random.sample(items, self.args.k_shot)
            targets = targets + [label for i in range(self.args.k_shot)]

        assert len(imgs) == self.args.k_shot*num_classes, 'Few-shot training set size is not NK'

        # update few-shot dataloader to only consider few-shot dataset
        train_images.imgs = imgs
        train_images.targets = targets
        train_images.samples = imgs
        train_loader = torch.utils.data.DataLoader(train_images, batch_size=self.args.train_batch_size, num_workers=8, shuffle=False)
        train_loader_shuffle = torch.utils.data.DataLoader(train_images, batch_size=self.args.train_batch_size, num_workers=8, shuffle=True) 

        string_classnames = utils.imagenet_classes()

        # For Imagenet the test and val sets are the same -- hence returning same sets for both
        return train_images, train_loader, train_loader_shuffle, val_images, val_loader, val_images, val_loader, num_classes, string_classnames

    def imagenet_r_load(self):  

        valdir = os.path.join(self.dataset_path, 'imagenet-r')
        val_images = torchvision.datasets.ImageFolder(valdir, transform=self.preprocess)
        val_loader = torch.utils.data.DataLoader(val_images, batch_size=self.args.val_batch_size, num_workers=8, shuffle=False)  

        num_classes = 200 
        string_classnames = utils.imagenet_r_classes()
        print('Load Imagenet-R data finished.')

        # For Imagenet-R reuse test set for all sets
        return val_images, val_loader, val_loader, val_images, val_loader, val_images, val_loader, num_classes, string_classnames

    def imagenet_sketch_load(self):

        valdir = os.path.join(self.dataset_path, 'images')
        val_images = torchvision.datasets.ImageFolder(valdir, transform=self.preprocess)
        val_loader = torch.utils.data.DataLoader(val_images, batch_size=self.args.val_batch_size, num_workers=8, shuffle=False)  

        num_classes = 1000 
        string_classnames = utils.imagenet_classes()
        print('Load Imagenet-Sketch data finished.')

        # For Imagenet-Sketch reuse test set for all sets
        return val_images, val_loader, val_loader, val_images, val_loader, val_images, val_loader, num_classes, string_classnames

    def custom_load(self):

        if(self.args.dataset == 'stanfordcars'):
            json_path = os.path.join(self.dataset_path, 'split_zhou_StanfordCars.json')
            root_data_dir = self.dataset_path
        elif(self.args.dataset == 'ucf101'):
            json_path = os.path.join(self.dataset_path, 'split_zhou_UCF101.json')
            root_data_dir = os.path.join(self.dataset_path, 'UCF-101-midframes')
        elif(self.args.dataset == 'caltech101'):
            json_path = os.path.join(self.dataset_path, 'split_zhou_Caltech101.json')
            root_data_dir = os.path.join(self.dataset_path, '101_ObjectCategories')
        elif(self.args.dataset == 'caltech256'):
            json_path = os.path.join(self.dataset_path, 'split_Caltech256.json')
            root_data_dir = os.path.join(self.dataset_path, '256_ObjectCategories')
        elif(self.args.dataset == 'cub'):
            json_path = os.path.join(self.dataset_path, 'split_CUB.json')
            root_data_dir = os.path.join(self.dataset_path, 'images')     
        elif(self.args.dataset == 'birdsnap'):
            json_path = os.path.join(self.dataset_path, 'split_Birdsnap.json')
            root_data_dir = os.path.join(self.dataset_path, 'images')
        elif(self.args.dataset == 'flowers102'):
            json_path = os.path.join(self.dataset_path, 'split_zhou_OxfordFlowers.json')
            root_data_dir = os.path.join(self.dataset_path, 'jpg')
        elif(self.args.dataset == 'sun397'):
            json_path = os.path.join(self.dataset_path, 'split_zhou_SUN397.json')
            root_data_dir = os.path.join(self.dataset_path, 'SUN397')
        elif(self.args.dataset == 'dtd'):
            json_path = os.path.join(self.dataset_path, 'split_zhou_DescribableTextures.json')
            root_data_dir = os.path.join(self.dataset_path, 'images')
        elif(self.args.dataset == 'eurosat'):
            json_path = os.path.join(self.dataset_path, 'split_zhou_EuroSAT.json')
            root_data_dir = os.path.join(self.dataset_path, '2750')
        elif(self.args.dataset == 'oxfordpets'):
            json_path = os.path.join(self.dataset_path, 'split_zhou_OxfordPets.json')
            root_data_dir = os.path.join(self.dataset_path, 'images')
        elif(self.args.dataset == 'food101'):
            json_path = os.path.join(self.dataset_path, 'split_zhou_Food101.json')
            root_data_dir = os.path.join(self.dataset_path, 'images')            
        else:
            raise ValueError("Dataset not supported")

        splits_paths = json.load(open(json_path))

        train_class_to_images_map, train_split, test_split, val_split, string_classnames = self.parse_image_paths(root_data_dir, splits_paths)

        if(self.args.dataset=='caltech256'):
            string_classnames = [s.split('.')[1].replace('-101', '') for s in string_classnames]

        img_paths = []
        targets = []

        # CLIP-style pre-processing
        train_tranform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])        

        # randomly sample k-shot images for the few-shot cache training
        # imgs and targets should be of size NK
        for class_id in list(train_class_to_images_map.keys()):
            img_paths = img_paths + random.sample(list(train_class_to_images_map[class_id]), self.args.k_shot)
            targets = targets + [class_id for i in range(self.args.k_shot)]

        train_dataset = ImageDatasetFromPaths(DataEntity(img_paths, targets), transform=train_tranform)
        val_dataset = ImageDatasetFromPaths(val_split, transform=self.preprocess)
        test_dataset = ImageDatasetFromPaths(test_split, transform=self.preprocess)

        print('Load '+str(self.args.dataset)+' data finished.')

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.val_batch_size, num_workers=8, shuffle=False)  
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.val_batch_size, num_workers=8, shuffle=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.train_batch_size, num_workers=8, shuffle=False)
        train_loader_shuffle = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.train_batch_size, num_workers=8, shuffle=True)

        num_classes = len(string_classnames)

        return train_dataset, train_loader, train_loader_shuffle, val_dataset, val_loader, test_dataset, test_loader, num_classes, string_classnames

    def fgvcaircraft_load(self):

        images_dir = os.path.join(self.dataset_path, 'images')

        train_split_image_names_file = os.path.join(self.dataset_path, 'images_variant_train.txt')
        val_split_image_names_file = os.path.join(self.dataset_path, 'images_variant_val.txt')
        test_split_image_names_file = os.path.join(self.dataset_path, 'images_variant_test.txt')
        classnames_file = os.path.join(self.dataset_path, 'variants.txt')

        label_to_classname_mapping = {}
        classname_to_label_mapping = {}

        class_to_samples_map = {}

        with open(classnames_file, 'r') as f:
            string_classnames = [f.strip() for f in f.readlines()]

            for i in range(len(string_classnames)):
                label_to_classname_mapping[i] = string_classnames[i]
                classname_to_label_mapping[string_classnames[i]] = i 

        train_image_paths = []
        train_classnames = []
        train_labels = []

        with open(train_split_image_names_file, 'r') as f:
            paths_and_classes = f.readlines()
            paths_and_classes = [p.strip().split() for p in paths_and_classes]

            for p in paths_and_classes:
                train_image_paths.append(os.path.join(images_dir, p[0]+'.jpg'))
                curr_classname = ' '.join(p[1:])
                train_classnames.append(curr_classname)
                train_labels.append(classname_to_label_mapping[curr_classname])

                if(curr_classname in class_to_samples_map):
                    class_to_samples_map[curr_classname].append(os.path.join(images_dir, p[0]+'.jpg'))
                else:
                    class_to_samples_map[curr_classname] = []
                    class_to_samples_map[curr_classname].append(os.path.join(images_dir, p[0]+'.jpg'))

        with open(test_split_image_names_file, 'r') as f:
            paths_and_classes = f.readlines()
            paths_and_classes = [p.strip().split() for p in paths_and_classes]

            test_image_paths = [os.path.join(images_dir, p[0]+'.jpg') for p in paths_and_classes]
            test_classnames = [' '.join(p[1:]) for p in paths_and_classes]
            test_labels = [classname_to_label_mapping[' '.join(p[1:])] for p in paths_and_classes]

        with open(val_split_image_names_file, 'r') as f:
            paths_and_classes = f.readlines()
            paths_and_classes = [p.strip().split() for p in paths_and_classes]

            val_image_paths = [os.path.join(images_dir, p[0]+'.jpg') for p in paths_and_classes]
            val_classnames = [' '.join(p[1:]) for p in paths_and_classes]
            val_labels = [classname_to_label_mapping[' '.join(p[1:])] for p in paths_and_classes]

        # CLIP-style pre-processing
        train_tranform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

        img_paths = []
        targets = []

        # randomly sample k-shot images for the few-shot cache training
        # imgs and targets should be of size NK
        for class_id in list(class_to_samples_map.keys()):
            img_paths = img_paths + random.sample(list(class_to_samples_map[class_id]), self.args.k_shot)
            targets = targets + [classname_to_label_mapping[class_id] for i in range(self.args.k_shot)]

        train_dataset = ImageDatasetFromPaths(DataEntity(img_paths, targets), transform=train_tranform)
        val_dataset = ImageDatasetFromPaths(DataEntity(val_image_paths, val_labels), transform=self.preprocess)
        test_dataset = ImageDatasetFromPaths(DataEntity(test_image_paths, test_labels), transform=self.preprocess)

        print('Load '+str(self.args.dataset)+' data finished.')

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.val_batch_size, num_workers=8, shuffle=False)  
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.val_batch_size, num_workers=8, shuffle=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.train_batch_size, num_workers=8, shuffle=False)
        train_loader_shuffle = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.train_batch_size, num_workers=8, shuffle=True)

        num_classes = len(string_classnames)

        return train_dataset, train_loader, train_loader_shuffle, val_dataset, val_loader, test_dataset, test_loader, num_classes, string_classnames

if __name__ == '__main__':

    # Test dataloaders for each dataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--k_shot', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='imagenet')
    args = parser.parse_args()

    preprocess = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

    # Imagenet
    args.dataset = 'imagenet'
    k_shot_dl = KShotDataLoader(args, preprocess)
    train_images, train_loader, train_loader_shuffle, val_images, val_loader, test_images, test_loader, num_classes, string_classnames = k_shot_dl.load_dataset()
    assert len(train_images) == args.k_shot*utils.get_num_classes(args.dataset) and len(val_images) == 50000 and len(test_images) == 50000

    for i, (images, targets) in enumerate(test_loader):
        grid = make_grid([images[0], images[1], images[2]])
        print(targets[:6])
        print([string_classnames[s] for s in targets[:6]])
        save_image(images[0:6], 'test_'+str(args.dataset)+'.png', nrow=3)
        break   

    # Imagenet-sketch
    args.dataset = 'imagenet-sketch'
    k_shot_dl = KShotDataLoader(args, preprocess)
    train_images, train_loader, train_loader_shuffle, val_images, val_loader, test_images, test_loader, num_classes, string_classnames = k_shot_dl.load_dataset()
    assert len(val_images) == 50889 and len(test_images) == 50889

    for i, (images, targets) in enumerate(test_loader):
        grid = make_grid([images[0], images[1], images[2]])
        print(targets[:6])
        print([string_classnames[s] for s in targets[:6]])
        save_image(images[0:6], 'test_'+str(args.dataset)+'.png', nrow=3)
        break   

    # Imagenet-R
    args.dataset = 'imagenet-r'
    k_shot_dl = KShotDataLoader(args, preprocess)
    train_images, train_loader, train_loader_shuffle, val_images, val_loader, test_images, test_loader, num_classes, string_classnames = k_shot_dl.load_dataset()
    assert len(val_images) == 30000 and len(test_images) == 30000

    for i, (images, targets) in enumerate(test_loader):
        grid = make_grid([images[0], images[1], images[2]])
        print(targets[:6])
        print([string_classnames[s] for s in targets[:6]])
        save_image(images[0:6], 'test_'+str(args.dataset)+'.png', nrow=3)
        break

    # CIFAR-10
    args.dataset = 'cifar10'    
    k_shot_dl = KShotDataLoader(args, preprocess)
    train_images, train_loader, train_loader_shuffle, val_images, val_loader, test_images, test_loader, num_classes, string_classnames = k_shot_dl.load_dataset()
    assert len(train_images) == args.k_shot*utils.get_num_classes(args.dataset) and len(val_images) == 10000 and len(test_images) == 10000

    for i, (images, targets) in enumerate(test_loader):
        grid = make_grid([images[0], images[1], images[2]])
        print(targets[:6])
        print([string_classnames[s] for s in targets[:6]])
        save_image(images[0:6], 'test_'+str(args.dataset)+'.png', nrow=3)
        break

    # Country 211
    args.dataset = 'country211'    
    k_shot_dl = KShotDataLoader(args, preprocess)
    train_images, train_loader, train_loader_shuffle, val_images, val_loader, test_images, test_loader, num_classes, string_classnames = k_shot_dl.load_dataset()
    assert len(train_images) == args.k_shot*utils.get_num_classes(args.dataset) and len(val_images) == 10550 and len(test_images) == 21100

    for i, (images, targets) in enumerate(test_loader):
        grid = make_grid([images[0], images[1], images[2]])
        print(targets[:6])
        print([string_classnames[s] for s in targets[:6]])
        save_image(images[0:6], 'test_'+str(args.dataset)+'.png', nrow=3)
        break

    # Birdsnap
    args.dataset = 'birdsnap'    
    k_shot_dl = KShotDataLoader(args, preprocess)
    train_images, train_loader, train_loader_shuffle, val_images, val_loader, test_images, test_loader, num_classes, string_classnames = k_shot_dl.load_dataset()
    assert len(train_images) == args.k_shot*utils.get_num_classes(args.dataset) and len(val_images) == 7774 and len(test_images) == 11747

    for i, (images, targets) in enumerate(test_loader):
        grid = make_grid([images[0], images[1], images[2]])
        print(targets[:6])
        print([string_classnames[s] for s in targets[:6]])
        save_image(images[0:6], 'test_'+str(args.dataset)+'.png', nrow=3)
        break

    # CIFAR-100
    args.dataset = 'cifar100'
    k_shot_dl = KShotDataLoader(args, preprocess)
    train_images, train_loader, train_loader_shuffle, val_images, val_loader, test_images, test_loader, num_classes, string_classnames = k_shot_dl.load_dataset()
    assert len(train_images) == args.k_shot*utils.get_num_classes(args.dataset) and len(val_images) == 10000 and len(test_images) == 10000

    for i, (images, targets) in enumerate(test_loader):
        grid = make_grid([images[0], images[1], images[2]])
        print(targets[:6])
        print([string_classnames[s] for s in targets[:6]])
        save_image(images[0:6], 'test_'+str(args.dataset)+'.png', nrow=3)
        break

    # Stanford cars
    args.dataset = 'stanfordcars'
    k_shot_dl = KShotDataLoader(args, preprocess)
    train_images, train_loader, train_loader_shuffle, val_images, val_loader, test_images, test_loader, num_classes, string_classnames = k_shot_dl.load_dataset()
    assert len(train_images) == args.k_shot*utils.get_num_classes(args.dataset) and len(val_images) == 1635 and len(test_images) == 8041 

    for i, (images, targets) in enumerate(test_loader):
        grid = make_grid([images[0], images[1], images[2]])
        print(targets[:6])
        print([string_classnames[s] for s in targets[:6]])
        save_image(images[0:6], 'test_'+str(args.dataset)+'.png', nrow=3)
        break

    # UCF101
    args.dataset = 'ucf101'
    k_shot_dl = KShotDataLoader(args, preprocess)
    train_images, train_loader, train_loader_shuffle, val_images, val_loader, test_images, test_loader, num_classes, string_classnames = k_shot_dl.load_dataset()
    assert len(train_images) == args.k_shot*utils.get_num_classes(args.dataset) and len(val_images) == 1898 and len(test_images) == 3783 

    for i, (images, targets) in enumerate(test_loader):
        grid = make_grid([images[0], images[1], images[2]])
        print(targets[:6])
        print([string_classnames[s] for s in targets[:6]])
        save_image(images[0:6], 'test_'+str(args.dataset)+'.png', nrow=3)
        break

    # Caltech101
    args.dataset = 'caltech101'
    k_shot_dl = KShotDataLoader(args, preprocess)
    train_images, train_loader, train_loader_shuffle, val_images, val_loader, test_images, test_loader, num_classes, string_classnames = k_shot_dl.load_dataset()
    assert len(train_images) == args.k_shot*utils.get_num_classes(args.dataset) and len(val_images) == 1649 and len(test_images) == 2465 

    for i, (images, targets) in enumerate(test_loader):
        grid = make_grid([images[0], images[1], images[2]])
        print(targets[:6])
        print([string_classnames[s] for s in targets[:6]])
        save_image(images[0:6], 'test_'+str(args.dataset)+'.png', nrow=3)
        break

    # Caltech256
    args.dataset = 'caltech256'
    k_shot_dl = KShotDataLoader(args, preprocess)
    train_images, train_loader, train_loader_shuffle, val_images, val_loader, test_images, test_loader, num_classes, string_classnames = k_shot_dl.load_dataset()
    assert len(train_images) == args.k_shot*utils.get_num_classes(args.dataset) and len(val_images) == 6027 and len(test_images) == 9076 

    for i, (images, targets) in enumerate(test_loader):
        grid = make_grid([images[0], images[1], images[2]])
        print(targets[:6])
        print([string_classnames[s] for s in targets[:6]])
        save_image(images[0:6], 'test_'+str(args.dataset)+'.png', nrow=3)
        break

    # CUB
    args.dataset = 'cub'
    k_shot_dl = KShotDataLoader(args, preprocess)
    train_images, train_loader, train_loader_shuffle, val_images, val_loader, test_images, test_loader, num_classes, string_classnames = k_shot_dl.load_dataset()
    assert len(train_images) == args.k_shot*utils.get_num_classes(args.dataset) and len(val_images) == 1194 and len(test_images) == 5794 

    for i, (images, targets) in enumerate(test_loader):
        grid = make_grid([images[0], images[1], images[2]])
        print(targets[:6])
        print([string_classnames[s] for s in targets[:6]])
        save_image(images[0:6], 'test_'+str(args.dataset)+'.png', nrow=3)
        break

    # Flowers102
    args.dataset = 'flowers102'
    k_shot_dl = KShotDataLoader(args, preprocess)
    train_images, train_loader, train_loader_shuffle, val_images, val_loader, test_images, test_loader, num_classes, string_classnames = k_shot_dl.load_dataset()
    assert len(train_images) == args.k_shot*utils.get_num_classes(args.dataset) and len(val_images) == 1633 and len(test_images) == 2463 

    for i, (images, targets) in enumerate(test_loader):
        grid = make_grid([images[0], images[1], images[2]])
        print(targets[:6])
        print([string_classnames[s] for s in targets[:6]])
        save_image(images[0:6], 'test_'+str(args.dataset)+'.png', nrow=3)
        break 

    # Sun397
    args.dataset = 'sun397'
    k_shot_dl = KShotDataLoader(args, preprocess)
    train_images, train_loader, train_loader_shuffle, val_images, val_loader, test_images, test_loader, num_classes, string_classnames = k_shot_dl.load_dataset()
    assert len(train_images) == args.k_shot*utils.get_num_classes(args.dataset) and len(val_images) == 3970 and len(test_images) == 19850 

    for i, (images, targets) in enumerate(tqdm(test_loader, ascii=True)):
        grid = make_grid([images[0], images[1], images[2]])
        print(targets[:6])
        print([string_classnames[s] for s in targets[:6]])
        save_image(images[0:6], 'test_'+str(args.dataset)+'.png', nrow=3)

    # DTD
    args.dataset = 'dtd'
    k_shot_dl = KShotDataLoader(args, preprocess)
    train_images, train_loader, train_loader_shuffle, val_images, val_loader, test_images, test_loader, num_classes, string_classnames = k_shot_dl.load_dataset()
    assert len(train_images) == args.k_shot*utils.get_num_classes(args.dataset) and len(val_images) == 1128 and len(test_images) == 1692 

    for i, (images, targets) in enumerate(test_loader):
        grid = make_grid([images[0], images[1], images[2]])
        print(targets[:6])
        print([string_classnames[s] for s in targets[:6]])
        save_image(images[0:6], 'test_'+str(args.dataset)+'.png', nrow=3)
        break

    # EuroSAT
    args.dataset = 'eurosat'
    k_shot_dl = KShotDataLoader(args, preprocess)
    train_images, train_loader, train_loader_shuffle, val_images, val_loader, test_images, test_loader, num_classes, string_classnames = k_shot_dl.load_dataset()
    assert len(train_images) == args.k_shot*utils.get_num_classes(args.dataset) and len(val_images) == 5400 and len(test_images) == 8100 

    for i, (images, targets) in enumerate(test_loader):
        grid = make_grid([images[0], images[1], images[2]])
        print(targets[:6])
        print([string_classnames[s] for s in targets[:6]])
        save_image(images[0:6], 'test_'+str(args.dataset)+'.png', nrow=3)
        break

    # FGVC Aircraft
    args.dataset = 'fgvcaircraft'
    k_shot_dl = KShotDataLoader(args, preprocess)
    train_images, train_loader, train_loader_shuffle, val_images, val_loader, test_images, test_loader, num_classes, string_classnames = k_shot_dl.load_dataset()
    assert len(train_images) == args.k_shot*utils.get_num_classes(args.dataset) and len(val_images) == 3333 and len(test_images) == 3333 

    for i, (images, targets) in enumerate(test_loader):
        grid = make_grid([images[0], images[1], images[2]])
        print(targets[:6])
        print([string_classnames[s] for s in targets[:6]])
        save_image(images[0:6], 'test_'+str(args.dataset)+'.png', nrow=3)
        break

    # Oxford pets
    args.dataset = 'oxfordpets'
    k_shot_dl = KShotDataLoader(args, preprocess)
    train_images, train_loader, train_loader_shuffle, val_images, val_loader, test_images, test_loader, num_classes, string_classnames = k_shot_dl.load_dataset()
    assert len(train_images) == args.k_shot*utils.get_num_classes(args.dataset) and len(val_images) == 736 and len(test_images) == 3669 

    for i, (images, targets) in enumerate(test_loader):
        grid = make_grid([images[0], images[1], images[2]])
        print(targets[:6])
        print([string_classnames[s] for s in targets[:6]])
        save_image(images[0:6], 'test_'+str(args.dataset)+'.png', nrow=3)
        break

    # Food101
    args.dataset = 'food101'
    k_shot_dl = KShotDataLoader(args, preprocess)
    train_images, train_loader, train_loader_shuffle, val_images, val_loader, test_images, test_loader, num_classes, string_classnames = k_shot_dl.load_dataset()
    assert len(train_images) == args.k_shot*utils.get_num_classes(args.dataset) and len(val_images) == 20200 and len(test_images) == 30300 

    for i, (images, targets) in enumerate(test_loader):
        grid = make_grid([images[0], images[1], images[2]])
        print(targets[:6])
        print([string_classnames[s] for s in targets[:6]])
        save_image(images[0:6], 'test_'+str(args.dataset)+'.png', nrow=3)
        break
