# Script to curate SuS-LC support sets
# Refer Sec 3.1 of paper

import os
from tqdm import tqdm
import pickle
from clip_retrieval.clip_client import ClipClient
import argparse
from utils.prompts_helper import return_photo_prompts
import json
import numpy as np
from dataloader import KShotDataLoader
from utils import utils

def return_cupl_prompts(args):
	# use CuPL prompting strategy: 
	# i.e. use prompts generated from GPT-3 using generate_gpt3_prompts.py	
	cupl_prompts = json.load(open('./gpt3_prompts/CuPL_prompts_{}.json'.format(args.dataset)))
	return cupl_prompts

def main(args):
	
	if(args.dataset=='cifar10'):
		string_classnames = utils.cifar10_clases()
		# modify 'plane' to airplane for search disambiguation
		string_classnames[0] = 'airplane'
	elif(args.dataset=='cifar100'):
		string_classnames = utils.cifar100_classes()
	else:
		# dummy parameters for dataloader
		args.k_shot = 2
		args.val_batch_size = 64
		args.train_batch_size = 256
		_, _, _, _, _, _, _, _, string_classnames = KShotDataLoader(args, None).load_dataset()
	string_classnames = [s.replace('_', ' ') for s in string_classnames]
	class_names = string_classnames

	if(args.prompt_shorthand=='photo'):
		search_prompt = return_photo_prompts(args.dataset)
	elif(args.prompt_shorthand=='cupl'):
		search_prompt = return_cupl_prompts(args)
	else:
		raise ValueError("Prompt type not recognised")

	if(args.prompt_shorthand=='photo'):

		num_images = args.num_images

		client = ClipClient(url="https://knn5.laion.ai/knn-service", indice_name="laion5B", num_images=num_images, aesthetic_weight=0)
		
		for img_class in tqdm(class_names):
			results = client.query(text=search_prompt.format(img_class).replace('_', ' '))
			class_root = os.path.join(args.dir_root, img_class)
			os.makedirs(class_root, exist_ok=True)
			with open(os.path.join(class_root, 'urls.pickle'), 'wb') as f:
				pickle.dump(results, f)
			print('Class: {}, Num res: {}'.format(img_class, len(results)))

	elif(args.prompt_shorthand=='cupl'):

		if(args.end_index>len(search_prompt)):
			args.end_index = len(search_prompt)

		# iterate over all classes
		for index, cls_name in enumerate(search_prompt):

			# mechanism to restart download smartly
			if(index<args.start_index):
				continue

			if(index>args.end_index):
				continue

			class_root = os.path.join(args.dir_root, cls_name)

			# search 10 random cupl prompts
			curr_sp = np.random.permutation(search_prompt[cls_name])[:10]
			num_images = args.num_images//len(curr_sp)

			client = ClipClient(url="https://knn5.laion.ai/knn-service", indice_name="laion5B", num_images=num_images, aesthetic_weight=0)

			res = []
			for c_prompt in curr_sp:
				results = client.query(text=c_prompt.replace('_', ' '))
				os.makedirs(class_root, exist_ok=True)
				res = res + results
			with open(os.path.join(class_root, 'urls.pickle'), 'wb') as f:
				pickle.dump(res, f)
			print('Index: {}, Class: {}, Num res: {}'.format(index, cls_name, len(res)))

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--start_index', help='Starting class index for downloading images', type=int, default=0)
	parser.add_argument('--end_index', help='Ending class index for downloading images', type=int, default=1000)
	parser.add_argument('--num_images', help='Number of image urls per class to download', type=int, default=5000)
	parser.add_argument('--batch_size', help='Batch size for each Stable Diffusion inference', type=int, default=5)
	parser.add_argument('--prompt_shorthand', help='Prompt type to search LAION for', type=str, default='cupl')
	parser.add_argument('--dataset', help='Dataset to download', type=str, default='cifar10')
	args = parser.parse_args()
	assert args.end_index>args.start_index, 'end_index is less than or equal to start_index'

	DIR_ROOT = './data/sus-lc/download_urls/{}/{}'.format(args.dataset, args.prompt_shorthand)
	if(not os.path.exists(DIR_ROOT)):
		os.makedirs(DIR_ROOT)
	args.dir_root = DIR_ROOT

	main(args)