# Script to generate SuS-SD support sets
# Refer Sec 3.1 of paper

import os
from diffusers import StableDiffusionPipeline
import torch
import argparse
import json
import random
from utils.prompts_helper import return_photo_prompts
from utils import utils
from dataloader import KShotDataLoader

def generate_prompt_array(args, class_name, batch_size, gpt3_prompts):
	if(args.prompt_shorthand == 'photo'):
		pr_ = return_photo_prompts(args.dataset).format(class_name)
		prompt = [pr_] * batch_size
	elif(args.prompt_shorthand == 'cupl'):
		if(args.batch_size>len(gpt3_prompts[class_name])):
			prompt = gpt3_prompts[class_name] + random.sample(gpt3_prompts[class_name], args.batch_size-len(gpt3_prompts[class_name]))
		else:
			prompt = random.sample(gpt3_prompts[class_name], args.batch_size)

	assert len(prompt) == args.batch_size, 'Prompt array is larger than batch size'
	return prompt

def get_save_folder_and_class_names(args):

	if(args.dataset=='imagenet'):
		imagenet_dir = './data/imagenet/val'
		imagenet_synsets = os.listdir(imagenet_dir)
		imagenet_classes = [utils.synset_to_class_map[im] for im in imagenet_synsets]
		return imagenet_synsets, imagenet_classes
	elif(args.dataset=='cifar10'):
		string_classnames = utils.cifar10_clases()
		# save class names according to right class naming
		string_classnames[0] = 'airplane'
		string_classnames[1] = 'automobile'
		return string_classnames, string_classnames
	elif(args.dataset=='cifar100'):
		string_classnames = utils.cifar100_classes()
		string_classnames = [s.replace('_', ' ') for s in string_classnames]
		return string_classnames, string_classnames
	else:
		_, _, _, _, _, _, _, _, string_classnames = KShotDataLoader(args, None).load_dataset()
		string_classnames = [s.replace('_', ' ') for s in string_classnames]
		return string_classnames, string_classnames

def main(args):
	auth_token = args.huggingface_key
	if(args.huggingface_key is None):
		raise ValueError("No HuggingFace token provided. Please provide one.")

	if(args.cache_dir is None):
		pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=auth_token)
	else:
		pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=auth_token, cache_dir=args.cache_dir)

	for attribute, _ in pipe.__dict__.items():
		if(attribute in ['vae', 'text_encoder', 'unet']):
			setattr(pipe, attribute, getattr(pipe, attribute).cuda())

	save_folders, class_names = get_save_folder_and_class_names(args)
	gpt3_prompts = None

	if(args.prompt_shorthand=='cupl'):
		# use CuPL prompting strategy: 
		# i.e. use prompts generated from GPT-3 using generate_gpt3_prompts.py
		gpt3_prompts = json.load(open('./gpt3_prompts/CuPL_prompts_{}.json'.format(args.dataset)))

	stable_diff_gen_dir = './data/sus-sd/{}/{}'.format(args.dataset, args.prompt_shorthand)

	num_images = args.num_images
	batch_size = args.batch_size

	start_index = args.start_index
	end_index = args.end_index

	if(end_index>len(class_names)):
		end_index = len(class_names)

	for (ind, save_folder, class_name) in zip(list(range(len(save_folders[start_index : end_index]))), save_folders[start_index : end_index], class_names[start_index : end_index]):

		print('Started class {}: {}: {}'.format(start_index+ind, class_name, save_folder))

		if(not os.path.exists(os.path.join(stable_diff_gen_dir, save_folder))):
			os.makedirs(os.path.join(stable_diff_gen_dir, save_folder))

		# mechanism to restart download smartly
		files_curr = os.listdir(os.path.join(stable_diff_gen_dir, save_folder))
		if(len(files_curr)>=num_images):
			print('Class {}: {}: {} already contains {} images or more'.format(start_index+ind, class_name, save_folder, str(num_images)))
			continue

		for batch_ind in range(num_images//batch_size):
			prompt = generate_prompt_array(args, class_name, batch_size, gpt3_prompts)
			generator = torch.Generator("cuda").manual_seed(batch_ind) 
			# generate support samples
			images = pipe(prompt, guidance_scale=args.guidance_scale, num_inference_steps=args.num_inference_steps, generator=generator)['sample']
			for image_ind, img in enumerate(images):
				img.save(os.path.join(stable_diff_gen_dir, save_folder, '{}_{}.JPEG'.format(batch_ind, image_ind)), 'JPEG')
		print('Finished class {}: {}: {}'.format(start_index+ind, class_name, save_folder))

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--start_index', help='Starting class index for downloading images', type=int, default=0)
	parser.add_argument('--end_index', help='Ending class index for downloading images', type=int, default=1000)
	parser.add_argument('--guidance_scale', help='Stable Diffusion guidance scale', type=float, default=9.5)
	parser.add_argument('--num_inference_steps', help='Number of denoising steps', type=int, default=85)
	parser.add_argument('--num_images', help='Number of images per class to download', type=int, default=100)
	parser.add_argument('--batch_size', help='Batch size for each Stable Diffusion inference', type=int, default=5)
	parser.add_argument('--dataset', help='Dataset to download', type=str, default='cifar10')
	parser.add_argument('--prompt_shorthand', help='Name of sub-directory for storing the dataset based on prompt', type=str, default='photo')
	parser.add_argument('--huggingface_key', help='Huggingface key', type=str, default=None)
	parser.add_argument('--cache_dir', help='Directory to store pre-trained stable diffusion model weights', type=str, default=None)
	args = parser.parse_args()
	assert args.end_index>args.start_index, 'end_index is less than or equal to start_index'

	# dummy parameters for dataloader
	args.k_shot = 2
	args.val_batch_size = 64 
	args.train_batch_size = 256

	main(args)