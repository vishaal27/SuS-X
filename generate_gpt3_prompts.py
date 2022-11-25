# Script to generate GPT-3 prompts given a set of category names
# Used for CuPL prompting strategy (refer Sec. 3.1 of paper)
# Adapted from: https://github.com/sarahpratt/CuPL/blob/main/generate_image_prompts.py

import os
import openai
import json
from tqdm import tqdm
import argparse
import time
from utils.prompts_helper import CUPL_PROMPTS as cupl_prompts
from dataloader import KShotDataLoader
from utils import utils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='Dataset to download', type=str, default='cifar10')
parser.add_argument('--openai_key', help='OpenAI key', type=str, default=None)
args = parser.parse_args()

dataset = args.dataset
output_dir = './gpt3_prompts'
openai.api_key = args.openai_key
if args.openai_key is None:
	raise ValueError("No OpenAI token provided. Please provide one for accessing GPT-3")

json_name = "CuPL_prompts_{}.json".format(dataset)

if(args.dataset=='cifar10'):
	string_classnames = utils.cifar10_clases()
	# modify 'plane' to airplane for search disambiguation
	string_classnames[0] = 'airplane'
	num_classes = 10
elif(args.dataset=='cifar100'):
	string_classnames = utils.cifar100_classes()
	num_classes = 100
else:
	# dummy parameters for dataloader
	args.k_shot = 2
	args.val_batch_size = 64
	args.train_batch_size = 256
	_, _, _, _, _, _, _, num_classes, string_classnames = KShotDataLoader(args, None).load_dataset()
string_classnames = [s.replace('_', ' ') for s in string_classnames]

category_list = string_classnames
all_responses = {}
vowel_list = ['A', 'E', 'I', 'O', 'U', 'a', 'e', 'i', 'o', 'u']

# mechanism to restart download smartly
index_to_restart = 0
if(os.path.exists(os.path.join(output_dir, json_name))):
	with open(os.path.join(output_dir, json_name), 'r') as f:
		json_dict = json.load(f)
		for index, item in enumerate(json_dict.items()):
			class_name = item[0]
			if(class_name==string_classnames[index]):
				index_to_restart += 1
				all_responses[class_name] = item[1]

for ind, category in tqdm(enumerate(category_list)):

	if(ind<index_to_restart):
		continue

	if category[0] in vowel_list:
		article = "an"
	else:
		article = "a"

	prompts = cupl_prompts[args.dataset]
	if(args.dataset=='ucf101' or args.dataset=='country211'):
		prompts = [p.format(category) for p in prompts]
	else:
		prompts = [p.format(article, category) for p in prompts]

	all_result = []
	prompt_id = 0
	while(prompt_id<len(prompts)):
		curr_prompt = prompts[prompt_id]
		try:
			# parameters taken directly from CuPL paper
			response = openai.Completion.create(
				engine="text-davinci-002",
				prompt=curr_prompt,
				temperature=.99,
				max_tokens = 50,
				n=10,
				stop="."
			)

			for r in range(len(response["choices"])):
				result = response["choices"][r]["text"]
				if('![' in result or '])' in result or '](' in result):
					continue
				if(len(result)<5):
					continue
				if(len(result)>1 and result[0]=='?'):
					continue
				all_result.append(result.replace("\n\n", "") + ".")
			# sleep to ensure no timeout error
			time.sleep(1)
			prompt_id += 1

		except openai.error.RateLimitError as e:
			# if we hit rate limit, retry for same prompt again
			pass

	all_responses[category] = all_result
	# sleep to ensure no timeout error
	time.sleep(1)

	os.makedirs(os.path.join(output_dir), exist_ok=True)
	with open(os.path.join(output_dir, json_name), 'w') as f:
		json.dump(all_responses, f, indent=4)
