CUPL_PROMPTS = {
	'imagenet-sketch': [
		"Describe how a black and white sketch of {} {} looks like",
		"A black and white sketch of {} {}",
		"Describe a black and white sketch from the internet of {} {}",
	],
	'imagenet-r': [
		"An art drawing of {} {}",
		"Artwork showing {} {}",
		"A cartoon {} {}",
		"An origami of {} {}",
		"A deviant art photo depicting {} {}",
		"An embroidery of {} {}",
		"A graffiti art showing {} {}",
		"A painting of {} {}",
		"A sculpture of {} {}",
		"A black and white sketch of {} {}",
		"A toy {} {}",
		"A videogame of {} {}",
	],
	'caltech101': [
		"Describe what {} {} looks like",
		"What does {} {} look like",
		"Describe a photo of {} {}"
	],
	'caltech256': [
		"Describe what {} {} looks like",
		"What does {} {} look like",
		"Describe a photo of {} {}"
	],
	'country211': [
		"Visually describe what {} looks like",
		"What does the landscape of {} look like",
		"Describe a photo taken in {}",
		"How does a typical photo taken in {} look like",
	],
	'birdsnap': [
		"Describe what {} {}, a species of bird, looks like",
		"What does {} {} look like",
		"Visually describe {} {}, a type of bird",
		"A caption of an image of {} {}, a type of bird",
		"Describe the appearance of {} {}",
		"What are the prominent features to identify {} {} bird",
	],
	'cub': [
		"Describe what {} {}, a species of bird, looks like",
		"What does {} {} look like",
		"Visually describe {} {}, a type of bird",
		"A caption of an image of {} {}, a type of bird",
		"Describe the appearance of {} {}",
		"What are the prominent features to identify {} {} bird",
	],
	'stanfordcars': [
		"How can you identify {} {}",
		"Description of {} {}, a type of car",
		"A caption of a photo of {} {}:",
		"What are the primary characteristics of {} {}?",
		"Description of the exterior of {} {}",
		"What are the identifying characteristics of {} {}, a type of car?",
		"Describe an image from the internet of {} {}",
		"What does {} {} look like?",
		"Describe what {} {}, a type of car, looks like",
	],
	'food101': [
		"Describe what {} {} looks like",
		"Visually describe {} {}",
		"How can you tell that the food in this photo is {} {}?",
	],
	'oxfordpets': [
		"Describe what {} {} pet looks like",
		"Visually describe {} {}, a type of pet",
	],
	'cifar10': [
		"Describe what {} {} looks like",
		"How can you identify {} {}?",
		"What does {} {} look like?",
		"Describe an image from the internet of {} {}",
		"A caption of an image of {} {}: ",
	],
	'cifar100': [
		"Describe what {} {} looks like",
		"How can you identify {} {}?",
		"What does {} {} look like?",
		"Describe an image from the internet of {} {}",
		"A caption of an image of {} {}: ",
	],
	'imagenet': [
		"Describe what {} {} looks like",
		"How can you identify {} {}?",
		"What does {} {} look like?",
		"Describe an image from the internet of {} {}",
		"A caption of an image of {} {}: ",
	],
	'fgvcaircraft': [
		"Describe {} {} aircraft",
		"Describe {} {} aircraft",
	],
	'dtd': [
		"What does {} {} material look like?",
		"What does {} {} surface look like?",
		"What does {} {} texture look like?",
		"What does {} {} object look like?",
		"What does {} {} thing look like?",
		"What does {} {} pattern look like?",
	],
	'sun397': [
		"Describe what {} {} looks like",
		"How can you identify {} {}?",
		"Describe a photo of {} {}",
	],
	'flowers102': [
		"What does {} {} flower look like",
		"Describe the appearance of {} {}",
		"A caption of an image of {} {}",
		"Visually describe {} {}, a type of flower",
	],
	'eurosat': [
		"Describe an aerial satellite view of {} {}",
		"How does a satellite photo of {} {} look like",
		"Visually describe a centered satellite view of {} {}",
	],
	'ucf101': [
		"What does a person doing {} look like",
		"Describe the process of {}",
		"How does a person {}",
	]
}

PHOTO_PROMPTS = {
	'imagenet': "A photo of a {}.",
	'imagenet-sketch': "A black and white pencil sketch of a {}.",
	'cifar10': "A photo of a {}.",
	'cifar100': "A photo of a {}.",
	'birdsnap': "A photo of a {}, a type of bird.",
	'country211': "A photo I took in {}",
	'cub': "A photo of a {}, a type of bird.",
	'caltech101': "A photo of a {}.",
	'caltech256': "A photo of a {}.",        
	'oxfordpets': "A photo of a pet {}.",
	'stanfordcars': "A photo of a {} car.",
	'flowers102': "A photo of a {}, a type of flower.",
	'food101': "A photo of a {}, a type of food.",
	'fgvcaircraft': "A photo of a {}, a type of aircraft",
	'sun397': "A photo of a {}.",
	'dtd': "{} texture.",
	'eurosat': "A centered satellite photo of {}.",
	'ucf101': "A photo of a person doing {}."
}

def return_photo_prompts(dataset):
    return PHOTO_PROMPTS[dataset]