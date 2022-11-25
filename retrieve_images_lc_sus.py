# Script to curate SuS-LC support sets
# Refer Sec 3.1 of paper

import os
from urllib.request import urlopen
from typing import List
from time import time
import pickle
from tqdm import tqdm
import argparse
import json

def validate_web_url(url: str, timeout: int = 10):
    try:
        urlopen(url, timeout=timeout)
        return True
    except:  # URLError
        return False

def main(args):

    download_dir_root: str = args.download_dir_root
    store_dir_root: str = args.store_dir_root

    num_images: int = args.num_images
    read_classnames = os.listdir(download_dir_root)
    # for Mac users, remove aux files
    read_classnames = [s for s in read_classnames if 'DS_Store' not in s]

    start_index = args.start_index
    end_index = args.end_index

    if(end_index>len(read_classnames)):
        end_index = len(read_classnames)  

    for (ind, read_name) in zip(list(range(len(read_classnames[start_index : end_index]))), read_classnames[start_index : end_index]):
        save_folder = read_name.replace(' ', '_').replace('/', '_')

        print('Started class {}: {}: {}'.format(start_index+ind, read_name, save_folder))

        class_store_root = os.path.join(store_dir_root, save_folder.replace(' ', '_').replace('/', '_'))
        os.makedirs(class_store_root, exist_ok=True)

        # mechanism for smart downloading
        files_curr = os.listdir(class_store_root)
        if(len(files_curr)>=num_images):
            print('Class {}: {}: {} already contains {} images or more'.format(start_index+ind, read_name, save_folder, str(num_images)))
            continue

        try:
            class_root_curr = os.path.join(download_dir_root, read_name)
            urls_pickle_file = list(os.listdir(class_root_curr))[0]
            urls_root = class_root_curr
        except Exception as e:
            print(e)
            urls_pickle_file = None

        assert urls_pickle_file is not None, 'Error'
        assert urls_pickle_file == 'urls.pickle', 'Error: '+urls_pickle_file+' '+save_folder+' '+read_name

        st = time()

        urls_path = os.path.join(urls_root, urls_pickle_file)

        with open(urls_path, 'rb') as f:
            urls_and_scores = pickle.load(f)

        n_valid_urls: int = 0
        list_valid_meta_data: List[str] = list()
        for meta_data in tqdm(urls_and_scores):
            url: str = meta_data["url"]
            if validate_web_url(url):
                ext = url.split('/')[-1].split('.')[-1]
                if ext not in ["png", "PNG", "jpg", "JPG", "jpeg", "JPEG"]:
                    if ".png" in url or ".PNG" in url:
                        ext = "png"
                    elif ".jpg" in url or ".JPG" in url:
                        ext = "jpg"
                    elif ".jpeg" in url or ".JPEG" in url:
                        ext = "jpeg"
                    else:
                        ext = "png"

                try:
                    filename = f"{class_store_root}/{n_valid_urls:06d}.{ext}"
                    with open(filename, 'wb') as g:
                        g.write(urlopen(url).read())
                        g.close()
                except:
                    continue

                list_valid_meta_data.append(meta_data)
                n_valid_urls += 1
            if n_valid_urls >= num_images:
                break
        # store metadata
        json.dump(list_valid_meta_data, open(f"{class_store_root}.json", 'w'))
        print(f"Time taken for the {read_name} category: {time() - st:.3f} sec.")
        print('Finished class {}: {}: {}'.format(start_index+ind, read_name, save_folder))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--start_index', help='Starting class index for downloading images', type=int, default=0)
    parser.add_argument('--end_index', help='Ending class index for downloading images', type=int, default=1000)
    parser.add_argument('--num_images', help='Number of images per class to download', type=int, default=100)
    parser.add_argument('--batch_size', help='Batch size for each Stable Diffusion inference', type=int, default=5)
    parser.add_argument('--dataset', help='Dataset to download', type=str, default='cifar10')
    parser.add_argument('--prompt_shorthand', type=str, default='cupl')
    args = parser.parse_args()

    # dummy parameters for dataloader
    args.k_shot = 2
    args.val_batch_size = 64 
    args.train_batch_size = 256

    assert args.end_index>args.start_index, 'end_index is less than or equal to start_index'

    print('Downloading dataset: {}'.format(args.dataset))
    DOWNLOAD_DIR_ROOT = "data/sus-lc/download_urls/{}/{}".format(args.dataset, args.prompt_shorthand)
    args.download_dir_root = DOWNLOAD_DIR_ROOT

    STORE_DIR_ROOT = "data/sus-lc/{}/{}".format(args.dataset, args.prompt_shorthand)
    args.store_dir_root = STORE_DIR_ROOT

    main(args)
