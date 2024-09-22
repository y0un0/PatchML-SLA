import os
import requests
from zipfile import ZipFile
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize COCO2014 dataset.',
        epilog='Example: python coco.py --download-dir ~/data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', type=str, default='~/data/', help='dataset directory on disk')
    args = parser.parse_args()
    return args

def download_url(url, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get('content-length', 0))
    filename = os.path.join(dir, url.split('/')[-1])
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=file_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

def extract_zip(file_path, extract_to):
    with ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def main(dataset_path):
    images_path = os.path.join(dataset_path, 'images')
    labels_url = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
    data_urls = [
        'http://images.cocodataset.org/zips/train2014.zip',
        'http://images.cocodataset.org/zips/val2014.zip'
    ]

    # Download and extract labels
    download_url(labels_url, dataset_path)
    extract_zip(os.path.join(dataset_path, 'annotations_trainval2014.zip'), dataset_path)
    
    # Download and extract images
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    for url in data_urls:
        zip_name = url.split('/')[-1]
        zip_path = os.path.join(images_path, zip_name)
        download_url(url, images_path)
        extract_zip(zip_path, images_path)
        os.remove(zip_path)  # Clean up zip file after extraction

if __name__ == '__main__':
    args = parse_args()
    path = os.path.expanduser(args.download_dir)
    main(path)