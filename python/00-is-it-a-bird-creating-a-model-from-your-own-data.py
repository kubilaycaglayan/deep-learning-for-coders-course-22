"""Is it a bird?

This file was extracted from the corresponding Jupyter notebook.
"""

# NB: Kaggle requires phone verification to use the internet or a GPU. If you
# haven't done that yet, the cell below will fail. This code is only here to
# check that your internet is enabled. It doesn't do anything else.
# Here's a help thread on getting your phone number verified:
# https://www.kaggle.com/product-feedback/135367

import os
import socket
import subprocess
import sys
import warnings


try:
    socket.setdefaulttimeout(1)
    socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(('1.1.1.1', 53))
except socket.error as ex:
    raise Exception("STOP: No internet. Click '>|' in top right and set 'Internet' switch to on")


# It's a good idea to ensure you're running the latest version of any libraries
# you need.
# NB: You can safely ignore any warnings or errors pip spits out about running
# as root or incompatibilities.
iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')

if iskaggle:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-Uqq', 'fastai'])


# Step 1: Download images of birds and non-birds
# Skip this section if you already have duckduckgo_search installed.
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-Uqq', 'duckduckgo_search'])

from duckduckgo_search import DDGS  # DuckDuckGo has changed the API so we need to update
from fastcore.all import *


def search_images(keywords, max_images=200):
    return L(DDGS().images(keywords, max_results=max_images)).itemgot('image')


urls = search_images('bird photos', max_images=1)
urls[0]

from fastdownload import download_url

dest = 'bird.jpg'
download_url(urls[0], dest, show_progress=False)

from fastai.vision.all import *

im = Image.open(dest)
im.to_thumb(256, 256)

download_url(search_images('forest photos', max_images=1)[0], 'forest.jpg', show_progress=False)
Image.open('forest.jpg').to_thumb(256, 256)

searches = 'forest', 'bird'
path = Path('bird_or_not')

from time import sleep

for o in searches:
    dest = path / o
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    sleep(10)  # Pause between searches to avoid over-loading server
    download_images(dest, urls=search_images(f'{o} sun photo'))
    sleep(10)
    download_images(dest, urls=search_images(f'{o} shade photo'))
    sleep(10)
    resize_images(path / o, max_size=400, dest=path / o)

failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path)

dls.show_batch(max_n=6)

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)

is_bird, _, probs = learn.predict(PILImage.create('bird.jpg'))
print(f"This is a: {is_bird}.")
print(f"Probability it's a bird: {probs[0]:.4f}\nProbability it's a forest image: {probs[1]:.4f}")
