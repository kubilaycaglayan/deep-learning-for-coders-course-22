"""02-saving-a-basic-fastai-model

This file was extracted from the corresponding Jupyter notebook.
"""

import subprocess
import sys

# ## Saving a Cats v Dogs Model
# Make sure we've got the latest version of fastai:
subprocess.check_call([sys.executable, '-m', "pip", "install", "-Uqq", "fastai"])

from fastai.vision.all import (
    ImageDataLoaders,
    Resize,
    URLs,
    error_rate,
    get_image_files,
    resnet18,
    untar_data,
    vision_learner,
)

path = untar_data(URLs.PETS)/'images'

def is_cat(x): return x[0].isupper() 

dls = ImageDataLoaders.from_name_func('.',
    get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat,
    item_tfms=Resize(192))

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)

learn.export('model.pkl')
