"""06-why-you-should-use-a-framework

This file was extracted from the corresponding Jupyter notebook.
"""

import subprocess
import sys
import os
import zipfile
from pathlib import Path

# ## Introduction and set up
import numpy as np
import pandas as pd
import torch

from fastai.tabular.all import (
    Categorify,
    CategoryBlock,
    FillMissing,
    Normalize,
    RandomSplitter,
    TabularPandas,
    accuracy,
    set_seed,
    slide,
    tabular_learner,
    valley,
)

iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
if iskaggle:
    path = Path('../input/titanic')
    subprocess.check_call([sys.executable, '-m', "pip", "install", "-Uqq", "fastai"])
else:
    import kaggle
    path = Path('titanic')
    kaggle.api.competition_download_cli(str(path))
    zipfile.ZipFile(f'{path}.zip').extractall(path)

pd.options.display.float_format = '{:.2f}'.format
set_seed(42)

# ## Prep the data
df = pd.read_csv(path/'train.csv')

def add_features(df):
    df['LogFare'] = np.log1p(df['Fare'])
    df['Deck'] = df.Cabin.str[0].map(dict(A="ABC", B="ABC", C="ABC", D="DE", E="DE", F="FG", G="FG"))
    df['Family'] = df.SibSp+df.Parch
    df['Alone'] = df.Family==1
    df['TicketFreq'] = df.groupby('Ticket')['Ticket'].transform('count')
    df['Title'] = df.Name.str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
    df['Title'] = df.Title.map(dict(Mr="Mr",Miss="Miss",Mrs="Mrs",Master="Master")).value_counts(dropna=False)

add_features(df)

splits = RandomSplitter(seed=42)(df)

dls = TabularPandas(
    df, splits=splits,
    procs = [Categorify, FillMissing, Normalize],
    cat_names=["Sex","Pclass","Embarked","Deck", "Title"],
    cont_names=['Age', 'SibSp', 'Parch', 'LogFare', 'Alone', 'TicketFreq', 'Family'],
    y_names="Survived", y_block = CategoryBlock(),
).dataloaders(path=".")

# ## Train the model
learn = tabular_learner(dls, metrics=accuracy, layers=[10,10])

learn.lr_find(suggest_funcs=(slide, valley))

learn.fit(16, lr=0.03)

# ## Submit to Kaggle
tst_df = pd.read_csv(path/'test.csv')
tst_df['Fare'] = tst_df.Fare.fillna(0)
add_features(tst_df)

tst_dl = learn.dls.test_dl(tst_df)

preds,_ = learn.get_preds(dl=tst_dl)

tst_df['Survived'] = (preds[:,1]>0.5).int()
sub_df = tst_df[['PassengerId','Survived']]
sub_df.to_csv('sub.csv', index=False)

subprocess.run(['head', "sub.csv"], check=True)

# ## Ensembling
def ensemble():
    learn = tabular_learner(dls, metrics=accuracy, layers=[10,10])
    with learn.no_bar(),learn.no_logging(): learn.fit(16, lr=0.03)
    return learn.get_preds(dl=tst_dl)[0]

learns = [ensemble() for _ in range(5)]

ens_preds = torch.stack(learns).mean(0)

tst_df['Survived'] = (ens_preds[:,1]>0.5).int()
sub_df = tst_df[['PassengerId','Survived']]
sub_df.to_csv('ens_sub.csv', index=False)

# ## Final thoughts
