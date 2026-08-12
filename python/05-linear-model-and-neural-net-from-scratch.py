"""05-linear-model-and-neural-net-from-scratch

This file was extracted from the corresponding Jupyter notebook.
"""

import os
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import sympy
import torch
import torch.nn.functional as F
from fastai.data.transforms import RandomSplitter
from torch import tensor

# ## Introduction
iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
if iskaggle: path = Path('../input/titanic')
else:
    path = Path('titanic')
    if not path.exists():
        import kaggle
        kaggle.api.competition_download_cli(str(path))
        zipfile.ZipFile(f'{path}.zip').extractall(path)

np.set_printoptions(linewidth=140)
torch.set_printoptions(linewidth=140, sci_mode=False, edgeitems=7)
pd.set_option('display.width', 140)

# ## Cleaning the data
df = pd.read_csv(path/'train.csv')
df

df.isna().sum()

modes = df.mode().iloc[0]
modes

df.fillna(modes, inplace=True)

df.isna().sum()

df.describe(include=(np.number))

df['Fare'].hist();

df['LogFare'] = np.log(df['Fare']+1)

df['LogFare'].hist();

pclasses = sorted(df.Pclass.unique())
pclasses

df.describe(include=[object])

df = pd.get_dummies(df, columns=["Sex","Pclass","Embarked"])
df.columns

added_cols = ['Sex_male', 'Sex_female', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
df[added_cols].head()

t_dep = tensor(df.Survived)

indep_cols = ['Age', 'SibSp', 'Parch', 'LogFare'] + added_cols

t_indep = tensor(df[indep_cols].values, dtype=torch.float)
t_indep

t_indep.shape

# ## Setting up a linear model
torch.manual_seed(442)

n_coeff = t_indep.shape[1]
coeffs = torch.rand(n_coeff)-0.5
coeffs

t_indep*coeffs

vals,indices = t_indep.max(dim=0)
t_indep = t_indep / vals

t_indep*coeffs

preds = (t_indep*coeffs).sum(axis=1)

preds[:10]

loss = torch.abs(preds-t_dep).mean()
loss

def calc_preds(coeffs, indeps): return (indeps*coeffs).sum(axis=1)
def calc_loss(coeffs, indeps, deps): return torch.abs(calc_preds(coeffs, indeps)-deps).mean()

# ## Doing a gradient descent step
coeffs.requires_grad_()

loss = calc_loss(coeffs, t_indep, t_dep)
loss

loss.backward()

coeffs.grad

loss = calc_loss(coeffs, t_indep, t_dep)
loss.backward()
coeffs.grad

loss = calc_loss(coeffs, t_indep, t_dep)
loss.backward()
with torch.no_grad():
    coeffs.sub_(coeffs.grad * 0.1)
    coeffs.grad.zero_()
    print(calc_loss(coeffs, t_indep, t_dep))

# ## Training the linear model
trn_split,val_split=RandomSplitter(seed=42)(df)

trn_indep,val_indep = t_indep[trn_split],t_indep[val_split]
trn_dep,val_dep = t_dep[trn_split],t_dep[val_split]
len(trn_indep),len(val_indep)

def update_coeffs(coeffs, lr):
    coeffs.sub_(coeffs.grad * lr)
    coeffs.grad.zero_()

def one_epoch(coeffs, lr):
    loss = calc_loss(coeffs, trn_indep, trn_dep)
    loss.backward()
    with torch.no_grad(): update_coeffs(coeffs, lr)
    print(f"{loss:.3f}", end="; ")

def init_coeffs(): return (torch.rand(n_coeff)-0.5).requires_grad_()

def train_model(epochs=30, lr=0.01):
    torch.manual_seed(442)
    coeffs = init_coeffs()
    for i in range(epochs): one_epoch(coeffs, lr=lr)
    return coeffs

coeffs = train_model(18, lr=0.2)

def show_coeffs(): return dict(zip(indep_cols, coeffs.requires_grad_(False)))
show_coeffs()

# ## Measuring accuracy
preds = calc_preds(coeffs, val_indep)

results = val_dep.bool()==(preds>0.5)
results[:16]

results.float().mean()

def acc(coeffs): return (val_dep.bool()==(calc_preds(coeffs, val_indep)>0.5)).float().mean()
acc(coeffs)

# ## Using sigmoid
preds[:28]

sympy.plot("1/(1+exp(-x))", xlim=(-5,5));

def calc_preds(coeffs, indeps): return torch.sigmoid((indeps*coeffs).sum(axis=1))

coeffs = train_model(lr=100)

acc(coeffs)

show_coeffs()

# ## Submitting to Kaggle
tst_df = pd.read_csv(path/'test.csv')

tst_df['Fare'] = tst_df.Fare.fillna(0)

tst_df.fillna(modes, inplace=True)
tst_df['LogFare'] = np.log(tst_df['Fare']+1)
tst_df = pd.get_dummies(tst_df, columns=["Sex","Pclass","Embarked"])

tst_indep = tensor(tst_df[indep_cols].values, dtype=torch.float)
tst_indep = tst_indep / vals

tst_df['Survived'] = (calc_preds(tst_indep, coeffs)>0.5).int()

sub_df = tst_df[['PassengerId','Survived']]
sub_df.to_csv('sub.csv', index=False)

subprocess.run(['head', "sub.csv"], check=True)

# ## Using matrix product
(val_indep*coeffs).sum(axis=1)

val_indep@coeffs

def calc_preds(coeffs, indeps): return torch.sigmoid(indeps@coeffs)

def init_coeffs(): return (torch.rand(n_coeff, 1)*0.1).requires_grad_()

trn_dep = trn_dep[:,None]
val_dep = val_dep[:,None]

coeffs = train_model(lr=100)

acc(coeffs)

# ## A neural network
def init_coeffs(n_hidden=20):
    layer1 = (torch.rand(n_coeff, n_hidden)-0.5)/n_hidden
    layer2 = torch.rand(n_hidden, 1)-0.3
    const = torch.rand(1)[0]
    return layer1.requires_grad_(),layer2.requires_grad_(),const.requires_grad_()

def calc_preds(coeffs, indeps):
    l1,l2,const = coeffs
    res = F.relu(indeps@l1)
    res = res@l2 + const
    return torch.sigmoid(res)

def update_coeffs(coeffs, lr):
    for layer in coeffs:
        layer.sub_(layer.grad * lr)
        layer.grad.zero_()

coeffs = train_model(lr=1.4)

coeffs = train_model(lr=20)

acc(coeffs)

# ## Deep learning
def init_coeffs():
    hiddens = [10, 10]  # <-- set this to the size of each hidden layer you want
    sizes = [n_coeff] + hiddens + [1]
    n = len(sizes)
    layers = [(torch.rand(sizes[i], sizes[i+1])-0.3)/sizes[i+1]*4 for i in range(n-1)]
    consts = [(torch.rand(1)[0]-0.5)*0.1 for i in range(n-1)]
    for l in layers+consts: l.requires_grad_()
    return layers,consts

def calc_preds(coeffs, indeps):
    layers,consts = coeffs
    n = len(layers)
    res = indeps
    for i,l in enumerate(layers):
        res = res@l + consts[i]
        if i!=n-1: res = F.relu(res)
    return torch.sigmoid(res)

def update_coeffs(coeffs, lr):
    layers,consts = coeffs
    for layer in layers+consts:
        layer.sub_(layer.grad * lr)
        layer.grad.zero_()

coeffs = train_model(lr=4)

acc(coeffs)

# ## Final thoughts
