"""04-how-does-a-neural-net-really-work

This file was extracted from the corresponding Jupyter notebook.
"""

# ## Fitting a function with *gradient descent*
from fastai.basics import *

plt.rc('figure', dpi=90)

def plot_function(f, title=None, min=-2.1, max=2.1, color='r', ylim=None):
    x = torch.linspace(min,max, 100)[:,None]
    if ylim: plt.ylim(ylim)
    plt.plot(x, f(x), color)
    if title is not None: plt.title(title)

def f(x): return 3*x**2 + 2*x + 1

plot_function(f, "$3x^2 + 2x + 1$")

def quad(a, b, c, x): return a*x**2 + b*x + c

def mk_quad(a,b,c): return partial(quad, a,b,c)

f2 = mk_quad(3,2,1)
plot_function(f2)

def noise(x, scale): return np.random.normal(scale=scale, size=x.shape)
def add_noise(x, mult, add): return x * (1+noise(x,mult)) + noise(x,add)

np.random.seed(42)

x = torch.linspace(-2, 2, steps=20)[:,None]
y = add_noise(f(x), 0.15, 1.5)

x[:5],y[:5]

plt.scatter(x,y);

# Interactive notebook widget omitted for standalone execution.
def plot_quad(a, b, c):
    plt.scatter(x,y)
    plot_function(mk_quad(a,b,c), ylim=(-3,13))

def mae(preds, acts): return (torch.abs(preds-acts)).mean()

# Interactive notebook widget omitted for standalone execution.
def plot_quad(a, b, c):
    f = mk_quad(a,b,c)
    plt.scatter(x,y)
    loss = mae(f(x), y)
    plot_function(f, ylim=(-3,12), title=f"MAE: {loss:.2f}")

# ## Automating gradient descent
def quad_mae(params):
    f = mk_quad(*params)
    return mae(f(x), y)

quad_mae([1.1, 1.1, 1.1])

abc = torch.tensor([1.1,1.1,1.1])

abc.requires_grad_()

loss = quad_mae(abc)
loss

loss.backward()

abc.grad

with torch.no_grad():
    abc -= abc.grad*0.01
    loss = quad_mae(abc)
    
print(f'loss={loss:.2f}')

for i in range(10):
    loss = quad_mae(abc)
    loss.backward()
    with torch.no_grad(): abc -= abc.grad*0.01
    print(f'step={i}; loss={loss:.2f}')

# ## How a neural network approximates any given function
def rectified_linear(m,b,x):
    y = m*x+b
    return torch.clip(y, 0.)

plot_function(partial(rectified_linear, 1,1))

import torch.nn.functional as F
def rectified_linear2(m,b,x): return F.relu(m*x+b)
plot_function(partial(rectified_linear2, 1,1))

# Interactive notebook widget omitted for standalone execution.
def plot_relu(m, b):
    plot_function(partial(rectified_linear, m,b), ylim=(-1,4))

def double_relu(m1,b1,m2,b2,x):
    return rectified_linear(m1,b1,x) + rectified_linear(m2,b2,x)

# Interactive notebook widget omitted for standalone execution.
def plot_double_relu(m1, b1, m2, b2):
    plot_function(partial(double_relu, m1,b1,m2,b2), ylim=(-1,6))

# ## How to recognise an owl
