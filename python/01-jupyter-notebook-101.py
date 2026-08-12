"""Jupyter Notebook 101

This file was extracted from the corresponding Jupyter notebook.
"""

# Introduction
1 + 1

# Writing
3 / 2

# Modes

# Other Important Considerations
import os

print(os.getcwd())

# Markdown Formatting

# Images

# Italics, Bold, Strikethrough, Inline, Blockquotes and Links

# Headings

# Lists

# Code Capabilities
a = 1
b = a + 1
c = b + a + 1
d = c + b + a + 1
a, b, c, d

import matplotlib.pyplot as plt

plt.plot([a, b, c, d])
plt.show()

# Running Jupyter Locally

# Shortcuts and Tricks

# Command Mode Shortcuts

# Cell Tricks
help(print)

# Line Magics
import timeit

timeit.timeit('[i+1 for i in range(1000)]', number=1000)

# Thanks for reading!
