<a id="top"></a>

# Assignment 2

[github.com/braydenoneal/csci-362/tree/master/assignments/2](https://github.com/braydenoneal/csci-362/tree/master/assignments/2#top)

## Source

```python
import torch
import csv
import numpy as np
import matplotlib.pyplot as plt

with open('assign2.csv') as data_file:
    reader = csv.reader(data_file, delimiter=',')
    next(data_file)
    x_values = []
    y_values = []
    for line in reader:
        x_values.append(float(line[0]))
        y_values.append(float(line[1]))

x_values = torch.tensor(x_values)
y_values = torch.tensor(y_values)

# TODO: calc w

plt.scatter(x_values, y_values)
plt.show()
```

## Output
                                                             
```
```
