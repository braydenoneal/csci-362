import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

with open('temp_co2_data.csv') as data_file:
    data = list(csv.reader(data_file))[1:]
    x_features = torch.tensor([[float(item) for item in line[2:]] for line in data])
    y_features = torch.tensor([[float(line[1])] for line in data])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot_surface([x_features[:, 0]], x_features[:, 1], y_features, cmap=cm.coolwarm)
fig.show()
