import numpy as np
import csv
import torch
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

with open('temp_co2_data.csv') as data_file:
    data = list(csv.reader(data_file))[1:]
    in_features = torch.tensor([[float(item) for item in line[2:]] for line in data])
    out_features = torch.tensor([[float(line[1])] for line in data])

design_matrix = torch.tensor([[1, *line] for line in in_features])
weights_linear_algebra = torch.linalg.lstsq(design_matrix, out_features, driver='gels').solution[:, 0]

x_mean = in_features.mean(0)
y_mean = out_features.mean()

x_standard_deviation = in_features.std(0)
y_standard_deviation = out_features.std()

in_features = (in_features - x_mean) / x_standard_deviation
out_features = (out_features - y_mean) / y_standard_deviation

design_matrix = torch.tensor([[1, *line] for line in in_features])
weights = torch.rand(3, 1) - 0.5 * torch.ones(3, 1)

learning_rate = 0.5
epochs = 5000

all_weights = []
all_losses = []

start = -1
stop = 1
steps = 32
step = (stop - start) / steps

for x in np.arange(start, stop + step, step):
    for y in np.arange(start, stop + step, step):
        for z in np.arange(start, stop + step, step):
            weights = (torch.rand(3, 1) - 0.5) * 1000 + 0
            weights[0] = x
            weights[1] = y
            weights[2] = z
            weights *= 1
            weights[1] += 0
            target_estimates = design_matrix.mm(weights)

            loss = (target_estimates - out_features).pow(2).sum() / design_matrix.size(0)

            all_weights.append([i for i in [*weights.squeeze(1)]])
            all_losses.append(loss)

for i in range(len(all_losses)):
    if all_losses[i] > 0.5:
        all_losses[i] = np.nan
        all_weights[i][0] = np.nan
        all_weights[i][1] = np.nan
        all_weights[i][2] = np.nan

all_weights = torch.tensor(all_weights)

figure, axis = plt.subplots(1, 2, dpi=200, subplot_kw=dict(projection='3d'))

color = cm.bwr
color.set_under('black')

axis[0].scatter(all_weights[:, 1], all_weights[:, 2], all_weights[:, 0], c=all_losses, cmap=cm.turbo)
# axis[0].scatter(all_weights[:, 1], all_weights[:, 2], all_weights[:, 0], c=all_losses, cmap=color, vmin=0.4, vmax=1.)
axis[0].set_title('Unmodified Data')
axis[0].set_xlabel('w1')
axis[0].set_ylabel('w2')
axis[0].set_zlabel('w0')

axis[1].scatter(all_weights[:, 1], all_weights[:, 2], all_losses, c=all_weights[:, 0], cmap=cm.turbo)
# axis[1].scatter(all_weights[:, 0], all_weights[:, 2], all_losses, c=all_weights[:, 1], cmap=color, vmin=0.1, vmax=1.)
axis[1].set_title('Mean Centered and Normalized')
axis[1].set_xlabel('w0')
axis[1].set_ylabel('w2')
axis[1].set_zlabel('Loss')

axis[1].shareview(axis[0])

plt.show()
