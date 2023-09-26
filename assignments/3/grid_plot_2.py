import numpy as np
import csv
import torch
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

data = list(csv.reader(open('temp_co2_data.csv')))[1:]
in_features = torch.tensor([[float(item) for item in line[2:]] for line in data])
out_features = torch.tensor([[float(line[1])] for line in data])

design_matrix = torch.tensor([[1, *line] for line in in_features])
weights_linear_algebra = torch.linalg.lstsq(design_matrix, out_features, driver='gels').solution[:, 0]

in_features_mean = in_features.mean(0)
out_features_mean = out_features.mean()

in_feature_standard_deviation = in_features.std(0)
out_features_standard_deviation = out_features.std()

in_features = (in_features - in_features_mean) / in_feature_standard_deviation
out_features = (out_features - out_features_mean) / out_features_standard_deviation

in_features = torch.tensor([[1, *line] for line in in_features])
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
            weights = torch.FloatTensor([[x], [y], [z]])
            target_estimates = in_features.mm(weights)
            all_losses.append((target_estimates - out_features).pow(2).sum() / in_features.size(0))
            all_weights.append([i for i in [*weights.squeeze(1)]])

all_weights = torch.tensor(all_weights)

fig = plt.figure(dpi=200)
ax = plt.axes(projection='3d')

ax.scatter(
    all_weights[:, 1],
    all_weights[:, 2],
    all_weights[:, 0],
    c=[np.nan if i > 0.5 else i for i in all_losses],
    cmap=cm.turbo
)

ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('w0')

plt.show()
