import csv
import torch
import numpy as np

data = list(csv.reader(open('temp_co2_data.csv')))[1:]
in_features = torch.tensor([[float(line[2]), float(line[3])] for line in data])
out_features = torch.tensor([[float(line[1])] for line in data])

in_features_mean = in_features.mean(0)
out_features_mean = out_features.mean()

in_features = in_features - in_features_mean
out_features = out_features - out_features_mean

in_features = torch.tensor([[1, *line] for line in in_features])
initial_weights = torch.rand(3, 1) - 0.5

start = 0.0
end = 0.1
steps = 1_000
step = (end - start) / steps

learning_rates = torch.arange(start, end + step, step)
epochs = 10_000

losses = []

for learning_rate in learning_rates:
    current_weights = initial_weights.clone().detach()
    loss = 0

    for epoch in range(epochs):
        target_estimates = in_features.mm(current_weights)

        loss = ((target_estimates - out_features).pow(2).sum() / in_features.size(0)).item()

        gradient = 2 * ((target_estimates - out_features) * in_features).sum(0, True).t() / in_features.size(0)
        current_weights -= learning_rate * gradient

    if not np.isnan(loss) and not np.isinf(loss):
        losses.append([learning_rate, loss, current_weights])

losses.sort(key=lambda line: line[1], reverse=True)

weights = losses[-1][2].squeeze(1)
weights[0] = weights[0] + out_features_mean - weights[1:] @ in_features_mean

print(
    f'The least-squares regression plane:\n'
    f'\tfound by the neural net is: y = {weights[0]:.3f} + {weights[1]:.3f} * x1 + {weights[2]:.3f} * x2\n'
    f'Learning rate: {losses[-1][0].item():>0.4f}\n'
    f'Final Loss: {losses[-1][1]:>0.4f}\n'
)
