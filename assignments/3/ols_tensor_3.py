import csv
import torch
import numpy as np

with open('temp_co2_data.csv') as data_file:
    data = list(csv.reader(data_file))[1:]
    x_features = torch.tensor([[float(item) for item in line[2:]] for line in data])
    y_features = torch.tensor([[float(line[1])] for line in data])

x_mean = x_features.mean(0)
y_mean = y_features.mean()

x_standard_deviation = x_features.std(0)
y_standard_deviation = y_features.std()

x_features = x_features - x_mean
y_features = y_features - y_mean

design_matrix = torch.tensor([[1, *line] for line in x_features])
weights = torch.rand(3, 1) - 0.5 * torch.ones(3, 1)

start = 0.003
end = 0.005
steps = 1000
step = (end - start) / steps
learning_rates = torch.arange(start, end + step, step)
epoch_sizes = torch.arange(3000, 3001, 1)

loss_data = []

for learning_rate in learning_rates:
    for epoch_size in epoch_sizes:
        current_weights = weights.clone().detach()
        loss = 0

        for epoch in range(epoch_size):
            target_estimates = design_matrix.mm(current_weights)

            loss = ((target_estimates - y_features).pow(2).sum() / design_matrix.size(0)).item()

            gradient = 2 * ((target_estimates - y_features) * design_matrix).sum(0, True).t() / design_matrix.size(0)
            current_weights -= learning_rate * gradient

        if not np.isnan(loss) and not np.isinf(loss):
            loss_data.append([learning_rate, epoch_size, loss, current_weights])


loss_data.sort(key=lambda line: line[2], reverse=True)

for data in loss_data:
    print(f'Learning Rate: {data[0].item()} '
          f'Epoch Size: {data[1].item()} '
          f'Loss: {data[2]}'
          f' Weights: {data[3]}\n')

# weights = weights.squeeze(1)
# weights[1:] = weights[1:] * y_standard_deviation / x_standard_deviation
# weights[0] = weights[0] * y_standard_deviation + y_mean - weights[1:] @ x_mean
#
# print(f'y = {weights[0]:.3f} + {weights[1]:.3f} * x1 + {weights[2]:.3f} * x2\n')
