import csv
import torch
import numpy as np

data = list(csv.reader(open('temp_co2_data.csv')))[1:]
in_features = torch.tensor([[float(item) for item in line[2:]] for line in data])
out_features = torch.tensor([[float(line[1])] for line in data])

x_mean = in_features.mean(0)
y_mean = out_features.mean()

in_features = in_features - x_mean
out_features = out_features - y_mean

in_features = torch.tensor([[1, *line] for line in in_features])
weights = torch.rand(3, 1) - 0.5

start = 0
end = .1
steps = 1_000
step = (end - start) / steps

learning_rates = torch.arange(start, end + step, step)
epoch_sizes = torch.arange(10_000, 10_001, 1)

loss_data = []

for learning_rate in learning_rates:
    for epoch_size in epoch_sizes:
        print(f'Learning Rate: {learning_rate}')
        print(f'Epoch Size: {epoch_size}')

        current_weights = weights.clone().detach()
        loss = 0

        for epoch in range(epoch_size):
            target_estimates = in_features.mm(current_weights)

            loss = ((target_estimates - out_features).pow(2).sum() / in_features.size(0)).item()

            gradient = 2 * ((target_estimates - out_features) * in_features).sum(0, True).t() / in_features.size(0)
            current_weights -= learning_rate * gradient

        if not np.isnan(loss) and not np.isinf(loss):
            loss_data.append([learning_rate, epoch_size, loss, current_weights])

loss_data.sort(key=lambda line: line[2], reverse=True)

for data in loss_data:
    print(f'Learning Rate: {data[0].item()}  Epoch Size: {data[1].item()}  Loss: {data[2]}\n')
