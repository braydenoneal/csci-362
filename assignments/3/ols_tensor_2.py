import csv
import torch

data = list(csv.reader(open('temp_co2_data.csv')))[1:]
in_features = torch.tensor([[float(line[2]), float(line[3])] for line in data])
out_features = torch.tensor([[float(line[1])] for line in data])

design_matrix = torch.tensor([[1, *line] for line in in_features])
weights_linear_algebra = torch.linalg.lstsq(design_matrix, out_features, driver='gels').solution[:, 0]

x_mean = in_features.mean(0)
y_mean = out_features.mean()

in_features -= x_mean
out_features -= y_mean

in_features = torch.tensor([[1, *line] for line in in_features])
weights = torch.rand(3, 1) - 0.5

learning_rate = 4e-3
epochs = 100_000

for epoch in range(epochs):
    target_estimates = in_features.mm(weights)

    loss = (target_estimates - out_features).pow(2).sum() / in_features.size(0)

    gradient = 2 * ((target_estimates - out_features) * in_features).sum(0, True).t() / in_features.size(0)
    weights -= learning_rate * gradient

weights = weights.squeeze(1)
weights[0] = weights[0] + y_mean - weights[1:] @ x_mean

print(
    f'The least-squares regression plane:\n'
    f'\tfound by the neural net is: y = {weights[0]:.3f} + {weights[1]:.3f} * x1 + {weights[2]:.3f} * x2\n'
    f'\tusing linear algebra:\t\ty = {weights_linear_algebra[0]:.3f} + {weights_linear_algebra[1]:.3f} * x1 + '
    f'{weights_linear_algebra[2]:.3f} * x2\n'
    f'Learning rate: {learning_rate}\n'
)
