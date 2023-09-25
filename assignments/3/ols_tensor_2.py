import csv
import torch

with open('temp_co2_data.csv') as data_file:
    data = list(csv.reader(data_file))[1:]
    x_features = torch.tensor([[float(item) for item in line[2:]] for line in data])
    y_features = torch.tensor([[float(line[1])] for line in data])

design_matrix = torch.tensor([[1, *line] for line in x_features])
weights_linear_algebra = torch.linalg.lstsq(design_matrix, y_features, driver='gels').solution[:, 0]

x_mean = x_features.mean(0)
y_mean = y_features.mean()

x_standard_deviation = x_features.std(0)
y_standard_deviation = y_features.std()

x_features = (x_features - x_mean) / x_standard_deviation
y_features = (y_features - y_mean) / y_standard_deviation

design_matrix = torch.tensor([[1, *line] for line in x_features])
weights = torch.rand(3, 1) - 0.5 * torch.ones(3, 1)

learning_rate = 0.5
epochs = 30

for epoch in range(epochs):
    target_estimates = design_matrix.mm(weights)
    print(weights)
    print(target_estimates)

    loss = (target_estimates - y_features).pow(2).sum() / design_matrix.size(0)
    print(f'epoch: {epoch + 1}, current loss: {loss.item()}')

    gradient = 2 * ((target_estimates - y_features) * design_matrix).sum(0, True).t() / design_matrix.size(0)
    weights -= learning_rate * gradient

weights = weights.squeeze(1)
weights[1:] = weights[1:] * y_standard_deviation / x_standard_deviation
weights[0] = weights[0] * y_standard_deviation + y_mean - weights[1:] @ x_mean

print(
    f'The least-squares regression plane:\n'
    f'\tfound by the neural net is: y = {weights[0]:.3f} + {weights[1]:.3f} * x1 + {weights[2]:.3f} * x2\n'
    f'\tusing linear algebra:\t\ty = {weights_linear_algebra[0]:.3f} + {weights_linear_algebra[1]:.3f} * x1 + '
    f'{weights_linear_algebra[2]:.3f} * x2\n'
    f'Learning rate: {learning_rate}\n'
)
