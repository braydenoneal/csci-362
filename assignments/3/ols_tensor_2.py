import csv
import torch

with open('temp_co2_data.csv') as csvfile:
    reader = [[float(item) for item in line] for line in list(csv.reader(csvfile, delimiter=',')).copy()[1:]]
    x_values = torch.tensor([line[2:] for line in reader])
    y_values = torch.tensor([[line[1]] for line in reader])

design_matrix = torch.cat((torch.ones(len(x_values), 1), x_values), 1)
weights_linear_algebra = torch.linalg.lstsq(design_matrix, y_values, driver='gels').solution[:, 0]

x_mean = x_values.mean(0)
y_mean = y_values.mean()

x_standard_deviation = x_values.std(0)
y_standard_deviation = y_values.std()

x_values = (x_values - x_mean) / x_standard_deviation
y_values = (y_values - y_mean) / y_standard_deviation

design_matrix = torch.cat((torch.ones(len(x_values), 1), x_values), 1)

weights = torch.rand(3, 1) - 0.5 * torch.ones(3, 1)

learning_rate = 0.5
epochs = 30

for epoch in range(epochs):
    target_estimates = design_matrix.mm(weights)

    loss = (target_estimates - y_values).pow(2).sum() / len(design_matrix)
    print(f"epoch: {epoch + 1}, current loss: {loss.item()}")

    gradient = 2 * ((target_estimates - y_values) * design_matrix).sum(0, True).t() / len(design_matrix)

    weights = weights - learning_rate * gradient

weights = weights.squeeze(1)

weights[1:] = weights[1:] * y_standard_deviation / x_standard_deviation
weights[0] = weights[0] * y_standard_deviation + y_mean - weights[1:] @ x_mean

print(
    f"The least-squares regression plane:\n"
    f"\tfound by the neural net is: y = {weights[0]:.3f} + {weights[1]:.3f} * x1 + {weights[2]:.3f} * x2\n"
    f"\tusing linear algebra:\t\ty = {weights_linear_algebra[0]:.3f} + {weights_linear_algebra[1]:.3f} * x1 + "
    f"{weights_linear_algebra[2]:.3f} * x2\n"
    f"Learning rate: {learning_rate}\n"
)
