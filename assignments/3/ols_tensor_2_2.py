import csv
import torch
import matplotlib.pyplot as plt
import numpy as np

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
epochs = 3000

all_weights = []
all_losses = []

loss = 0

for epoch in range(epochs):
    target_estimates = design_matrix.mm(weights)

    prev_loss = loss
    loss = (target_estimates - y_features).pow(2).sum() / design_matrix.size(0)
    print(f'epoch: {epoch + 1}, current loss: {loss.item()}')

    gradient = 2 * ((target_estimates - y_features) * design_matrix).sum(0, True).t() / design_matrix.size(0)
    weights -= learning_rate * gradient
    w_diff = torch.clone(weights).squeeze(1)
    w_diff[1:] = w_diff[1:] * y_standard_deviation / x_standard_deviation
    w_diff[0] = w_diff[0] * y_standard_deviation + y_mean - w_diff[1:] @ x_mean
    all_weights.append([i * 100000 for i in [*w_diff]])
    all_losses.append(loss * 1000 - prev_loss * 1000)

all_weights = torch.tensor(all_weights)

figure, axis = plt.subplots(2, 2)

axis[0, 0].plot(all_weights[:, 0], all_losses)
axis[0, 1].plot(all_weights[:, 1], all_losses)
axis[1, 0].plot(all_weights[:, 2], all_losses)

plt.show()

with open("output.csv", "w") as writefile:
    writer = csv.writer(writefile)
    writer.writerows([[*[j.item() for j in all_weights[i]], all_losses[i].item()] for i in range(len(all_losses))])

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
