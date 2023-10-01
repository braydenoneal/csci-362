import csv
import torch


class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.layer = torch.nn.Linear(2, 1)

    def forward(self, features):
        return self.layer(features)


with open('temp_co2_data.csv') as data_file:
    data = list(csv.reader(data_file))[1:]
    x_features = torch.tensor([[float(line[2]), float(line[3])] for line in data])
    y_features = torch.tensor([[float(line[1])] for line in data])

weights_linear_algebra = torch.linalg.lstsq(
    torch.tensor([[1, *line] for line in x_features]), y_features, driver='gels'
).solution[:, 0]

x_features_means = x_features.mean(0)
y_features_mean = y_features.mean()

x_features_standard_deviations = x_features.std(0)
y_features_standard_deviation = y_features.std()

x_features = (x_features - x_features_means) / x_features_standard_deviations
y_features = (y_features - y_features_mean) / y_features_standard_deviation

model = LinearRegressionModel()
criterion = torch.nn.MSELoss()

print(f'The model is:\n{model}')

learning_rate = 0.5
epochs = 30
batch_size = 4

for epoch in range(epochs):
    random_indices = torch.randperm(x_features.size(0))

    x_feature_batches = torch.split(x_features[random_indices], batch_size)
    y_feature_batches = torch.split(y_features[random_indices], batch_size)

    current_total_loss = 0

    for x_features_batch, y_features_batch in zip(x_feature_batches, y_feature_batches):
        loss = criterion(model.forward(x_features_batch), y_features_batch)

        current_total_loss += loss.item()

        model.zero_grad()
        loss.backward()

        for parameter in model.parameters():
            parameter.data.sub_(parameter.grad.data * learning_rate)

    print(f'epoch: {epoch + 1}, current loss: {current_total_loss * batch_size / x_features.size(0)}')

parameters = list(model.parameters())

weights = torch.zeros(3)
weights[1:] = parameters[0] * y_features_standard_deviation / x_features_standard_deviations
weights[0] = (parameters[1].data.item() * y_features_standard_deviation + y_features_mean - weights[1:]
              @ x_features_means)

print(
    f'The least-squares regression plane:\n'
    f'\tfound by the neural net is: y = {weights[0]:.3f} + {weights[1]:.3f} * x1 + {weights[2]:.3f} * x2\n'
    f'\tusing linear algebra:\t\ty = {weights_linear_algebra[0]:.3f} + {weights_linear_algebra[1]:.3f} * x1 + '
    f'{weights_linear_algebra[2]:.3f} * x2\n'
    f'Learning rate: {learning_rate}\n'
)
