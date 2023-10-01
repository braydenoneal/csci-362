import csv
import numpy as np
import torch


class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.layer = torch.nn.Linear(2, 1)

    def forward(self, features):
        return self.layer(features)


with open('temp_co2_data.csv') as data_file:
    data = list(csv.reader(data_file))[1:]
    in_features = torch.tensor([[float(line[2]), float(line[3])] for line in data])
    out_features = torch.tensor([[float(line[1])] for line in data])

design_matrix = torch.cat((torch.ones(len(in_features), 1), in_features), 1)
weights_linear_algebra = torch.linalg.lstsq(design_matrix, out_features, driver='gels').solution[:, 0]

in_features_means = in_features.mean(0)
out_features_mean = out_features.mean()

in_features_standard_deviations = in_features.std(0)
out_features_standard_deviation = out_features.std()

in_features = (in_features - in_features_means) / in_features_standard_deviations
out_features = (out_features - out_features_mean) / out_features_standard_deviation

model = LinearRegressionModel()
criterion = torch.nn.MSELoss()

print(f'The model is:\n{model}')

learning_rate = 0.5
epochs = 30

features_size = in_features.size(0)

batch_size = 32

for epoch in range(epochs):
    in_features_permuted = in_features[torch.randperm(features_size)]
    out_features_permuted = out_features[torch.randperm(features_size)]

    current_total_loss = 0

    for batch in np.arange(0, features_size, batch_size):
        in_features_batch = in_features_permuted[batch:batch + batch_size]
        out_features_batch = out_features_permuted[batch:batch + batch_size]

        out_features_prediction = model.forward(in_features_batch)

        loss = criterion(out_features_prediction, out_features_batch)

        current_total_loss += loss.item()

        model.zero_grad()
        loss.backward()

        for param in model.parameters():
            param.data.sub_(param.grad.data * learning_rate)

    print(f'epoch: {epoch + 1}, current loss: {current_total_loss * batch_size / len(in_features)}')

parameters = list(model.parameters())

weights = torch.zeros(3)
weights[1:] = parameters[0] * out_features_standard_deviation / in_features_standard_deviations
weights[0] = (parameters[1].data.item() * out_features_standard_deviation + out_features_mean - weights[1:]
              @ in_features_means)

print(
    f'The least-squares regression plane:\n'
    f'\tfound by the neural net is: y = {weights[0]:.3f} + {weights[1]:.3f} * x1 + {weights[2]:.3f} * x2\n'
    f'\tusing linear algebra:\t\ty = {weights_linear_algebra[0]:.3f} + {weights_linear_algebra[1]:.3f} * x1 + '
    f'{weights_linear_algebra[2]:.3f} * x2\n'
    f'Learning rate: {learning_rate}\n'
)
