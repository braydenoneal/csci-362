import csv
import torch
import torch.nn as nn


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.layer = nn.Linear(2, 1)

    def forward(self, features):
        return self.layer(features)


data = list(csv.reader(open('temp_co2_data.csv')))[1:]
in_features = torch.tensor([[float(line[2]), float(line[3])] for line in data])
out_features = torch.tensor([[float(line[1])] for line in data])

design_matrix = torch.cat((torch.ones(len(in_features), 1), in_features), 1)
weights_linear_algebra = torch.linalg.lstsq(design_matrix, out_features, driver='gels').solution[:, 0]

in_features_means = in_features.mean(0)
out_features_mean = out_features.mean()

in_features_standard_deviations = in_features.std(0)
out_features_standard_deviation = out_features.std()

in_features = in_features - in_features_means / in_features_standard_deviations
out_features = out_features - out_features_mean / out_features_standard_deviation

model = LinearRegressionModel()
criterion = nn.MSELoss()
print('The model is:\n', model)

learning_rate = 0.5
epochs = 30

""" stochasticity: float between 0 and 1 """
stochasticity = 0.5
batch_size = int(len(in_features) * stochasticity)

for epoch in range(epochs):
    indices = torch.randperm(len(in_features))

    current_total_loss = 0

    batch_xss = []
    batch_yss = []

    for index in indices[:batch_size]:
        batch_xss.append([item.item() for item in in_features[index]])
        batch_yss.append([out_features[index].item()])

    batch_xss = torch.tensor(batch_xss)
    batch_yss = torch.tensor(batch_yss)

    yss_pred = model(batch_xss)

    loss = criterion(yss_pred, batch_yss)

    """ accum_loss * batch_size / num_examples """
    print(f'epoch: {epoch + 1}, current loss: {loss.item()}')

    model.zero_grad()
    loss.backward()

    for param in model.parameters():
        param.data.sub_(param.grad.data * learning_rate)

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
