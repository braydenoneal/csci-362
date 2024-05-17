# Assignment 4: PyTorch Module

## Source

```python
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

    # split xs and ys into chunks of size batch_size
    x_feature_batches = torch.split(x_features[random_indices], batch_size)
    y_feature_batches = torch.split(y_features[random_indices], batch_size)

    current_total_loss = 0

    # loop through x and y chunks and calculate loss
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
```

## Output

```
The model is:
LinearRegressionModel(
  (layer): Linear(in_features=2, out_features=1, bias=True)
)
epoch: 1, current loss: 1.5336381699889898
epoch: 2, current loss: 0.4908260842785239
epoch: 3, current loss: 1.2391103208065033
epoch: 4, current loss: 0.48017595894634724
epoch: 5, current loss: 0.489103801548481
epoch: 6, current loss: 0.6865882091224194
epoch: 7, current loss: 1.1504546515643597
epoch: 8, current loss: 0.3644995605573058
epoch: 9, current loss: 0.40189870446920395
epoch: 10, current loss: 1.191077671945095
epoch: 11, current loss: 0.46276230178773403
epoch: 12, current loss: 0.578103382140398
epoch: 13, current loss: 0.4130382817238569
epoch: 14, current loss: 0.40075053507462144
epoch: 15, current loss: 0.5746108815073967
epoch: 16, current loss: 0.7306371442973614
epoch: 17, current loss: 0.5283911675214767
epoch: 18, current loss: 0.43469279911369085
epoch: 19, current loss: 0.42683455999940634
epoch: 20, current loss: 0.3636915497481823
epoch: 21, current loss: 1.3032681290060282
epoch: 22, current loss: 0.8853312153369188
epoch: 23, current loss: 2.1973255295306444
epoch: 24, current loss: 2.5954504534602165
epoch: 25, current loss: 0.8971390910446644
epoch: 26, current loss: 0.8876387421041727
epoch: 27, current loss: 0.6627807393670082
epoch: 28, current loss: 0.8299236763268709
epoch: 29, current loss: 3.105827309191227
epoch: 30, current loss: 0.5501606408506632
The least-squares regression plane:
	found by the neural net is: y = -26645.848 + 1.154 * x1 + 19.216 * x2
	using linear algebra:		y = -11371.969 + 1.147 * x1 + 8.047 * x2
Learning rate: 0.5
```
