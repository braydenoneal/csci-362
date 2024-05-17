# Assignment 5: Momentum

## Source

```python
# ols.py                                                     SSimmons March 2018
#                                                    Brayden O'Neal October 2023
"""
Uses a neural net to find the ordinary least-squares regression model. Trains
with batch gradient descent, and computes r^2 to gauge predictive quality.
Implements momentum.
"""

import torch
import pandas as pd
import torch.nn as nn

# Read the named columns from the csv file into a dataframe.
names = ['SalePrice', '1st_Flr_SF', '2nd_Flr_SF', 'Lot_Area', 'Overall_Qual',
         'Overall_Cond', 'Year_Built', 'Year_Remod/Add', 'BsmtFin_SF_1', 'Total_Bsmt_SF',
         'Gr_Liv_Area', 'TotRms_AbvGrd', 'Bsmt_Unf_SF', 'Full_Bath']
df = pd.read_csv('AmesHousing.csv', names=names)
data = df.values  # read data into a numpy array (as a list of lists)
data = data[1:]  # remove the first list which consists of the labels
data = data.astype(float)  # coerce the entries in the numpy array to floats
data = torch.FloatTensor(data)  # convert data to a Torch tensor

data.sub_(data.mean(0))  # mean-center
data.div_(data.std(0))  # normalize

xss = data[:, 1:]
yss = data[:, :1]


# define a model class
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.layer1 = nn.Linear(13, 1)

    def forward(self, values):
        return self.layer1(values)


# create and print an instance of the model class
model = LinearModel()
print(model)

# Create momentum weights
z_parameters = []
for param in model.parameters():
    z_parameters.append(param.data.clone())
for param in z_parameters:
    param.zero_()

criterion = nn.MSELoss()

num_examples = len(data)
batch_size = 30
learning_rate = 0.000355
momentum = 0.899
epochs = 1000

# train the model
for epoch in range(epochs):
    random_indices = torch.randperm(xss.size(0))

    x_feature_batches = torch.split(xss[random_indices], batch_size)
    y_feature_batches = torch.split(yss[random_indices], batch_size)

    current_total_loss = 0

    for x_features_batch, y_features_batch in zip(x_feature_batches, y_feature_batches):
        loss = criterion(model.forward(x_features_batch), y_features_batch)

        current_total_loss += loss.item()

        model.zero_grad()
        loss.backward()

        # Adjust the weights with momentum
        for i, (z_param, param) in enumerate(zip(z_parameters, model.parameters())):
            z_parameters[i] = momentum * z_param + param.grad.data
            param.data.sub_(z_parameters[i] * learning_rate)

    print_str = f'epoch: {epoch + 1}, loss: {current_total_loss * batch_size / xss.size(0):11.8f}'

    if epoch < 8 or epoch > epochs - 8:
        print(print_str)
    if epoch == 8:
        print('...')
    if epoch > 8:
        print(print_str, end='\b' * len(print_str))

print('total number of examples:', num_examples, end='; ')
print('batch size:', batch_size)
print('learning rate:', learning_rate)
print('momentum:', momentum)

# Compute 1-SSE/SST which is the proportion of the variance in the data
# explained by the regression hyperplane.
SS_E = 0.0
SS_T = 0.0
mean = yss.mean()

for xs, ys in zip(xss, yss):
    SS_E = SS_E + (ys - model(xs)) ** 2
    SS_T = SS_T + (ys - mean) ** 2

print(f'1-SSE/SST = {1.0 - (SS_E / SS_T).item():1.4f}')
```

## Output

```
LinearModel(
  (layer1): Linear(in_features=13, out_features=1, bias=True)
)
epoch: 1, loss:  0.36088991
epoch: 2, loss:  0.19327251
epoch: 3, loss:  0.16635019
epoch: 4, loss:  0.15532728
epoch: 5, loss:  0.15029621
epoch: 6, loss:  0.14718570
epoch: 7, loss:  0.14415080
epoch: 8, loss:  0.14320454
...
epoch: 994, loss:  0.13919826
epoch: 995, loss:  0.13839924
epoch: 996, loss:  0.13847842
epoch: 997, loss:  0.13834132
epoch: 998, loss:  0.13960692
epoch: 999, loss:  0.14066740
epoch: 1000, loss:  0.13919200
total number of examples: 2264; batch size: 30
learning rate: 0.000355
momentum: 0.899
1-SSE/SST = 0.8628
```
