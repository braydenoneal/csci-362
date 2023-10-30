# ols.py                                                     SSimmons March 2018
#                                                    Brayden O'Neal October 2023
"""
Uses a neural net to find the ordinary least-squares regression model. Trains
with batch gradient descent, and computes r^2 to gauge predictive quality.
Implements momentum.
"""
import math
import torch
import pandas as pd
import torch.nn as nn
import du.lib as dulib

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

random_split = torch.randperm(xss.size(0))

xss_train = xss[random_split][:(math.floor(xss.size(0) * 0.8))]
xss_test = xss[random_split][(math.floor(xss.size(0) * 0.8)):]

yss_train = yss[random_split][:(math.floor(yss.size(0) * 0.8))]
yss_test = yss[random_split][(math.floor(yss.size(0) * 0.8)):]

# define a model class
class NonLinearModel(nn.Module):
    def __init__(self):
        super(NonLinearModel, self).__init__()
        self.layer1 = nn.Linear(13, 10)
        self.layer2 = nn.Linear(10, 1)

    def forward(self, values):
        values = self.layer1(values)
        values = torch.relu(values)
        return self.layer2(values)


# create and print an instance of the model class
model = NonLinearModel()
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
    random_indices = torch.randperm(xss_train.size(0))

    x_feature_batches = torch.split(xss_train[random_indices], batch_size)
    y_feature_batches = torch.split(yss_train[random_indices], batch_size)

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

    print_str = f'epoch: {epoch + 1}, loss: {current_total_loss * batch_size / xss_train.size(0):11.8f}'

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

print("explained variation:", dulib.explained_var(model, (xss_train, yss_train)))
print("explained variation:", dulib.explained_var(model, (xss_test, yss_test)))
