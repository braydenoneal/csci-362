# Assignment 11: Convolutional Neural Network

```python
import torch
import torch.nn as nn
from skimage import io
import du.lib as dulib
import math

train_amount = 0.8
learning_rate = 0.1
momentum = 0.53
epochs = 96
batch_size = 32
centered = True
normalized = True

digits = io.imread('digits.png')
xss = torch.Tensor(5000, 20, 20)
xss_index = 0

for i in range(0, 1000, 20):
    for j in range(0, 2000, 20):
        xss[xss_index] = torch.Tensor((digits[i:i + 20, j:j + 20]))
        xss_index = xss_index + 1

yss = torch.LongTensor(len(xss))
for i in range(len(yss)):
    yss[i] = i // 500

random_split = torch.randperm(xss.size(0))
train_split_amount = math.floor(xss.size(0) * train_amount)

xss_train = xss[random_split][:train_split_amount]
xss_test = xss[random_split][train_split_amount:]

if centered:
    xss_train, xss_train_means = dulib.center(xss_train)
    xss_test, _ = dulib.center(xss_test, xss_train_means)

if normalized:
    xss_train, xss_train_stds = dulib.normalize(xss_train)
    xss_test, _ = dulib.normalize(xss_test, xss_train_stds)

yss_train = yss[random_split][:train_split_amount]
yss_test = yss[random_split][train_split_amount:]


class ConvolutionalModel(nn.Module):
    def __init__(self):
        super(ConvolutionalModel, self).__init__()
        self.meta_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.meta_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.fc_layer1 = nn.Linear(800, 200)
        self.fc_layer2 = nn.Linear(200, 10)

    def forward(self, forward_xss):
        forward_xss = torch.unsqueeze(forward_xss, dim=1)
        forward_xss = self.meta_layer1(forward_xss)
        forward_xss = self.meta_layer2(forward_xss)
        forward_xss = torch.reshape(forward_xss, (-1, 800))
        forward_xss = self.fc_layer1(forward_xss)
        forward_xss = self.fc_layer2(forward_xss)
        return torch.log_softmax(forward_xss, dim=1)


model = ConvolutionalModel()
criterion = nn.NLLLoss()

model, valids = dulib.cv_train(
    model,
    crit=criterion,
    train_data=(xss_train, yss_train),
    learn_params={'lr': learning_rate, 'mo': momentum},
    epochs=epochs,
    bs=batch_size,
    verb=10,
    k=10,
    bail_after=30,
)

print('\nTraining Data Confusion Matrix\n')
pct_training = dulib.class_accuracy(model, (xss_train, yss_train), show_cm=True)

print('\nTesting Data Confusion Matrix\n')
pct_testing = dulib.class_accuracy(model, (xss_test, yss_test), show_cm=True)

print(
    f'\n'
    f'Percentage correct on training data: {100 * pct_training:.2f}\n'
    f'Percentage correct on testing data: {100 * pct_testing:.2f}\n'
    f'\n'
    f'Valids: {valids}\n'
    f'Train Amount: {train_amount * 100}%\n'
    f'Learning Rate: {learning_rate}\n'
    f'Momentum: {momentum}\n'
    f'Epochs: {epochs}\n'
    f'Batch Size: {batch_size}\n'
    f'Centered: {centered}\n'
    f'Normalized: {normalized}'
)
```

## Output

Using two meta layers and cross-validation the model trains to around 96% to 98% accuracy.

```
Testing Data Confusion Matrix

                          Actual
         0    1    2    3    4    5    6    7    8    9 (correct)
     --------------------------------------------------
  0 | 10.2    0    0    0    0    0   .1    0    0    0 (100.0% of 102)
  1 |    0 11.6    0    0    0    0    0    0    0   .1 (98.3% of 118)
  2 |    0   .2 10.4    0    0    0    0   .1   .1    0 (98.1% of 106)
  3 |    0    0    0  9.8    0    0    0    0    0   .2 (99.0% of 99)
  4 |    0    0    0    0  9.4    0    0    0    0    0 (98.9% of 95)
  5 |    0    0    0   .1    0  9.7    0    0    0    0 (99.0% of 98)
  6 |    0    0    0    0   .1   .1 11.2    0   .1    0 (99.1% of 113)
  7 |    0    0   .1    0    0    0    0  8.8    0   .1 (97.8% of 90)
  8 |    0    0    0    0    0    0    0    0  8.6   .1 (97.7% of 88)
  9 |    0    0   .1    0    0    0    0   .1    0  8.6 (94.5% of 91)

Percentage correct on training data: 100.00
Percentage correct on testing data: 98.30

Valids: 0.0
Train Amount: 80.0%
Learning Rate: 0.1
Momentum: 0.53
Epochs: 96
Batch Size: 32
Centered: True
Normalized: True
```
