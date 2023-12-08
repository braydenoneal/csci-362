import torch
import torch.nn as nn
from skimage import io
import du.lib as dulib
import math

train_amount = 0.9
learning_rate = 0.01
momentum = 0.53
epochs = 96
batch_size = 64
centered = True
normalized = True

digits = io.imread('digits.png')
xss = torch.Tensor(5000, 20, 20)
idx = 0
for i in range(0, 1000, 20):
    for j in range(0, 2000, 20):
        xss[idx] = torch.Tensor((digits[i:i + 20, j:j + 20]))
        idx = idx + 1

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

    xss, xss_means = dulib.center(xss)
if normalized:
    xss_train, xss_train_stds = dulib.normalize(xss_train)
    xss_test, _ = dulib.normalize(xss_test, xss_train_stds)

    xss, xss_stds = dulib.normalize(xss)

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


def pct_correct(xss_test_, yss_test_):
    count = 0

    for x, y in zip(xss_test_, yss_test_):
        if torch.argmax(x).item() == y.item():
            count += 1

    return 100 * count / len(xss_test_)


model, valids = dulib.cv_train(
    model,
    crit=criterion,
    train_data=(xss, yss),
    learn_params={'lr': learning_rate, 'mo': momentum},
    epochs=epochs,
    bs=batch_size,
    verb=10,
    k=8,
)

print(model)
print(valids)

print('\nAll Data Confusion Matrix\n')
pct_all = dulib.class_accuracy(model, (xss, yss), show_cm=True)

print('\nTraining Data Confusion Matrix\n')
pct_training = dulib.class_accuracy(model, (xss_train, yss_train), show_cm=True)

print('\nTesting Data Confusion Matrix\n')
pct_testing = dulib.class_accuracy(model, (xss_test, yss_test), show_cm=True)

print(
    f'\n'
    f'Percentage correct on all data: {100 * pct_all:.2f}\n'
    f'Percentage correct on training data: {100 * pct_training:.2f}\n'
    f'Percentage correct on testing data: {100 * pct_testing:.2f}\n'
    f'\n'
    f'Learning Rate: {learning_rate}\n'
    f'Momentum: {momentum}\n'
    f'Epochs: {epochs}\n'
    f'Batch Size: {batch_size}'
)
