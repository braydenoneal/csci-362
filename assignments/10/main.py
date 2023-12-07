import torch
import torch.nn as nn
from skimage import io
import du.lib as dulib
import math

train_amount = 0.8
learning_rate = 0.01
momentum = 0.75
epochs = 64
batch_size = 32
centered = True
normalized = True
hidden_layer_widths = [200]

digits = io.imread('digits.png')
xss = torch.Tensor(5000, 400)
idx = 0
for i in range(0, 1000, 20):
    for j in range(0, 2000, 20):
        xss[idx] = torch.Tensor((digits[i:i + 20, j:j + 20]).flatten())
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
if normalized:
    xss_train, xss_train_stds = dulib.normalize(xss_train)
    xss_test, _ = dulib.normalize(xss_test, xss_train_stds)

yss_train = yss[random_split][:train_split_amount]
yss_test = yss[random_split][train_split_amount:]


class LogSoftmaxModel(nn.Module):
    def __init__(self):
        super(LogSoftmaxModel, self).__init__()
        widths = hidden_layer_widths.copy()
        widths.insert(0, 400)
        widths.append(10)

        self.in_layer = nn.Linear(400, widths[0])

        hidden_layers = []

        for j in range(len(widths) - 1):
            hidden_layers.append(nn.Linear(widths[j], widths[j + 1]))

        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.out_layer = nn.Linear(widths[-1], 10)

    def forward(self, x):
        x = self.in_layer(x)
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        x = self.out_layer(x)
        return torch.log_softmax(x, dim=1)


model = LogSoftmaxModel()
criterion = nn.NLLLoss()


def pct_correct(xss_test_, yss_test_):
    count = 0

    for x, y in zip(xss_test_, yss_test_):
        if torch.argmax(x).item() == y.item():
            count += 1

    return 100 * count / len(xss_test_)


model = dulib.train(
    model,
    crit=criterion,
    train_data=(xss_train, yss_train),
    valid_data=(xss_test, yss_test),
    learn_params={'lr': learning_rate, 'mo': momentum},
    epochs=epochs,
    bs=batch_size,
    valid_metric=pct_correct,
    graph=1,
    print_lines=(-1,)
)

print('\nTraining Data Confusion Matrix\n')
pct_training = dulib.class_accuracy(model, (xss_train, yss_train), show_cm=True)

print('\nTesting Data Confusion Matrix\n')
pct_testing = dulib.class_accuracy(model, (xss_test, yss_test), show_cm=True)

weights = 401 * hidden_layer_widths[0]

for i in range(len(hidden_layer_widths) - 1):
    weights += (hidden_layer_widths[i] + 1) * hidden_layer_widths[i + 1]

weights += (hidden_layer_widths[len(hidden_layer_widths) - 1] + 1) * 10

print(
    f'\n'
    f'Percentage correct on training data: {100 * pct_training:.2f}\n'
    f'Percentage correct on testing data: {100 * pct_testing:.2f}\n'
    f'\n'
    f'Train Amount: {100 * train_amount}%\n'
    f'Learning Rate: {learning_rate}\n'
    f'Momentum: {momentum}\n'
    f'Epochs: {epochs}\n'
    f'Batch Size: {batch_size}\n'
    f'Hidden Layer Widths: {hidden_layer_widths}\n'
    f'Weights: {weights}'
)
