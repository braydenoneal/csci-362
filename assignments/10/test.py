import torch
import torch.cuda
import torch.nn as nn
from skimage import io
import du.lib as dulib
import math
import random

digits = io.imread('digits.png')
xss_init = torch.Tensor(5000, 400)
idx = 0
for i in range(0, 1000, 20):
    for j in range(0, 2000, 20):
        xss_init[idx] = torch.Tensor((digits[i:i + 20, j:j + 20]).flatten())
        idx = idx + 1

yss = torch.LongTensor(len(xss_init))
for i in range(len(yss)):
    yss[i] = i // 500

epochs = 256

outcomes = []

for i in range(1000):
    train_amount = 0.8
    learning_rate = random.uniform(0.00001, 0.1)
    momentum = random.uniform(0.0, 0.9)
    batch_size = random.randint(8, 512)
    centered = random.randint(0, 1)
    normalized = random.randint(0, 1)
    hidden_layer_count = random.randint(1, 10)
    widths = []
    for j in range(hidden_layer_count):
        widths.append(random.randint(1, 400))

    random_split = torch.randperm(xss_init.size(0))
    train_split_amount = math.floor(xss_init.size(0) * train_amount)

    xss_train = xss_init[random_split][:train_split_amount]
    xss_test = xss_init[random_split][train_split_amount:]

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

            self.layer_start = nn.Linear(400, widths[0])

            layers = []
            for j in range(len(widths) - 1):
                layers.append(nn.Linear(widths[j], widths[j + 1]))

            self.layers_hidden = nn.ModuleList(layers)
            self.layer_final = nn.Linear(widths[-1], 10)

        def forward(self, x):
            x = self.layer_start(x)
            for layer in self.layers_hidden:
                x = torch.relu(layer(x))
            x = self.layer_final(x)
            return torch.log_softmax(x, dim=1)


    model = LogSoftmaxModel()
    criterion = nn.NLLLoss()

    model = dulib.train(
        model,
        crit=criterion,
        train_data=(xss_train, yss_train),
        valid_data=(xss_test, yss_test),
        learn_params={'lr': learning_rate, 'mo': momentum},
        epochs=epochs,
        bs=batch_size,
        verb=1,
    )

    pct_testing = dulib.class_accuracy(model, (xss_test, yss_test), show_cm=False)

    outcome = [pct_testing, train_amount, learning_rate, momentum, batch_size, centered, normalized, hidden_layer_count, widths]

    outcomes.append(outcome)

    outcomes.sort(key=lambda x: x[0], reverse=True)

    best = outcomes[0]

    print(
        f'\nPrevious\n'
        f'--------\n'
        f'Percentage correct: {pct_testing}\n'
        f'Train amount: {train_amount}\n'
        f'Learning rate: {learning_rate}\n'
        f'Momentum: {momentum}\n'
        f'Batch size: {batch_size}\n'
        f'Centered: {centered}\n'
        f'Normalized: {normalized}\n'
        f'Hidden layer count: {hidden_layer_count}\n'
        f'Hidden layer widths: {widths}\n'
        f'\n'
        f'Best\n'
        f'----\n'
        f'Percentage correct: {best[0]}\n'
        f'Train amount: {best[1]}\n'
        f'Learning rate: {best[2]}\n'
        f'Momentum: {best[3]}\n'
        f'Batch size: {best[4]}\n'
        f'Centered: {best[5]}\n'
        f'Normalized: {best[6]}\n'
        f'Hidden layer count: {best[7]}\n'
        f'Hidden layer widths: {best[8]}\n'
        f'==========================================='
    )

    output_file = open("output.txt", "a")
    output_file.write(f'{", ".join(str(parameter) for parameter in outcome)}\n')
    output_file.close()
