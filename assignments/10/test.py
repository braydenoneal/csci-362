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
    train_amount = random.uniform(0.5, 0.9)
    learning_rate = random.uniform(0.00001, 0.1)
    momentum = random.uniform(0.1, 0.9)
    batch_size = random.randint(16, 256)
    center = random.randint(0, 1)
    normalize = random.randint(0, 1)
    hidden_layer_count = random.randint(1, 10)
    widths = []
    for j in range(hidden_layer_count):
        widths.append(random.randint(1, 400))

    if center:
        xss, xss_means = dulib.center(xss_init)
    else:
        xss = xss_init
    if normalize:
        xss, xss_stds = dulib.normalize(xss)

    random_split = torch.randperm(xss.size(0))

    xss_train = xss[random_split][:(math.floor(xss.size(0) * train_amount))]
    xss_test = xss[random_split][(math.floor(xss.size(0) * train_amount)):]

    yss_train = yss[random_split][:(math.floor(yss.size(0) * train_amount))]
    yss_test = yss[random_split][(math.floor(yss.size(0) * train_amount)):]


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

    outcome = [pct_testing, train_amount, learning_rate, momentum, batch_size, center, normalize, hidden_layer_count, widths]

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
        f'Centered: {center}\n'
        f'Normalized: {normalize}\n'
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

"""
Percentage correct: 0.9471264367816095
Train amount: 0.8274492625518783
Learning rate: 0.0007
Momentum: 0.8780680615072185
Batch size: 77
Hidden layer neurons: 238

Percentage correct: 0.9464285714285713
Train amount: 0.8339569889222629
Learning rate: 0.0007
Momentum: 0.6515284841974023
Batch size: 22
Hidden layer neurons: 302

Percentage correct: 0.9317567567567577
Train amount: 0.705475597191265
Learning rate: 0.006
Momentum: 0.7728091493690272
Batch size: 158
Hidden layer neurons: 189
Centered: 1
Normalized: 1

Percentage correct: 0.9442153493699887
Train amount: 0.8063084048404897
Learning rate: 0.1
Momentum: 0.809065803511045
Batch size: 120
Hidden layer neurons: 56
Centered: 1
Normalized: 1

Percentage correct: 0.9540540540540537
Train amount: 0.8532992593969861
Learning rate: 0.001
Momentum: 0.5387377083688535
Batch size: 18
Hidden layer neurons: 283
Centered: 0
Normalized: 0

Percentage correct: 0.958064516129032
Train amount: 0.8768518172847125
Learning rate: 0.06
Momentum: 0.5372809918549417
Batch size: 70
Hidden layer neurons: 273
Centered: 1
Normalized: 1

Percentage correct: 0.9620689655172412
Train amount: 0.8856627772022108
Learning rate: 0.0966781630899637
Momentum: 0.5831917228617731
Batch size: 17
Hidden layer neurons: 160
Centered: 0
Normalized: 1

Percentage correct: 0.9684210526315788
Train amount: 0.8874703565476355
Learning rate: 0.03524971724959869
Momentum: 0.3535090099859246
Batch size: 187
Centered: 1
Normalized: 0
Hidden layer count: 6
Hidden layer widths: [206, 363, 383, 318, 74, 347]
"""
