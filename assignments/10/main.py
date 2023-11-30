import torch
import torch.cuda
import torch.nn as nn
from skimage import io
import du.lib as dulib
from matplotlib import pyplot as plt
import math

train_amount = 0.8874703565476355
learning_rate = 0.03524971724959869
momentum = 0.3535090099859246
epochs = 256
batch_size = 187
centered = True
normalized = False
hidden_layer_widths = [206, 363, 383, 318, 74, 347]

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

xss_train_means = 0
xss_train_stds = 1

xss_train = xss[random_split][:(math.floor(xss.size(0) * train_amount))]

if centered:
    xss_train, xss_train_means = dulib.center(xss_train)
if normalized:
    xss_train, xss_train_stds = dulib.normalize(xss_train)

xss_test = xss[random_split][(math.floor(xss.size(0) * train_amount)):] * xss_train_stds + xss_train_means

yss_train = yss[random_split][:(math.floor(yss.size(0) * train_amount))]
yss_test = yss[random_split][(math.floor(yss.size(0) * train_amount)):]


class LogSoftmaxModel(nn.Module):
    def __init__(self):
        super(LogSoftmaxModel, self).__init__()
        widths = hidden_layer_widths
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

misread_images = []

count = 0

for i in range(len(xss_test)):
    prediction = torch.argmax(model(xss_test[i].unsqueeze(0).cuda())).item()
    actual = yss_test[i].item()

    if prediction == actual:
        count += 1
    else:
        image = (xss_test[i] * xss_train_stds + xss_train_means).reshape(20, 20).detach().cpu().numpy()
        misread_images.append((prediction, actual, image))

print(
    f'\n'
    f'Percentage correct on training data: {100 * pct_training:.2f}\n'
    f'Percentage correct on testing data: {100 * pct_testing:.2f}\n'
    f'\n'
    f'Learning Rate: {learning_rate}\n'
    f'Momentum: {momentum}\n'
    f'Epochs: {epochs}\n'
    f'Batch Size: {batch_size}'
)

if len(misread_images) > 0:
    figure, subplots = plt.subplots(1, len(misread_images), squeeze=False)

    for i in range(len(misread_images)):
        subplots[0][i].imshow(misread_images[i][2], cmap='gray')
        subplots[0][i].title.set_text(f'{misread_images[i][0]}/{misread_images[i][1]}')
        subplots[0][i].set_axis_off()

    plt.show()
