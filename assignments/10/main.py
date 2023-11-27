import torch
import torch.cuda
import torch.nn as nn
from skimage import io
import du.lib as dulib
from matplotlib import pyplot as plt
import math

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

# train_amount = 0.8
train_amount = 0.88

xss, xss_means = dulib.center(xss)
xss, xss_stds = dulib.normalize(xss)

random_split = torch.randperm(xss.size(0))

xss_train = xss[random_split][:(math.floor(xss.size(0) * train_amount))]
xss_test = xss[random_split][(math.floor(xss.size(0) * train_amount)):]

yss_train = yss[random_split][:(math.floor(yss.size(0) * train_amount))]
yss_test = yss[random_split][(math.floor(yss.size(0) * train_amount)):]


class LogSoftmaxModel(nn.Module):
    def __init__(self):
        super(LogSoftmaxModel, self).__init__()
        self.layer1 = nn.Linear(400, 273)
        self.layer2 = nn.Linear(273, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return torch.log_softmax(x, dim=1)


model = LogSoftmaxModel()
criterion = nn.NLLLoss()


def pct_correct(xss_test_, yss_test_):
    count = 0

    for x, y in zip(xss_test_, yss_test_):
        if torch.argmax(x).item() == y.item():
            count += 1

    return 100 * count / len(xss_test_)


# learning_rate = 0.0002
# momentum = 0.9
# epochs = 256
# batch_size = 32

learning_rate = 0.06
momentum = 0.537
epochs = 16384
batch_size = 70

model = dulib.train(
    model,
    crit=criterion,
    train_data=(xss_train, yss_train),
    valid_data=(xss_test, yss_test),
    learn_params={'lr': learning_rate, 'mo': momentum},
    epochs=epochs,
    bs=batch_size,
    valid_metric=pct_correct,
    # graph=1
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
        # image = (xss_test[i] + xss_means).reshape(20, 20).detach().cpu().numpy()
        image = (xss_test[i] * xss_stds + xss_means).reshape(20, 20).detach().cpu().numpy()
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
