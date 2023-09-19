import torch
import csv
import matplotlib.pyplot as plt

with open('assign2.csv') as data_file:
    data = list(csv.reader(data_file))[1:]
    x_values = torch.tensor([float(line[0]) for line in data])
    y_values = torch.tensor([float(line[1]) for line in data])

design_matrix = torch.tensor([[1, value] for value in x_values])

weights = (
    design_matrix.transpose(0, 1)
    .mm(design_matrix).inverse()
    .mm(design_matrix.transpose(0, 1))
    .mm(y_values.unsqueeze(1))
)

intercept = weights[0].item()
slope = weights[1].item()

plt.title(f'Slope: {slope:.4}\tIntercept: {intercept:.6}')
plt.scatter(x_values, y_values)
plt.plot(x_values, x_values * slope + intercept, color='red')
plt.show()
