`braydenoneal/csci-362/assignments/1`

`1`

Prints the stringified value of a tensor containing 30 numbers generated from the standard normal distribution.

`2`

Source:

```python
import torch

xs = torch.randn(30)
print(xs)
print(xs.mean())
print(xs.std())
```

Output:

```
tensor([-0.7321, -0.6067, -0.5423, -1.2276,  0.5744,  0.1481,  1.4688, -1.2581,
        -0.1337, -0.8179,  2.2215, -1.4647,  0.1637,  1.0833, -2.4075, -0.4537,
        -0.6952, -0.6152, -0.1830,  3.2637, -1.0484,  0.3515, -2.2769,  0.5462,
         0.2901,  0.2300, -2.2028, -0.6533, -0.0662,  2.3971])
tensor(-0.1549)
tensor(1.3192)
```

* The mean is not exactly zero because the generated values are randomized, and the mean will be expected to be close to zero, but not exactly zero.
* The standard deviation is not exactly one because the generated values are randomized, and the standard deviation will be expected to be close to one, but not exactly one.

`3`

Source:

```python
import torch

xs = torch.empty(30).normal_(100, 25)
print(xs)
print(xs.mean())
print(xs.std())
```

Output:

```
tensor([105.8417, 113.7340, 106.8730, 102.0120,  71.4599, 100.2225, 104.1790,
        121.2611, 108.2696,  89.2471,  92.7100,  95.5478, 136.8538,  77.3129,
        104.6020, 108.6879, 149.2650,  76.0163,  94.2826, 101.4331, 114.2307,
        148.6491, 103.5268,  67.9319, 101.6384, 106.8466,  91.4479, 118.6108,
        119.6548,  56.4506])
tensor(102.9600)
tensor(21.1079)
```

`4`

I would expect the mean of the means to get closer to 100 as the number of samples increases.

Source:

```python
import torch

means = []

for i in range(0, 100000):
    means.append(torch.empty(30).normal_(100, 25).mean())

print(torch.tensor(means).mean())
```

Output:

```
tensor(100.0121)
```

`5`

I would expect the mean of the standard deviations to get closer to the correct value as the number of samples increases.

Source:

```python
import torch

standard_deviations = []

for i in range(0, 1000000):
    standard_deviations.append(torch.empty(30).normal_(100, 25).std())

print(torch.tensor(standard_deviations).mean())
```

Output:

```
tensor(24.7867)
```

`6`

Source:

```python
import torch

means = []
standard_deviations = []

for i in range(0, 100000):
    tensor = torch.empty(30).uniform_()
    means.append(tensor.mean())
    standard_deviations.append(tensor.std())

print(f"Mean: {torch.tensor(means).mean()}")
print(f"Standard Deviation: {torch.tensor(standard_deviations).mean()}")
```

Output:
```
Mean: 0.5003089308738708
Standard Deviation: 0.28754571080207825
```

The means of the means approaches 0.5 and the mean of the standard deviations approaches sqrt(12).
