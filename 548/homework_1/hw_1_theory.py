import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


def discontinuous_func(x):
    if int(torch.gt(x, 0)) == 1:
        return x ** 2
    elif int(torch.lt(x, 0)) == 1:
        return torch.exp(x)
    return x + 2 - x


def discontinuous_func_2(x):
    if int(torch.gt(x, 0)) == 1:
        return x ** 2
    elif int(torch.lt(x, 0)) == 1:
        return torch.exp(x)
    return (2 * x) + 2


def problem4a():
    y = [
        discontinuous_func(torch.zeros(1) + x)[0]
        for x in torch.arange(-3, 3, .01)
    ]
    plt.scatter(torch.arange(-3, 3, .01).numpy(), y, s=10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    # problem4a()
    x = Variable(torch.zeros(1), requires_grad=True)
    y = discontinuous_func_2(x)
    print(y)
    y.backward()
    print(x.grad)
