from fwdgrad.model import NeuralNet
import torch
from torch.autograd.functional import jvp
net = NeuralNet(4, [20, 30], 10)
loss = torch.nn.CrossEntropyLoss(reduce=False)
batch_size = 3
inputs = torch.rand(batch_size, 4, requires_grad=True)
label = torch.empty(batch_size, dtype=torch.long).random_(10)


def fun(inputs):
    o = net(inputs)
    l = loss(o, label)
    return l


o = fun(inputs).sum()
o.backward()
gd = inputs.grad.data

# v = torch.normal(0, 1, (2,4))
v = torch.zeros(batch_size, 4)
v[0, 0] = 1.0
r = jvp(fun, inputs, v)
print(r[1][0], gd[0, 0])

v = torch.zeros(batch_size, 4)
v[0, 1] = 1.0
r = jvp(fun, inputs, v)
print(r[1][0], gd[0, 1])

v = torch.zeros(batch_size, 4)
v[0, 2] = 1.0
r = jvp(fun, inputs, v)
print(r[1][0], gd[0, 2])

v = torch.zeros(batch_size, 4)
v[0, 3] = 1.0
r = jvp(fun, inputs, v)
print(r[1][0], gd[0, 3])

v = torch.zeros(batch_size, 4)
v[1, 0] = 1.0
r = jvp(fun, inputs, v)
print(r[1][1], gd[1, 0])

v = torch.zeros(batch_size, 4)
v[1, 1] = 1.0
r = jvp(fun, inputs, v)
print(r[1][1], gd[1, 1])

v = torch.zeros(batch_size, 4)
v[1, 2] = 1.0
r = jvp(fun, inputs, v)
print(r[1][1], gd[1, 2])

v = torch.zeros(batch_size, 4)
v[1, 3] = 1.0
r = jvp(fun, inputs, v)
print(r[1][1], gd[1, 3])

num_estimates = 100

with open('result_rad.csv', 'w') as f:
    estimate_nums = torch.range(0, 11)
    for estimate_num in estimate_nums:
        print(estimate_num)
        errors = []
        for _ in range(num_estimates):
            sum_vec = 0
            nums = 2**estimate_num
            nums = int(nums.item())
            for i in range(nums):
                # v = torch.normal(0, 1, (2,4))
                v = torch.randint(0, 2, (batch_size, 4))*2 - 1
                r = jvp(fun, inputs, v)
                sum_vec = sum_vec + r[1][:, None]*v
            # print(sum_vec/estimate_num)
            error = ((gd-sum_vec/nums)**2.0).mean().sqrt()
            errors.append(str(error.item()))
#            print(nums, error.item())
        error_str = ','.join(errors)
        f.write(f'{nums},{error_str}\n')

with open('result_norm.csv', 'w') as f:
    estimate_nums = torch.range(0, 11)
    for estimate_num in estimate_nums:
        print(estimate_num)
        errors = []
        for _ in range(num_estimates):
            sum_vec = 0
            nums = 2**estimate_num
            nums = int(nums.item())
            for i in range(nums):
                v = torch.normal(0, 1, (batch_size, 4))
                r = jvp(fun, inputs, v)
                sum_vec = sum_vec + r[1][:, None]*v
            # print(sum_vec/estimate_num)
            error = ((gd-sum_vec/nums)**2.0).mean().sqrt()
            errors.append(str(error.item()))
#            print(nums, error.item())
        error_str = ','.join(errors)
        f.write(f'{nums},{error_str}\n')