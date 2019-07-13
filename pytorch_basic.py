
#check torch
import torch
print(torch.__version__)

#intro to tensor
v = torch.tensor([1,2,3,4])
#accessing
print(v[0])
#type tensor int64
print(v.dtype)
#slicing
print(v[0:])
print(v[0:2])

v = torch.FloatTensor([1,2,3,4,5,6])
print(v)
print(v.dtype)
#reshape
print(v.view(6,1))   #v.view(3,-1) #dont want to specify second dimension

import numpy as np
a = np.array([1,2,3,4,5])
#numpy to tensor
ten_v = torch.from_numpy(a);
print(ten_v)
#tensor to numpy
a = ten_v.numpy()
print(a)


#addition
t1 = torch.tensor([1,2,3])
t2 = torch.tensor([4,5,6])
print(t1+t2)
#multiplication with scaler
print(t1*5)
#multiplication with tensor
print(t1*t2)
#dot product
print(torch.dot(t1,t2))

#arg1 = start arg2 = end arg3 = size
t1 = torch.linspace(0,10,5)
print(t1)

import matplotlib.pyplot as plt
x = torch.linspace(0,10,100)
y = torch.exp(x)

#cannot plot tensor but does numpy
plt.plot(x.numpy(),y.numpy())
plt.show()
