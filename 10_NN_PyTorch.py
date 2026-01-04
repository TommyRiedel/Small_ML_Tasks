import torch
import numpy as np

X = torch.tensor([[1.0, 4.0, 7.0], [2.0, 3.0, 6.0]])
print(X)
print(X.shape)
print(X.dtype)
print(X[0, 1])
print(X[:, 1])

print(10 * (X + 1.0))
print(X.exp())
print(X.mean())
print(X.max(dim=0))
print(X@X.T)

print(X.numpy())
print(torch.tensor(np.array([[1., 4., 7.], [2., 3., 6.]])))
print(torch.FloatTensor([[1., 4., 7.], [2., 3., 6.]]))
X[:, 1] = 99
print(X)

X.relu_()
print(X)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(device)