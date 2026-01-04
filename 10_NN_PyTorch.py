import torch
import numpy as np
import time

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
print(torch.tensor(np.array([[1., 4., 7.], [2., 3., 6.]]), dtype=torch.float32))
print(torch.FloatTensor([[1., 4., 7.], [2., 3., 6.]]))
X[:, 1] = -99
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

M = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
M = M.to(device)
print(M.device)
M = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device=device)
R = M @ M.T
print(R)

M = torch.rand(1000, 1000)
start = time.time()
M @ M.T
time_cpu = time.time() - start
print(f"CPU Time: {time_cpu:.4f} seconds")

M = torch.rand(1000, 1000, device="mps")
start = time.time()
M @ M.T
time_gpu = time.time() - start
print(f"GPU Time: {time_gpu:.4f} seconds")
print(f"GPU is {time_cpu / time_gpu:.2f} times faster than CPU")

### Autograd
learning_rate = 0.1
x = torch.tensor(5.0, requires_grad=True)
for iteration in range(100):
    f = x ** 2  # forward pass
    f.backward()  # backward pass
    with torch.no_grad():
        x -= learning_rate * x.grad  # gradient descent step
    x.grad.zero_()  # reset the gradients
print(x)

######
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

try:
    housing = fetch_california_housing()
except Exception as e:
    print(f"Warning: {e}")
    from sklearn.datasets import make_regression
    X_data, y_data = make_regression(n_samples=20640, n_features=8, noise=10, random_state=42)
    class Housing:
        def __init__(self, data, target):
            self.data = data
            self.target = target
    housing = Housing(X_data, y_data)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=42
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42
)

X_train = torch.FloatTensor(X_train)
X_valid = torch.FloatTensor(X_valid)
X_test = torch.FloatTensor(X_test)
means = X_train.mean(dim=0, keepdim=True)
stds = X_train.std(dim=0, keepdim=True)
X_train = (X_train - means) / stds
X_valid = (X_valid - means) / stds
X_test = (X_test - means) / stds

y_train = torch.FloatTensor(y_train).reshape(-1, 1)
y_valid = torch.FloatTensor(y_valid).reshape(-1, 1)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)

torch.manual_seed(42)
n_features = X_train.shape[1]
w = torch.randn((n_features, 1), requires_grad=True)
b = torch.tensor(0., requires_grad=True)

learning_rate = 0.4
n_epochs = 20
for epoch in range(n_epochs):
    y_pred = X_train @ w + b
    loss = ((y_pred - y_train) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()
    print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")

X_new = X_test[:3]
with torch.no_grad():
    y_pred = X_new @ w + b
print(y_pred)


###
import torch.nn as nn
torch.manual_seed(42)
model = nn.Linear(in_features=n_features, out_features=1)
print(model.bias)
print(model.weight)

for param in model.parameters():
    print(param)

print(model(X_train[:2]))
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
mse = nn.MSELoss()

def train_bgd(model, optimizer, criterion, X_train, y_train, n_epochs):
    for epoch in range(n_epochs):
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")

train_bgd(model, optimizer, mse, X_train, y_train, n_epochs)
X_new = X_test[:3]
with torch.no_grad():
    y_pred = model(X_new)
print(y_pred)

###
torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(n_features, 50),
    nn.ReLU(),
    nn.Linear(50, 40),
    nn.ReLU(),
    nn.Linear(40, 1)
)
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
mse = nn.MSELoss()
train_bgd(model, optimizer, mse, X_train, y_train, n_epochs)

###
from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(n_features, 50),
    nn.ReLU(),
    nn.Linear(50, 40),
    nn.ReLU(),
    nn.Linear(40, 1)
).to(device)

# extra code – build the optimizer and loss function, as earlier
learning_rate = 0.02
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
mse = nn.MSELoss()

def train(model, optimizer, criterion, train_loader, n_epochs):
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        mean_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{n_epochs}: Loss = {mean_loss:.4f}")

train(model, optimizer, mse, train_loader, n_epochs)


### Model Evaluation
def evaluate(model, data_loader, metric_fn, aggregate_fn=torch.mean):
    model.eval()
    metrichs = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_batch)
            metric = metric_fn(y_pred, y_batch)
            metrichs.append(metric)
        return aggregate_fn(torch.stack(metrichs))
    
valid_dataset = TensorDataset(X_valid, y_valid)
valid_loader = DataLoader(valid_dataset, batch_size=32)
valid_mse = evaluate(model, valid_loader, mse)
print(f"Validation MSE: {valid_mse:.4f}")

def mse(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean().sqrt()

print(evaluate(model, valid_loader, mse))
print(valid_mse.sqrt())

print(evaluate(model, valid_loader, mse, aggregate_fn = lambda metrics:torch.sqrt(torch.mean(metrics))))
import torchmetrics

def evaluate_tm(model, data_loader, metric):
    model.eval()
    metric.reset()
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch)
        return metric.compute()
    
rmse = torchmetrics.MeanSquaredError(squared=False).to(device)
print(evaluate_tm(model, valid_loader, rmse))

import matplotlib.pyplot as plt
def train2(model, optimizer, criterion, metric, train_loader, valid_loader,
               n_epochs):
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
    for epoch in range(n_epochs):
        total_loss = 0.
        metric.reset()
        for X_batch, y_batch in train_loader:
            model.train()
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            metric.update(y_pred, y_batch)
        mean_loss = total_loss / len(train_loader)
        history["train_losses"].append(mean_loss)
        history["train_metrics"].append(metric.compute().item())
        history["valid_metrics"].append(
            evaluate_tm(model, valid_loader, metric).item())
        print(f"Epoch {epoch + 1}/{n_epochs}, "
              f"train loss: {history['train_losses'][-1]:.4f}, "
              f"train metric: {history['train_metrics'][-1]:.4f}, "
              f"valid metric: {history['valid_metrics'][-1]:.4f}")
    return history

torch.manual_seed(42)
learning_rate = 0.01
model = nn.Sequential(
    nn.Linear(n_features, 50), nn.ReLU(),
    nn.Linear(50, 40), nn.ReLU(),
    nn.Linear(40, 30), nn.ReLU(),
    nn.Linear(30, 1)
)
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
mse = nn.MSELoss()
rmse = torchmetrics.MeanSquaredError(squared=False).to(device)
history = train2(model, optimizer, mse, rmse, train_loader, valid_loader,
                 n_epochs)

# Since we compute the training metric
plt.plot(np.arange(n_epochs) + 0.5, history["train_metrics"], ".--",
         label="Training")
plt.plot(np.arange(n_epochs) + 1.0, history["valid_metrics"], ".-",
         label="Validation")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.grid()
plt.title("Learning curves")
plt.axis([0.5, 20, 0.4, 1.0])
plt.legend()
plt.show()

### Non-sequetntial model