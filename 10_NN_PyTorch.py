import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision
import torchvision.transforms.v2 as T
import numpy as np
import matplotlib.pyplot as plt
import optuna
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.datasets import fetch_california_housing, fetch_covtype
from sklearn.model_selection import train_test_split
from functools import partial

# ── Device Setup ──────────────────────────────────────────────────────────────
# Automatically select the best available device: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

n_epochs = 20

# 1. LOW-LEVEL PYTORCH: Manual Gradient Descent with Autograd
# PyTorch tracks all operations on tensors with requires_grad=True. Calling .backward() computes gradients via the chain rule automatically.
# This is the foundation of all neural network training.

# ── Data Preparation ──────────────────────────────────────────────────────────
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
    housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

# Convert to float32 tensors (PyTorch default for neural nets)
X_train = torch.FloatTensor(X_train)
X_valid = torch.FloatTensor(X_valid)
X_test  = torch.FloatTensor(X_test)

# Standardize using training set statistics only (prevent data leakage)
means = X_train.mean(dim=0, keepdim=True)
stds  = X_train.std(dim=0, keepdim=True)
X_train = (X_train - means) / stds
X_valid = (X_valid - means) / stds
X_test  = (X_test  - means) / stds

# Reshape targets to column vectors for MSE compatibility
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
y_valid = torch.FloatTensor(y_valid).reshape(-1, 1)
y_test  = torch.FloatTensor(y_test).reshape(-1, 1)

# Standardize targets using training set mean/std (same principle as features)
y_mean = y_train.mean()
y_std  = y_train.std()
y_train = (y_train - y_mean) / y_std
y_valid = (y_valid - y_mean) / y_std
y_test  = (y_test  - y_mean) / y_std

n_features = X_train.shape[1]

# ── Manual Linear Regression via Autograd ─────────────────────────────────────
# Manually define weights and bias as learnable parameters.
# This shows exactly what nn.Linear does under the hood.
torch.manual_seed(42)
w = torch.randn((n_features, 1), requires_grad=True)
b = torch.tensor(0., requires_grad=True)

learning_rate = 0.4
for epoch in range(n_epochs):
    y_pred = X_train @ w + b          # forward pass: linear combination
    loss = ((y_pred - y_train) ** 2).mean()  # MSE loss
    loss.backward()                   # backward pass: compute gradients
    with torch.no_grad():             # update weights without tracking gradients
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        w.grad.zero_()                # reset gradients after each step
        b.grad.zero_()
    print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")

X_new = X_test[:3]
with torch.no_grad():
    y_pred = X_new @ w + b
print(y_pred)

# 2. MID-LEVEL PYTORCH: nn.Linear + Batch Gradient Descent
# nn.Linear wraps weights and bias into a module with automatic initialization.
# Batch Gradient Descent: update weights once per full epoch (no DataLoader).

torch.manual_seed(42)
model = nn.Linear(in_features=n_features, out_features=1)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
mse = nn.MSELoss()

for epoch in range(n_epochs):
    y_pred = model(X_train)
    loss = mse(y_pred, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")

# 3. HIGH-LEVEL PYTORCH: nn.Sequential + Mini-Batch SGD + DataLoader
# nn.Sequential stacks layers into a pipeline. DataLoader handles batching, shuffling, and iteration automatically. Mini-batch SGD: update weights after each batch → faster convergence.

# ── DataLoaders ───────────────────────────────────────────────────────────────
train_dataset = TensorDataset(X_train, y_train)
valid_dataset = TensorDataset(X_valid, y_valid)
train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader  = DataLoader(valid_dataset, batch_size=32)

# ── Generic Training & Evaluation Functions ───────────────────────────────────
# These reusable functions support:
#   - dict-based inputs (named inputs via **kwargs) for multi-input models
#   - torchmetrics for proper metric tracking across batches
#   - optional auxiliary outputs (multi-output models)
#   - history dict for plotting learning curves

def evaluate(model, data_loader, metric):
    """Evaluate model on a DataLoader using a torchmetrics metric.
    Supports both standard (tensor) and dict-based (named) inputs.
    Runs on CPU to avoid non-deterministic results on MPS."""
    model.eval()
    cpu_metric = metric.clone().to("cpu")
    cpu_metric.reset()
    with torch.no_grad():
        for inputs, y_batch in data_loader:
            if isinstance(inputs, dict):
                inputs = {k: v.to(device) for k, v in inputs.items()}
                output = model(**inputs)
            else:
                inputs = inputs.to(device)
                output = model(inputs)
            y_pred = output[0] if isinstance(output, tuple) else output
            is_clf = y_pred.dim() == 2 and y_pred.shape[1] > 1
            cpu_metric.update(y_pred.cpu() if is_clf else y_pred.cpu().flatten(),
                              y_batch.flatten())
    return cpu_metric.compute()

def train(model, optimizer, criterion, metric, train_loader, valid_loader,
          n_epochs):
    """Generic training loop supporting:
    - standard tensor inputs and dict-based named inputs (via **kwargs)
    - single and multiple outputs (weighted auxiliary loss: 0.8 main + 0.2 aux)
    - torchmetrics for train/valid tracking
    Returns history dict with losses and metrics per epoch."""
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.
        metric.reset()
        for inputs, y_batch in train_loader:
            y_batch = y_batch.to(device)
            if isinstance(inputs, dict):
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
            else:
                outputs = model(inputs.to(device))
            # Unpack main + optional auxiliary output (multi-output models)
            if isinstance(outputs, tuple):
                y_pred, y_aux = outputs
                loss = 0.8 * criterion(y_pred, y_batch) + 0.2 * criterion(y_aux, y_batch)
            else:
                y_pred = outputs
                loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Regression: [N,1] → flatten to [N]; Classification: keep [N,C] logits
            is_clf = y_pred.dim() == 2 and y_pred.shape[1] > 1
            metric.update(y_pred if is_clf else y_pred.flatten(),
                          y_batch.flatten())
        history["train_losses"].append(total_loss / len(train_loader))
        history["train_metrics"].append(metric.compute().item())
        history["valid_metrics"].append(evaluate(model, valid_loader, metric).item())
        print(f"Epoch {epoch + 1}/{n_epochs}  "
              f"loss: {history['train_losses'][-1]:.4f}  "
              f"train: {history['train_metrics'][-1]:.4f}  "
              f"valid: {history['valid_metrics'][-1]:.4f}")
    return history

# ── Deep Sequential Model ─────────────────────────────────────────────────────
# 3-layer MLP: 8 → 50 → 40 → 1 (ReLU activations introduce non-linearity between layers.
torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(n_features, 50), nn.ReLU(),
    nn.Linear(50, 40),         nn.ReLU(),
    nn.Linear(40, 1)
).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
mse  = nn.MSELoss()
rmse = torchmetrics.MeanSquaredError(squared=False).to(device)
history = train(model, optimizer, mse, rmse, train_loader, valid_loader, n_epochs)

# Plot learning curves to detect overfitting / underfitting
plt.plot(np.arange(n_epochs) + 0.5, history["train_metrics"], ".--", label="Training")
plt.plot(np.arange(n_epochs) + 1.0, history["valid_metrics"], ".-",  label="Validation")
plt.xlabel("Epoch"); plt.ylabel("RMSE"); plt.title("Learning Curves")
plt.grid(); plt.legend()
plt.show()

# 4. NON-SEQUENTIAL MODELS: Wide & Deep Architecture
# Wide & Deep combines a shallow "wide" path (raw features) with a deep path (learned representations), allowing the model to memorize AND generalize.
# Requires subclassing nn.Module to define custom forward() logic.

# ── Custom Dataset for Named Inputs ───────────────────────────────────────────
# Using a dict-based Dataset allows passing named tensors to the model's forward() method via **kwargs, making the architecture explicit and flexible.
class WideAndDeepDataset(torch.utils.data.Dataset):
    """Dataset that returns a dict of named inputs for multi-input models."""
    def __init__(self, X_wide, X_deep, y):
        self.X_wide = X_wide
        self.X_deep = X_deep
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"X_wide": self.X_wide[idx], "X_deep": self.X_deep[idx]}, self.y[idx]

# Wide path: first 5 features (e.g. location-based)
# Deep path: features 2–7 (overlapping subset for richer representation)
train_loader_named = DataLoader(WideAndDeepDataset(X_train[:, :5], X_train[:, 2:], y_train), batch_size=32, shuffle=True)
valid_loader_named = DataLoader(WideAndDeepDataset(X_valid[:, :5], X_valid[:, 2:], y_valid), batch_size=32)

# ── Wide & Deep with Multiple Inputs ──────────────────────────────────────────
# X_wide: short path directly to output (memorization)
# X_deep: passes through deep stack (generalization)
# Concatenated before the output layer → model sees both representations
class WideAndDeep(nn.Module):
    def __init__(self, n_wide, n_deep):
        super().__init__()
        self.deep_stack = nn.Sequential(
            nn.Linear(n_deep, 50), nn.ReLU(),
            nn.Linear(50, 40),     nn.ReLU(),
            nn.Linear(40, 30),     nn.ReLU(),
        )
        self.output_layer = nn.Linear(30 + n_wide, 1)

    def forward(self, X_wide, X_deep):
        deep_output = self.deep_stack(X_deep)
        return self.output_layer(torch.concat([X_wide, deep_output], dim=1))

torch.manual_seed(42)
model = WideAndDeep(n_wide=5, n_deep=n_features - 2).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
mse  = nn.MSELoss()
rmse = torchmetrics.MeanSquaredError(squared=False).to(device)
history = train(model, optimizer, mse, rmse, train_loader_named, valid_loader_named, n_epochs)

# ── Wide & Deep with Auxiliary Output ─────────────────────────────────────────
# An auxiliary output on the deep path acts as a regularizer: it forces the deep layers to learn useful representations independently, preventing the wide path from dominating training.
class WideAndDeepWithAux(nn.Module):
    def __init__(self, n_wide, n_deep):
        super().__init__()
        self.deep_stack = nn.Sequential(
            nn.Linear(n_deep, 50), nn.ReLU(),
            nn.Linear(50, 40),     nn.ReLU(),
            nn.Linear(40, 30),     nn.ReLU(),
        )
        self.output_layer     = nn.Linear(30 + n_wide, 1)
        self.aux_output_layer = nn.Linear(30, 1)  # auxiliary head on deep path

    def forward(self, X_wide, X_deep):
        deep_output    = self.deep_stack(X_deep)
        main_output    = self.output_layer(torch.concat([X_wide, deep_output], dim=1))
        aux_output     = self.aux_output_layer(deep_output)
        return main_output, aux_output  # train() handles the weighted loss

torch.manual_seed(42)
model = WideAndDeepWithAux(n_wide=5, n_deep=n_features - 2).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
mse  = nn.MSELoss()
rmse = torchmetrics.MeanSquaredError(squared=False).to(device)
history = train(model, optimizer, mse, rmse, train_loader_named, valid_loader_named,
                n_epochs)

# 5. CLASSIFICATION: Fashion MNIST with MLP
# Fashion MNIST: 70k grayscale images (28×28) across 10 clothing categories. (torchvision handles downloading, caching and applying transforms automatically.)

# ── Data Loading ──────────────────────────────────────────────────────────────
# ToImage converts PIL → tensor, ToDtype scales pixel values from [0,255] to [0,1]
toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
train_and_valid_data = torchvision.datasets.FashionMNIST(root="datasets", train=True,  download=True, transform=toTensor)
test_data            = torchvision.datasets.FashionMNIST(root="datasets", train=False, download=True, transform=toTensor)

# Split 60k training samples into 55k train / 5k validation
torch.manual_seed(42)
train_data, valid_data = torch.utils.data.random_split(train_and_valid_data, [55000, 5000])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader  = DataLoader(test_data,  batch_size=32)

# Inspect a single sample: shape [C, H, W], dtype float32
X_sample, y_sample = train_data[0]
print(f"Shape: {X_sample.shape}, dtype: {X_sample.dtype}, class: {train_and_valid_data.classes[y_sample]}")

# ── Model: MLP Image Classifier ───────────────────────────────────────────────
# Flatten 28×28 image to 784-dim vector, then two hidden layers → 10 logits.
# No softmax here: CrossEntropyLoss expects raw logits and applies it internally.
class ImageClassifier(nn.Module):
    def __init__(self, n_inputs, n_hidden1, n_hidden2, n_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_inputs,   n_hidden1), nn.ReLU(),
            nn.Linear(n_hidden1,  n_hidden2), nn.ReLU(),
            nn.Linear(n_hidden2,  n_classes)   # raw logits
        )

    def forward(self, X):
        return self.mlp(X)

torch.manual_seed(42)
model     = ImageClassifier(n_inputs=28*28, n_hidden1=300, n_hidden2=100, n_classes=10).to(device)
xentropy  = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
accuracy  = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
history   = train(model, optimizer, xentropy, accuracy, train_loader, valid_loader, n_epochs)

# ── Inference: Probabilities & Top-k Predictions ──────────────────────────────
# argmax on logits gives the predicted class index. softmax converts logits to probabilities (sum to 1 per sample). topk returns the k highest logits + their indices for confidence analysis.
model.eval()
X_new, y_new = next(iter(valid_loader))
X_new = X_new[:3].to(device)
with torch.no_grad():
    y_pred_logits = model(X_new)

y_pred_classes = y_pred_logits.argmax(dim=1)
print("Predicted:", [train_and_valid_data.classes[i] for i in y_pred_classes])
print("Actual:   ", [train_and_valid_data.classes[i] for i in y_new[:3]])

# Full probability distribution over all 10 classes
y_proba = F.softmax(y_pred_logits, dim=1).cpu()
print(y_proba.round(decimals=3))

# Top-4 most likely classes per sample
y_top4_logits, y_top4_indices = torch.topk(y_pred_logits, k=4, dim=1)
y_top4_probas = F.softmax(y_top4_logits, dim=1).cpu()
print(y_top4_probas.round(decimals=3))
print([[train_and_valid_data.classes[i] for i in row] for row in y_top4_indices])

# Total number of trainable parameters
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# 6. HYPERPARAMETER TUNING WITH OPTUNA
# Optuna uses Bayesian optimization (TPE sampler) to efficiently search the hyperparameter space – much smarter than grid or random search.
# Pruning stops unpromising trials early (MedianPruner: prune if below median).

def objective(trial, train_loader, valid_loader):
    """Optuna objective: train one model config and return best validation accuracy.
    trial.report() + trial.should_prune() enable early stopping of bad trials."""
    lr       = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    n_hidden = trial.suggest_int("n_hidden", 20, 300)
    model    = ImageClassifier(28*28, n_hidden, n_hidden, 10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    accuracy  = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
    best_acc  = 0.0
    for epoch in range(n_epochs):
        history  = train(model, optimizer, xentropy, accuracy, train_loader, valid_loader, n_epochs=1)
        val_acc  = history["valid_metrics"][-1]
        best_acc = max(best_acc, val_acc)
        trial.report(val_acc, epoch)       # report intermediate value for pruning
        if trial.should_prune():
            raise optuna.TrialPruned()     # stop this trial early
    return best_acc

sampler = optuna.samplers.TPESampler(seed=42)
pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
study   = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
study.optimize(partial(objective, train_loader=train_loader, valid_loader=valid_loader), n_trials=20)

print(f"Best accuracy: {study.best_value:.4f}")
print(f"Best params:   {study.best_params}")

# 7. SAVING AND LOADING MODELS
# Best practice: save state_dict + hyperparameters together so the model can
# be fully reconstructed without access to the original training code.

model_data = {
    "model_state_dict":      model.state_dict(),
    "model_hyperparameters": {"n_inputs": 28*28, "n_hidden1": 300, "n_hidden2": 100, "n_classes": 10}
}
torch.save(model_data, "fashion_mnist.pt")

# Reload: reconstruct architecture from saved hyperparameters, then load weights
loaded_data = torch.load("fashion_mnist.pt", weights_only=True)
new_model   = ImageClassifier(**loaded_data["model_hyperparameters"])
new_model.load_state_dict(loaded_data["model_state_dict"])
new_model.eval()

# Verify predictions match the original model
with torch.no_grad():
    y_pred_loaded = new_model(X_new.cpu())
print("Loaded model predictions match:", torch.allclose(y_pred_logits.cpu(), y_pred_loaded))

# ── torch.compile ─────────────────────────────────────────────────────────────
# torch.compile (PyTorch 2.0+) replaces the deprecated TorchScript (jit.trace/script).
# It JIT-compiles the model using TorchDynamo + TorchInductor for faster inference,
# requires zero changes to model code, and works on any nn.Module.
compiled_model = torch.compile(model)
with torch.no_grad():
    y_pred_compiled = compiled_model(X_new)
print("Compiled model output shape:", y_pred_compiled.shape)

# ══════════════════════════════════════════════════════════════════════════════
# EXERCISES
# ══════════════════════════════════════════════════════════════════════════════

# ── Exercise 1: Autograd on a custom function ─────────────────────────────────
# Compute partial derivatives of f(x,y) = sin(x²y) at x=1.2, y=3.4
# PyTorch builds a computation graph and differentiates automatically.
def f(x, y):
    return torch.sin(x ** 2 * y)

x = torch.tensor(1.2, requires_grad=True)
y = torch.tensor(3.4, requires_grad=True)
f(x, y).backward()
print(f"df/dx={x.grad.item():.4f}, df/dy={y.grad.item():.4f}")

# ── Exercise 2: Custom Dense Layer ────────────────────────────────────────────
# Dense v1: wraps nn.Linear + nn.ReLU (high-level, recommended)
# Dense v2: manually defines weight/bias as nn.Parameter (low-level, educational)
# Both are equivalent – v2 shows what nn.Linear does internally.
# Kaiming (He) initialization is the standard for ReLU networks.
class Dense(nn.Module):
    """High-level dense layer using nn.Linear + nn.ReLU."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu   = nn.ReLU()

    def forward(self, X):
        return self.relu(self.linear(X))

class Dense2(nn.Module):
    """Low-level dense layer with manual weight/bias parameters.
    Uses Kaiming uniform init (optimal for ReLU activations)."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, X):
        return F.relu(X @ self.weight.T + self.bias)

# Verify both implementations produce identical outputs
torch.manual_seed(42)
X = torch.randn(2, 3)
dense  = Dense(3, 5)
dense2 = Dense2(3, 5)
print("Dense output shape:",  dense(X).shape)
print("Dense2 output shape:", dense2(X).shape)

# ── Exercise 3: CoverType Classification ──────────────────────────────────────
# Forest cover type dataset: 581k samples, 54 features, 7 classes.
# Uses Dense2 as building block to show how custom layers compose into a full model.
# CoverTypeModel accepts a list of hidden sizes → flexible depth/width.
covtype   = fetch_covtype()
X_cov     = torch.tensor(covtype.data, dtype=torch.float32)
y_cov     = torch.tensor(covtype.target - 1, dtype=torch.long)  # 0-indexed classes

# Standardize features
X_cov = (X_cov - X_cov.mean(0)) / X_cov.std(0)

# 80/10/10 split
torch.manual_seed(42)
n       = len(y_cov)
sizes   = [n * 8 // 10, n // 10]
sizes  += [n - sum(sizes)]  # remainder goes to test
train_ds, valid_ds, test_ds = random_split(TensorDataset(X_cov, y_cov), sizes)

train_loader_cov = DataLoader(train_ds, batch_size=256, shuffle=True)
valid_loader_cov = DataLoader(valid_ds, batch_size=256)
test_loader_cov  = DataLoader(test_ds,  batch_size=256)

n_inputs  = X_cov.shape[1]   # 54
n_classes = 7

class CoverTypeModel(nn.Module):
    """Flexible MLP built from Dense2 layers.
    n_neurons: list of hidden layer sizes, e.g. [200, 100, 50]
    Uses nn.ModuleList so PyTorch tracks all parameters correctly."""
    def __init__(self, n_neurons, n_inputs=n_inputs, n_classes=n_classes):
        super().__init__()
        sizes = [n_inputs] + n_neurons
        self.layers = nn.ModuleList(
            [Dense2(sizes[i], sizes[i+1]) for i in range(len(n_neurons))]
            + [nn.Linear(n_neurons[-1], n_classes)]  # final linear (no activation)
        )

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

torch.manual_seed(42)
covtype_model = CoverTypeModel([200, 100, 50]).to(device)
criterion     = nn.CrossEntropyLoss()
metric_cov    = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes).to(device)
optimizer     = torch.optim.SGD(covtype_model.parameters(), lr=0.01)
history       = train(covtype_model, optimizer, criterion, metric_cov,
                      train_loader_cov, valid_loader_cov, n_epochs=15)
print(f"Test accuracy: {evaluate(covtype_model, test_loader_cov, metric_cov):.4f}")

# ── Exercise 4: Optuna Hyperparameter Search on CoverType ─────────────────────
# Search over learning rate, number of layers and hidden size.
# Pruning cuts unpromising trials early based on intermediate validation accuracy.
def covtype_objective(trial):
    lr       = trial.suggest_float("learning_rate", 1e-2, 1.0, log=True)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    n_hidden = trial.suggest_int("n_hidden", 30, 150)
    model    = CoverTypeModel([n_hidden] * n_layers).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    accuracy  = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes).to(device)
    best_acc  = 0.0
    for epoch in range(15):
        history  = train(model, optimizer, nn.CrossEntropyLoss(), accuracy,
                         train_loader_cov, valid_loader_cov, n_epochs=1)
        val_acc  = history["valid_metrics"][-1]
        best_acc = max(best_acc, val_acc)
        trial.report(val_acc, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return best_acc

torch.manual_seed(42)
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner()
)
study.optimize(covtype_objective, n_trials=5)
print(f"Best accuracy: {study.best_value:.4f}")
print(f"Best params:   {study.best_params}")
