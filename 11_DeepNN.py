import torch
import torch.nn as nn

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

layer = nn.Linear(40, 10)
layer.weight.data *= 6 ** 0.5
torch.zero_(layer.bias.data)
nn.init.kaiming_uniform_(layer.weight)
nn.init.zeros_(layer.bias)

def use_he_init(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight)
        nn.init.zeros_(module.bias)

model = nn.Sequential(nn.Linear(50, 40), nn.ReLU(), nn.Linear(40, 1), nn.ReLU())
model.apply(use_he_init)

alpha = 0.2
model = nn.Sequential(nn.Linear(50, 40), nn.LeakyReLU(negative_slope=alpha))
nn.init.kaiming_uniform_(model[0].weight, alpha, nonlinearity='leaky_relu')


###
model = nn.Sequential(
    nn.Flatten(),
    nn.BatchNorm1d(1 * 28 * 28),
    nn.Linear(1 * 28 * 28, 300, bias=False),
    nn.ReLU(),
    nn.BatchNorm1d(300),
    nn.Linear(300, 100, bias=False),
    nn.ReLU(),
    nn.BatchNorm1d(100),
    nn.Linear(100, 10)
)

print(dict(model[1].named_parameters()).keys())
print(dict(model[1].named_buffers()).keys())

torch.manual_seed(42)
inputs = torch.randn(32, 3, 100, 200)
layer_norm = nn.LayerNorm([100, 200])
result = layer_norm(inputs)

means = inputs.mean(dim=[2,3], keepdim=True)
vars_ = inputs.var(dim=[2, 3], keepdim=True, unbiased=False)
stds = torch.sqrt(vars_ + layer_norm.eps)
result2 = layer_norm.weight * (inputs - means) / stds + layer_norm.bias
assert torch.allclose(result, result2)

layer_norm = nn.LayerNorm([3, 100, 200])
result = layer_norm(inputs)

####
torch.manual_seed(42)
model_A = nn.Sequential(nn.Flatten(), nn.Linear(3 * 28 * 28, 100), nn.ReLU(),
                        nn.Linear(100, 100), nn.ReLU(), 
                        nn.Linear(100, 100), nn.ReLU(), 
                        nn.Linear(100, 8))

import copy
import torchmetrics
reused_layers = copy.deepcopy(model_A[:-1])
model_B_on_A = nn.Sequential(*reused_layers, nn.Linear(100, 1)).to(device)

for layer in model_B_on_A[:-1]:
    for param in layer.parameters():
        param.requires_grad = False

xentropy = nn.BCEWithLogitsLoss()
accuracy = torchmetrics.Accuracy(task="binary").to(device)
