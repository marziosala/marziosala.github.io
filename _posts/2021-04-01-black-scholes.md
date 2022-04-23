---
layout: splash
permalink: /black-scholes/
title: "Approximating the Black-Scholes Equation"
header:
  overlay_image: /assets/images/black-scholes/black-scholes-splash.jpeg
excerpt: "Using neural network to approximate the Black-Scholes equation for pricing vanilla options."
---

What we want to do here is to approximate the famous [Black-Scholes formula](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model), presented in 1973 by Myron Scholes, Fisher Black and Robert Merton, with neural networks. The formula easy to understand and, after some algebraic manipulations, depends on only two parameters: the *moneyness* $\log(F/K)$ and the variance $\sigma^2 (T - t)$. To make the problem a bit more interesting, we approximate both the value of the option as well as its derivative (the so-called *delta*). Therefore, we have two parameters in input and two output.


```python
import matplotlib.pylab as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import seaborn as sns
from sklearn.model_selection import train_test_split
```

The formula requires the computation of the cumulative density function (CDF) of the standard normal distribution,

$$
F_X(x) = \frac{1}{\sqrt{2 \pi}} \int_{-\infty}^x e^{-z^2/2} dz,
$$

so we define `norm_dist` to compute that with Torch.


```python
norm_dist = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))
softmax = nn.Softmax(dim=1)
```

It is simpler to predict the price minus the intrinsic value $\max\{S - K, 0\}$, because far from the current spot level the remainder is close to zero.


```python
def training_price(log_F_over_K, σ_sqrt_τ, subtract_intrinsic):
    F_over_K = torch.exp(log_F_over_K)
    d_p = (log_F_over_K + 0.5 * σ_sqrt_τ**2) / σ_sqrt_τ
    N_p = norm_dist.cdf(d_p)
    d_m = (log_F_over_K - 0.5 * σ_sqrt_τ**2) / σ_sqrt_τ
    N_m = norm_dist.cdf(d_m)
    tv = F_over_K * N_p - N_m
    Δ = norm_dist.cdf(d_p)
    if subtract_intrinsic:
        intrinsic = torch.maximum(F_over_K - torch.tensor(1.0), torch.tensor(0.0))
        return torch.hstack([tv - intrinsic, Δ])
    else:
        torch.hstack([tv, Δ])
```


```python
def price(S, K, r, q, σ, τ):
    F = S * torch.exp((r - q) * τ)
    log_F_over_K = torch.log(F / K)
    σ_sqrt_τ = σ_ * torch.sqrt(τ)
    y = training_price(log_F_over_K, σ_sqrt_τ, False)
    return torch.exp(-r * τ) * K * y[0], torch.exp(-q * τ) * y[1]
```


```python
n = 1_000

x1 = torch.normal(0, 0.25, size=(n, 1))
x2 = ((torch.rand(n, 1) * 0.5) * (torch.rand(n, 1)).sqrt() + 1e-5)
X = torch.hstack([x1, x2])

Y = training_price(x1, x2, True)
```


```python
fig, (ax0, ax1) = plt.subplots(figsize=(12, 4), ncols=2)
ax0.scatter(x1, Y[:, 0])
ax1.scatter(x1, Y[:, 1])
fig.tight_layout()
```


    
![png](/assets/images/black-scholes/black-scholes-1.png)
    



```python
X_train, X_test, Y_train, Y_test = train_test_split(X.numpy(), Y.numpy(), test_size=0.1)
print(f"Selected {len(X_train)} instances for training the model and {len(X_test)} to test it.")
```

    Selected 900 instances for training the model and 100 to test it.
    


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```


```python
class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, X, Y):
        assert len(X) == len(Y)
        self.X = X
        self.Y = Y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        X = torch.tensor(self.X[index], dtype=torch.float32)
        Y = torch.tensor(self.Y[index], dtype=torch.float32)
        return X, Y
```


```python
class Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 2)
    
    def forward(self, x):
        x = func.relu(self.layer1(x))
        x = func.relu(self.layer2(x))
        x = self.layer3(x)
        return x
```


```python
model = Model()
```


```python
params = {'batch_size': 32, 'shuffle': True}

dataset_training = Dataset(X_train_scaled, Y_train)
generator_training = torch.utils.data.DataLoader(dataset_training, **params)

dataset_validation = Dataset(X_test_scaled, Y_test)
generator_validation = torch.utils.data.DataLoader(dataset_validation, **params)
```


```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
max_epochs = 500

for epoch in range(max_epochs):
    # training
    total_training_loss = 0.0
    for X_batch, y_batch in generator_training:
        optimizer.zero_grad()
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_batch_pred = model.forward(X_batch)
        loss = criterion(y_batch_pred, y_batch)
        total_training_loss += loss.item()
        loss.backward()
        optimizer.step()
    # validation
    with torch.set_grad_enabled(False):
        total_validation_loss = 0.0
        for X_batch, y_batch in generator_validation:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_batch_pred = model.forward(X_batch)
            loss = criterion(y_batch_pred, y_batch)
            total_validation_loss += loss.item()
    if (epoch + 1) % 100 == 0:
        print(f"epoch: {epoch + 1}, training loss: {total_training_loss}, val loss: {total_validation_loss}")
```

    epoch: 100, training loss: 0.13884549913927913, val loss: 0.023546889424324036
    epoch: 200, training loss: 0.0972703694133088, val loss: 0.019179114140570164
    epoch: 300, training loss: 0.07735404965933412, val loss: 0.01914267987012863
    epoch: 400, training loss: 0.0818604618543759, val loss: 0.018637395463883877
    epoch: 500, training loss: 0.06204671808518469, val loss: 0.016463090665638447
    


```python
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(figsize=(12, 8), nrows=2, ncols=2)

Y_pred = model(torch.tensor(X_test_scaled)).detach().numpy()

ax0.scatter(X_test_scaled[:, 0], Y_test[:, 0])
ax0.scatter(X_test_scaled[:, 0], Y_pred[:, 0])
ax1.scatter(X_test_scaled[:, 0], (Y_test - Y_pred)[:, 0])

ax2.scatter(X_test_scaled[:, 0], Y_test[:, 1])
ax2.scatter(X_test_scaled[:, 0], Y_pred[:, 1])
ax3.scatter(X_test_scaled[:, 0], (Y_test - Y_pred)[:, 1])

fig.tight_layout()
```


    
![png](/assets/images/black-scholes/black-scholes-2.png)
    


Considering we are using only one thousand points, the errors in both the option price and delta are quite small. The delta has some large discrepancies around zero logmoneyess, as it jumps from 0 to 1 quickly. These errors can be removed by smoothing out the delta itself in the training.
