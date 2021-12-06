---
layout: splash
permalink: /variational-autoencoders/
title: "Variational Autoencoders"
header:
  overlay_image: /assets/images/variational-autoencoders/variational-autoencoders-splash.png
excerpt: "Variational autoencoders applied to mathematical functions."
---

In the *autoencoder* post we have seen how to approximate the PDF of the Beta distribution. As we noticed, in general the encoding space is a non-convex manifold and the codes have arbitrary scales. This makes basic autoencoders a poor choice for generative models. *Variational autoencoders* fix this issue by ensuring that the coding space follows a desirable distribution from which we can easily sample from. This distribution typically is the standard normal distribution.

https://mathybit.github.io/auto-var/


```python
import seaborn as sns
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt
```


```python
device = 'cpu' # if torch.cuda.is_available() else 'cpu'
```


```python
uniform_dist = torch.distributions.uniform.Uniform(0, 5)
```


```python
num_inputs = 5_000
num_points = 100
```


```python
grid = torch.linspace(0.01, 0.99, num_points)
```


```python
def target_func(grid, α, β):
    return torch.distributions.Beta(α, β).log_prob(grid)
```


```python
X, Y = [], []
for _ in range(num_inputs):
    α = uniform_dist.sample()
    β = uniform_dist.sample()
    X.append(target_func(grid, α, β))
    Y.append([α, β])
X = torch.vstack(X)

assert X.shape[0] == num_inputs
assert X.shape[1] == num_points
assert X.isnan().sum() == 0.0
```


```python
for x, (α, β) in zip(X[:5], Y[:5]):
    plt.plot(grid, x.exp(), label=f'α={α.item():.4f}, β={β.item():.4f}')
plt.legend()
```




    <matplotlib.legend.Legend at 0x1bd0bee68c8>




    
![png](/assets/images/variational-autoencoders/variational-autoencoders-1.png)
    



```python
num_hidden = 256
```


```python
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.linear = nn.Linear(num_points, num_hidden)
        self.linear_mu = nn.Linear(num_hidden, latent_dims)
        self.linear_logsigma2 = nn.Linear(num_hidden, latent_dims)
        self.N = torch.distributions.Normal(0, 1)
        if device == 'cuda':
            self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = F.relu(self.linear(x))
        mu =  self.linear_mu(x)
        logsigma2 = self.linear_logsigma2(x)
        sigma = torch.exp(logsigma2 / 2)
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = ((sigma**2 + mu**2 - logsigma2 - 1) / 2).sum()
        return z
```


```python
class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.linear1 = nn.Linear(latent_dims, num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_points)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        return self.linear2(z)  # can use F.relu() as well as this is a CDF
```


```python
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
```


```python
def train(vae, data, epochs=20, lr=1e-3, gamma=0.95, β=1, print_every=1):
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    history = []
    for epoch in range(epochs):
        last_lr = scheduler.get_last_lr()
        total_diff_loss, total_kl_loss = 0.0, 0.0
        for x in data:
            x = x.to(device) # GPU
            optimizer.zero_grad()
            x_hat = vae(x)
            diff = ((x - x_hat)**2).sum()
            total_diff_loss += diff.item()
            total_kl_loss += vae.encoder.kl.item()
            loss = diff + β * vae.encoder.kl
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % print_every == 0:
            print(f"Epoch: {epoch + 1:3d}, lr=: {last_lr[0]:.4e}, " \
                  f"total diff loss: {total_diff_loss:.4f}, total KL loss: {total_kl_loss:.4f}")
        history.append((total_diff_loss, total_kl_loss))
    return history
```


```python
%%time
latent_dims = 2
vae = VariationalAutoencoder(latent_dims).to(device)
history = train(vae, X, epochs=100, lr=1e-4, gamma=0.975, β=1, print_every=10)
```

    Epoch:  10, lr=: 7.9624e-05, total diff loss: 9686.8539, total KL loss: 32634.2520
    Epoch:  20, lr=: 6.1814e-05, total diff loss: 7537.9995, total KL loss: 31047.5534
    Epoch:  30, lr=: 4.7988e-05, total diff loss: 6642.3115, total KL loss: 30592.5397
    Epoch:  40, lr=: 3.7255e-05, total diff loss: 6179.9556, total KL loss: 30344.2045
    Epoch:  50, lr=: 2.8922e-05, total diff loss: 5872.4004, total KL loss: 30209.7972
    Epoch:  60, lr=: 2.2453e-05, total diff loss: 5739.7177, total KL loss: 30099.9995
    Epoch:  70, lr=: 1.7431e-05, total diff loss: 5599.2377, total KL loss: 30060.6767
    Epoch:  80, lr=: 1.3532e-05, total diff loss: 5446.7897, total KL loss: 30120.5200
    Epoch:  90, lr=: 1.0505e-05, total diff loss: 5408.5707, total KL loss: 30101.2888
    Epoch: 100, lr=: 8.1556e-06, total diff loss: 5340.8148, total KL loss: 30093.3920
    Wall time: 19min 31s
    


```python
C = []
for entry in X:
    C.append(vae.encoder(entry.to(device)).cpu().detach().numpy())
C = np.array(C)
```


```python
fig, (ax0, ax1) = plt.subplots(figsize=(10, 4), ncols=2)
sns.histplot(C[:, 0], ax=ax0)
sns.histplot(C[:, 1], ax=ax1)
ax0.set_xlabel('c_1')
ax1.set_xlabel('c_2')
fig.tight_layout()
```


    
![png](/assets/images/variational-autoencoders/variational-autoencoders-2.png)
    


Although the second code $c_2$ isn't quite symmetric, they both are broadly in the (-3, 3) interval, as we would expect from a standard normal variate. It is then much easier to generate new values or look for the optimal ones that match a given, which is what we try now to do. First, we generate two random values for $\alpha$ and $\beta$ and compute the corresponding exact PDF; then we use `scipy` to find the values of $(c_1, c_2)$ that produce the closest match.


```python
from scipy.optimize import minimize
```


```python
α = uniform_dist.sample()
β = uniform_dist.sample()

print(f'Params for the test: α={α.item():.4f}, β={β.item():.4f}')

X_target = target_func(grid, α, β).numpy()

def func(x):
    X_pred = vae.decoder(torch.FloatTensor(x).to(device))
    X_pred = X_pred.cpu().detach().numpy()
    diff = np.linalg.norm(X_target - X_pred).item()
    return diff
```

    Params for the test: α=1.7887, β=4.1733
    


```python
res = minimize(func, [0.0, 0.0], method='Nelder-Mead')
res
```




     final_simplex: (array([[ 1.1548077 , -0.61330039],
           [ 1.15478857, -0.61321818],
           [ 1.15484943, -0.61327551]]), array([0.05398799, 0.05399115, 0.05399147]))
               fun: 0.05398799479007721
           message: 'Optimization terminated successfully.'
              nfev: 122
               nit: 67
            status: 0
           success: True
                 x: array([ 1.1548077 , -0.61330039])




```python
X_pred = vae.decoder(torch.FloatTensor(res.x).to(device)).cpu().detach().numpy()
```


```python
fig, (ax0, ax1) = plt.subplots(figsize=(10, 4), ncols=2)
ax0.plot(grid, np.exp(X_target), label=f'α={α.item():.4f}, β={β.item():.4f}')
ax0.plot(grid, np.exp(X_pred), label=f'c1={res.x[0].item():.4f}, c_2={res.x[1].item():.4f}')
ax0.legend()
ax1.plot(grid, np.exp(X_target) - np.exp(X_pred))
ax0.set_xlabel('x')
ax0.set_ylabel('β(x; α, β)')
ax0.set_title('PDF')
ax1.set_xlabel('x')
ax1.set_ylabel('Error')
ax1.set_title('β(x; α, β) - β_NN(x; c_1, c_2)')
fig.tight_layout()
```


    
![png](/assets/images/variational-autoencoders/variational-autoencoders-3.png)
    


The results isn't too bad -- true, the reconstructed curve oscillates a bit, but at a small scale. We can say that the encoder has managed to compress the input data to two parameters and the decoder to define how to build the PDF of the beta distribution from those.
