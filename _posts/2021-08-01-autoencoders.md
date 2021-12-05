---
layout: splash
permalink: /autoencoders/
title: "Autoencoders"
header:
  overlay_image: /assets/images/autoencoders/autoencoders-splash.png
excerpt: "Autoencoders applied to mathematical functions."
---

An *autoencoder* is a type of neural network that can learn efficient representations of data. The learning is unsupervised -- they try to reconstruct the inputs from the inputs themselves.

In its simplest version, it is composed by two neural networks stacked on the top of each other: an *encoder* which compresses the data to a smaller dimensional encoding, and a *decoder*, which tries to reconstruct the original data from this encoding.

Most of the tutorials on autoencoders take images. Indeed, images are a very important application of generative machine learning models; however, here we will work with the reconstruction of mathematical functions. What we want to do is the following: 
- we consider the beta distribution, which depends on $x \in (0, 1)$ and also on two parameters $\alpha$ and $\beta$;
- the parameters $\alpha$ and $\beta$ are sampled in $\mathcal{U}(0, 5)$, so uniformly between 0 and 5;
- for given values of $\alpha$ and $\beta$, we compute the PDF of the beta distribution on a (fixed) grid of points.

The last step is repeated 


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
device = 'cpu' #cuda' if torch.cuda.is_available() else 'cpu'
```


```python
uniform_dist = torch.distributions.uniform.Uniform(0, 5)
```


```python
num_inputs = 10_000
num_points = 100
```


```python
grid = torch.linspace(0.01, 0.99, num_points)
```


```python
def target_func(grid, Î±, Î²):
    return torch.distributions.Beta(Î±, Î²).log_prob(grid)
```


```python
X, Y = [], []
for _ in range(num_inputs):
    Î± = uniform_dist.sample()
    Î² = uniform_dist.sample()
    X.append(target_func(grid, Î±, Î²))
    Y.append([Î±, Î²])
X = torch.vstack(X)

assert X.shape[0] == num_inputs
assert X.shape[1] == num_points
assert X.isnan().sum() == 0.0
```


```python
for x, (Î±, Î²) in zip(X[:5], Y[:5]):
    plt.plot(grid, x.exp(), label=f'Î±={Î±.item():.4f}, Î²={Î².item():.4f}')
plt.legend()
```




    <matplotlib.legend.Legend at 0x1e391558488>




    
![png](/assets/images/autoencoders/autoencoders_8_1.png)
    



```python
num_hidden = 128
```


```python
class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(num_points, num_hidden)
        self.linear2 = nn.Linear(num_hidden, latent_dims)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)
```


```python
class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_points)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        return self.linear2(z)  # can use F.relu() as well as this is a CDF
```


```python
class Autoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
```


```python
def train(autoencoder, data, epochs=20, lr=1e-3, gamma=0.95):
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    history = []
    for epoch in range(epochs):
        last_lr = scheduler.get_last_lr()
        total_loss = 0.0
        for x in data:
            x = x.to(device) # GPU
            optimizer.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch: {epoch:3d}, lr=: {last_lr[0]:.4e}, total loss: {total_loss:.4f}")
        history.append(total_loss)
    return history
```


```python
%%time
latent_dims = 2
autoencoder = Autoencoder(latent_dims).to(device)
history = train(autoencoder, X, epochs=20, lr=1e-4, gamma=0.99)
```

    Epoch:   0, lr=: 0.0001, loss: 259696.3656
    Epoch:   1, lr=: 0.0001, loss: 9327.7636
    Epoch:   2, lr=: 0.0001, loss: 5177.8648
    Epoch:   3, lr=: 0.0001, loss: 3849.1633
    Epoch:   4, lr=: 0.0001, loss: 3077.2303
    Epoch:   5, lr=: 0.0001, loss: 3008.3965
    Epoch:   6, lr=: 0.0001, loss: 2607.3506
    Epoch:   7, lr=: 0.0001, loss: 2554.9801
    Epoch:   8, lr=: 0.0001, loss: 2244.1520
    Epoch:   9, lr=: 0.0001, loss: 2170.8816
    Epoch:  10, lr=: 0.0001, loss: 2083.4034
    Epoch:  11, lr=: 0.0001, loss: 2126.0767
    Epoch:  12, lr=: 0.0001, loss: 1858.4479
    Epoch:  13, lr=: 0.0001, loss: 2036.4268
    Epoch:  14, lr=: 0.0001, loss: 1777.3554
    Epoch:  15, lr=: 0.0001, loss: 1739.6955
    Epoch:  16, lr=: 0.0001, loss: 1833.3813
    Epoch:  17, lr=: 0.0001, loss: 1676.8849
    Epoch:  18, lr=: 0.0001, loss: 1740.7370
    Epoch:  19, lr=: 0.0001, loss: 1745.3988
    Wall time: 3min 42s
    


```python
plt.plot(history)
```




    [<matplotlib.lines.Line2D at 0x1e3916dc348>]




    
![png](/assets/images/autoencoders/autoencoders_15_1.png)
    



```python
C = []
for entry in X:
    C.append(autoencoder.encoder(entry.to(device)).cpu().detach().numpy())
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


    
![png](/assets/images/autoencoders/autoencoders_17_0.png)
    



```python
from scipy.optimize import minimize
```


```python
Î± = torch.tensor(2.0) # uniform_dist.sample()
Î² = torch.tensor(4.0) # uniform_dist.sample()

print(f'Params for the test: Î±={Î±.item():.4f}, Î²={Î².item():.4f}')

X_target = target_func(grid, Î±, Î²).numpy()

def func(x):
    X_pred = autoencoder.decoder(torch.FloatTensor(x).to(device))
    X_pred = X_pred.cpu().detach().numpy()
    diff = np.linalg.norm(X_target - X_pred).item()
    return diff
```

    Params for the test: Î±=2.0000, Î²=4.0000
    


```python
res = minimize(func, [0.0, 0.0], method='Nelder-Mead')
res
```




     final_simplex: (array([[-3.44212426, -5.51789163],
           [-3.4421665 , -5.51785568],
           [-3.44207613, -5.51784012]]), array([0.087826  , 0.08782606, 0.0878263 ]))
               fun: 0.08782599866390228
           message: 'Optimization terminated successfully.'
              nfev: 150
               nit: 78
            status: 0
           success: True
                 x: array([-3.44212426, -5.51789163])




```python
X_pred = autoencoder.decoder(torch.FloatTensor(res.x).to(device)).cpu().detach().numpy()
```


```python
fig, (ax0, ax1) = plt.subplots(figsize=(10, 4), ncols=2)
ax0.plot(grid, np.exp(X_target), label=f'Î±={Î±.item():.4f}, Î²={Î².item():.4f}')
ax0.plot(grid, np.exp(X_pred), label=f'c1={res.x[0].item():.4f}, c_2={res.x[1].item():.4f}')
ax0.legend()
ax1.plot(grid, np.exp(X_target) - np.exp(X_pred))
ax0.set_xlabel('x')
ax0.set_ylabel('Î²(x; Î±, Î²)')
ax0.set_title('PDF')
ax1.set_xlabel('x')
ax1.set_ylabel('Error')
ax1.set_title('Î²(x; Î±, Î²) - Î²_NN(x; c_1, c_2)')
fig.tight_layout()
```


    
![png](/assets/images/autoencoders/autoencoders_22_0.png)
    

