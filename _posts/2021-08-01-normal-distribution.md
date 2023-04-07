---
layout: splash
permalink: /normal-distribution/
title: "Autoencoders"
header:
  overlay_image: /assets/images/normal-distribution/normal-distribution-splash.png
excerpt: "Autoencoders applied to mathematical functions."
---

An *autoencoder* is a type of neural network that can learn efficient representations of data. The learning is unsupervised -- they try to reconstruct the inputs from the inputs themselves.

In its simplest version, it is composed by two neural networks stacked on the top of each other: an *encoder* which compresses the data to a smaller dimensional encoding, and a *decoder*, which tries to reconstruct the original data from this encoding.

Most of the tutorials on autoencoders take images. Indeed, images are a very important application of generative machine learning models; however, here we will work with the reconstruction of a well-known mathematical function: the probability density function of the [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution),

$$
Φ(x; μ, σ) = \frac{1}{\sqrt{2 \pi} \sigma} \exp \left(-\frac{1}{2}\left( \frac{x - \mu}{\sigma} \right)^2 \right).
$$

This function has two parameters, $\mu$ and $\sigma$, apart from the input variable $x$. What we do is to create a grid of points $[x_1, x_2, \ldots, x_n]$ on which we will sample $Φ(x; μ, σ)$ for given values of $\mu$ and $\sigma$; this vector of $n$ points is the quantity that the autoencoder will operator upon. 
Since there are only two parameters we should be able to capture it with an autoencoder with two latent variables, meaning that the encoder will project the $n$ points into 2 latent variables, while the decoder will do the opposite, moving from the two laten variables to $n$ points, idelly very close to the input ones.

Our procedure is the following: 
- the parameters $\mu$ and $\sigma$ are sampled, the first in $\mathcal{U}(-3, 3)$ and the second
in $\mathcal{U}(0.5, 3)$;
- for given values of $\mu$ and $\sigma$, we compute the values $y(x)$ on a (fixed) grid of points;
- we train an autoencoder and check the quality of the fit.

The following packages were used:

```powershell
conda create --name normal-distribution python==3.9 --no-default-packages -y
conda activate normal-distribution
pip install torch numpy scipy matplotlib seaborn
```


```python
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.utils.data import Dataset, DataLoader
import torch.distributions
```

We specify a `device` such that we can use CPUs, GPUs, or anything else supported by PyTorch.


```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device {device}.")
```

    Using device cpu.
    

The first phase is the generation of the training dataset. The dataset is composed by a vector sampling out target function, on a predefined grid, for some values of $\mu$ and $\sigma$ that are given by a random number generator.


```python
uniform_dist_μ = torch.distributions.uniform.Uniform(-3.0, 3.0)
uniform_dist_σ = torch.distributions.uniform.Uniform(0.5, 3.0)
```


```python
num_inputs = 100_000
num_points = 1_000
```


```python
grid = torch.linspace(-5.0, 5.0, num_points)
```


```python
def target_func(x, μ, σ):
    return 1 / np.sqrt(2 * np.pi) / σ * np.exp(-0.5 * (x - μ)**2 /  σ**2)
```


```python
X, Y = [], []
for _ in range(num_inputs):
    μ = uniform_dist_μ.sample()
    σ = uniform_dist_σ.sample()
    X.append(target_func(grid, μ, σ))
    Y.append([μ, σ])
X = torch.vstack(X)

assert X.shape[0] == num_inputs
assert X.shape[1] == num_points
assert X.isnan().sum() == 0.0
```

Plotting a few entries of the dataset shows what we have. For the selected parameter ranges, the functions are smooth, yet potentially with large gradients when $\sigma$ is small.


```python
for x, (μ, σ) in zip(X[:5], Y[:5]):
    plt.plot(grid, x, label=f'μ={μ.item():.4f}, σ={σ.item():.4f}')
plt.legend()
plt.xlabel('x')
plt.ylabel('Φ(x)');
```


    
![png](/assets/images/normal-distribution/normal-distribution-1.png)
    


The architecture of the encoder is quite simple: a classical sequential network with four layers and `tanh` activation function. The decoder is symmetric and quite simple as well. As we will see, this suffices for our goals so we won't try more complicated architectures.


```python
class Encoder(nn.Module):
    def __init__(self, latent_dims, num_hidden):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(num_points, num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_hidden)
        self.linear3 = nn.Linear(num_hidden, num_hidden)
        self.linear4 = nn.Linear(num_hidden, latent_dims)

    def forward(self, x):
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        x = F.tanh(self.linear3(x))
        return self.linear4(x)
```


```python
class Decoder(nn.Module):
    def __init__(self, latent_dims, num_hidden):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_hidden)
        self.linear3 = nn.Linear(num_hidden, num_hidden)
        self.linear4 = nn.Linear(num_hidden, num_points)

    def forward(self, z):
        z = F.tanh(self.linear1(z))
        z = F.tanh(self.linear2(z))
        z = F.tanh(self.linear3(z))
        return self.linear4(z)
```

The autoencoder composed the encoder and the decoder, passing the input data through the former and then the latter.


```python
class Autoencoder(nn.Module):
    def __init__(self, latent_dims, num_hidden):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dims, num_hidden)
        self.decoder = Decoder(latent_dims, num_hidden)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
```

We are almost ready to do the training. A small wrapper to the `Dataset` class is used such that we can define a `DataLoader` and specify the batch size (here, 256) and shuffling.


```python
class FuncDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]
```


```python
data_loader = DataLoader(FuncDataset(X), batch_size=256, shuffle=True)
```


```python
def train(autoencoder, data_loader, epochs, lr, gamma, print_every):
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    history = []
    for epoch in range(1, epochs + 1):
        last_lr = scheduler.get_last_lr()
        total_loss = 0.0
        for x in data_loader:
            x = x.to(device)  # to GPU if necessary
            optimizer.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        total_loss /= len(data_loader)
        if epoch % print_every == 0:
            print(f"Epoch: {epoch:3d}, lr: {last_lr[0]:.4e}, avg loss: {total_loss:.4f}")
        history.append(total_loss)
    return history
```


```python
latent_dims = 2
autoencoder = Autoencoder(latent_dims=latent_dims, num_hidden=16).to(device)
```


```python
history = train(autoencoder, data_loader, epochs=200, lr=1e-3, gamma=0.98, print_every=10)
```

    Epoch:  10, lr: 8.3375e-04, avg loss: 37.3808
    Epoch:  20, lr: 6.8123e-04, avg loss: 11.0866
    Epoch:  30, lr: 5.5662e-04, avg loss: 6.4911
    Epoch:  40, lr: 4.5480e-04, avg loss: 5.0347
    Epoch:  50, lr: 3.7160e-04, avg loss: 4.1864
    Epoch:  60, lr: 3.0363e-04, avg loss: 3.6390
    Epoch:  70, lr: 2.4808e-04, avg loss: 3.2396
    Epoch:  80, lr: 2.0270e-04, avg loss: 2.9316
    Epoch:  90, lr: 1.6562e-04, avg loss: 2.6938
    Epoch: 100, lr: 1.3533e-04, avg loss: 2.5059
    Epoch: 110, lr: 1.1057e-04, avg loss: 2.3567
    Epoch: 120, lr: 9.0345e-05, avg loss: 2.2413
    Epoch: 130, lr: 7.3818e-05, avg loss: 2.1486
    Epoch: 140, lr: 6.0315e-05, avg loss: 2.0738
    Epoch: 150, lr: 4.9282e-05, avg loss: 2.0130
    Epoch: 160, lr: 4.0267e-05, avg loss: 1.9625
    Epoch: 170, lr: 3.2901e-05, avg loss: 1.9240
    Epoch: 180, lr: 2.6882e-05, avg loss: 1.8926
    Epoch: 190, lr: 2.1965e-05, avg loss: 1.8665
    Epoch: 200, lr: 1.7947e-05, avg loss: 1.8446
    


```python
plt.semilogy(history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
```




    [<matplotlib.lines.Line2D at 0x24b75f05dc0>]




    
![png](/assets/images/normal-distribution/normal-distribution-2.png)
    


It is important at this point to look at the latent space. as we have no control on how to is defined and which shape it has. To do that, we apply the encoder to all out dataset and store the results in the `Z` array; we then plot the distribution of $Z_1$ and $Z_2$, as well as all the points $(z_1, z_2)$. From the first two graphs we can appreciate that the first latent dimension goes from about -3 to 2, while the second from -4 to about 1.5. The third graph is the most interesting: the points have a peculiar shape and are not well distributed around the origin. The dashed red line represents the $(-1, 1) \times (-1, 1)$ square, and we can see that the lower left corner has not been touched in training. This means that points generated by the decoder when $z_1=-1$ and $z_2=-1$ will be based on extrapolation rather than interpolation and will probably be of poor quality.


```python
Z = []
for entry in X:
    Z.append(autoencoder.encoder(entry.to(device)).cpu().detach().numpy())
Z = np.array(Z)

fig, (ax0, ax1, ax2) = plt.subplots(figsize=(12, 4), ncols=3)
sns.histplot(Z[:, 0], ax=ax0, stat='density')
sns.histplot(Z[:, 1], ax=ax1, stat='density')
ax2.scatter(Z[:, 0], Z[:, 1], color='salmon')
ax0.set_xlabel('$z_1$')
ax1.set_xlabel('$z_2$')
ax2.set_xlabel('$z_1$'); ax2.set_ylabel('$z_2$')
ax2.add_patch(Rectangle((-1, -1), 2, 2, linestyle='dashed', color='red', alpha=1, fill=None))
fig.tight_layout()
```


    
![png](/assets/images/normal-distribution/normal-distribution-3.png)
    


It is now time to verify the quality of our decoder. We define new random values for $\mu$ and $\sigma$ and look for the latent variables that provide the best fit. The search is performed using SciPy's `minimize` and is extremely fast.


```python
μ = uniform_dist_μ.sample()
σ = uniform_dist_σ.sample()

print(f'Params for the test: μ={μ.item():.4f}, σ={σ.item():.4f}\n')

X_target = target_func(grid, μ, σ).numpy()

def func(x):
    X_pred = autoencoder.decoder(torch.FloatTensor(x).to(device))
    X_pred = X_pred.cpu().detach().numpy()
    diff = np.linalg.norm(X_target - X_pred).item()
    return diff

res = minimize(func, [0.0, 0.0], method='Nelder-Mead')
print(res)
```

    Params for the test: μ=2.3721, σ=1.6699
    
           message: Optimization terminated successfully.
           success: True
            status: 0
               fun: 0.04667074233293533
                 x: [ 1.614e-01 -1.525e+00]
               nit: 68
              nfev: 133
     final_simplex: (array([[ 1.614e-01, -1.525e+00],
                           [ 1.614e-01, -1.525e+00],
                           [ 1.614e-01, -1.525e+00]]), array([ 4.667e-02,  4.667e-02,  4.667e-02]))
    

Plotting the results shows quite a good agreement between the exact solution and the one produced by the decoder; increasing the size of the training dataset or the number of epoch would increase the quality.


```python
X_pred = autoencoder.decoder(torch.FloatTensor(res.x).to(device)).cpu().detach().numpy()
fig, (ax0, ax1) = plt.subplots(figsize=(10, 4), ncols=2)
ax0.plot(grid, X_target, label=f'α={α.item():.4f}, β={β.item():.4f}')
ax0.plot(grid, X_pred, label=f'$z_1={res.x[0].item():.4f}, z_2={res.x[1].item():.4f}$')
ax0.legend(loc='upper left')
ax1.plot(grid, X_target - X_pred)
ax0.set_xlabel('x')
ax0.set_ylabel('Φ(x; μ, σ)')
ax0.set_title('Function')
ax1.set_xlabel('x')
ax1.set_ylabel('$Φ(x; μ, σ) - \hat Φ(x; z_1, z_2)$')
ax1.set_title('Error')
fig.tight_layout()
```


    
![png](/assets/images/normal-distribution/normal-distribution-4.png)
    


And finally a nice example of what each latent variable represents by plotting, on a 10 by 10 grid, the result of the decoder for several values of $z_1$ and $z_2$, both ranging from -1 to 1. As we discussed above, the bottom left corner was not covered during the training and indeed the results aren't very meaningful; otherise we can see our distributions changing with the two parameters. 


```python
n = 10
fig, axes = plt.subplots(nrows=n, ncols=n, figsize=(8, 8), sharex=True, sharey=True)
for i, c_1 in enumerate(np.linspace(-1, 1, n)):
    for j, c_2 in enumerate(np.linspace(-1, 1, n)):
        y = autoencoder.decoder(torch.FloatTensor([c_1, c_2]).to(device)).detach().cpu()
        axes[n - 1 - j, i].plot(grid, y)
        axes[n - 1 - j, i].axis('off')
        axes[n - 1 - j, i].set_title(f'{c_1:.2f},{c_2:.2f}', fontsize=6)
fig.tight_layout()
```


    
![png](/assets/images/normal-distribution/normal-distribution-5.png)
    

