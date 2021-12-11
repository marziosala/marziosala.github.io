---
layout: splash
permalink: /conditional-variational-autoencoders/
title: "Conditional Variational Autoencoders"
header:
  overlay_image: /assets/images/conditional-variational-autoencoders/conditional-variational-autoencoders-splash.png
excerpt: "Conditional variational autoencoders applied to the classical MNIST dataset."
---

![](conditional-variational-autoencoders-net.png)


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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device {device}")
```

    Using device cuda
    


```python
def to_one_hot(index, n):

    assert torch.max(index).item() < n

    if index.dim() == 1:
        index = index.unsqueeze(1)
    onehot = torch.zeros(index.size(0), n).to(index.device)
    onehot.scatter_(1, index, 1)
    
    return onehot
```


```python
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims, num_labels):
        super().__init__()
        self.linear = nn.Linear(784 + num_labels, 256)
        self.linear_mu = nn.Linear(256, latent_dims)
        self.linear_sigma = nn.Linear(256, latent_dims)
        self.num_labels = num_labels
        
        self.N = torch.distributions.Normal(0, 1)
        if device == 'cuda':
            self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()

    def forward(self, x, c):
        x = torch.flatten(x, start_dim=1)
        c = to_one_hot(c, self.num_labels)
        x = torch.cat((x, c), dim=1)
        x = F.relu(self.linear(x))
        mu =  self.linear_mu(x)
        sigma = torch.exp(self.linear_sigma(x))
        z = mu + sigma * self.N.sample(mu.shape)
        kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, kl
```


```python
class Decoder(nn.Module):
    def __init__(self, latent_dims, num_labels):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims + num_labels, 256)
        self.linear2 = nn.Linear(256, 784)
        self.num_labels = num_labels

    def forward(self, z, c):
        c = to_one_hot(c, self.num_labels)
        z = torch.cat((z, c), dim=1)
        z = torch.sigmoid(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))
```


```python
class CVAE(nn.Module):
    def __init__(self, latent_dims, num_labels):
        super().__init__()
        self.encoder = VariationalEncoder(latent_dims, num_labels)
        self.decoder = Decoder(latent_dims, num_labels)
        self.num_labels = num_labels

    def forward(self, x, c):
        z, kl = self.encoder(x, c)
        return self.decoder(z, c), kl
```


```python
def train(cvae, data, epochs=20, lr=1e-4):
    opt = torch.optim.Adam(cvae.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in data:
            x, y = x.to(device), y.long().to(device)
            opt.zero_grad()
            x_hat, kl = cvae(x, y)
            loss = ((x - x_hat)**2).sum() + kl
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch + 1:3d}, loss: {total_loss:.4e}")
```


```python
data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data',
               transform=torchvision.transforms.ToTensor(),
               download=True),
        batch_size=128,
        shuffle=True)
```


```python
latent_dims = 2
cvae = CVAE(latent_dims, 10).to(device) # GPU
train(cvae, data, epochs=100, lr=1e-3)
```

    Epoch:  10, loss: 2.1029e+06
    Epoch:  20, loss: 1.9389e+06
    Epoch:  30, loss: 1.8888e+06
    Epoch:  40, loss: 1.8625e+06
    Epoch:  50, loss: 1.8480e+06
    Epoch:  60, loss: 1.8375e+06
    Epoch:  70, loss: 1.8311e+06
    Epoch:  80, loss: 1.8251e+06
    Epoch:  90, loss: 1.8219e+06
    Epoch: 100, loss: 1.8178e+06
    Wall time: 9min 28s
    


```python
def plot_label(label, ax):
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    for x, y in data:
        subset = y == label
        if sum(subset) == 0: continue
        x_label = x[subset].to(device)
        y_label = y[subset].to(device)
        z_label, _ = cvae.encoder(x_label, y_label)
        z_label = z_label.to('cpu').detach().numpy()
        ax.scatter(z_label[:, 0], z_label[:, 1], c=colors[label], s=4, alpha=0.5, cmap='tab10')
```


```python
fig, axes = plt.subplots(figsize=(10, 20), nrows=5, ncols=2, sharex=True, sharey=True)
for row in range(5):
    for col in range(2):
        label = col + row * 2
        ax = axes[row, col]
        plot_label(label, ax)
        ax.set_title(f'Digit: {label}')
    axes[row, 0].set_ylabel('Z_0')
axes[4, 0].set_xlabel('Z_1')
axes[4, 1].set_xlabel('Z_1');
```




    Text(0.5, 0, 'Z_1')




    
![png](/assets/images/conditional-variational-autoencoders/conditional-variational-autoencoders-1.png)
    



```python
def plot_reconstructed(cvae, digit, ax, r0=(-3, 3), r1=(-3, 3), n=20):
    w = 28
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = cvae.decoder(z, torch.full_like(z, digit).long())
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    ax.imshow(img, extent=[*r0, *r1])
```


```python
fig, axes = plt.subplots(figsize=(8, 20), nrows=5, ncols=2, sharex=True, sharey=True)
for row in range(5):
    for col in range(2):
        digit = col + row * 2
        ax = axes[row, col]
        plot_reconstructed(cvae, digit, ax, r0=(-2, 2), r1=(-2, 2), n=20)
        ax.set_title(f'Digit: {digit}')
fig.tight_layout()
```


    
![png](/assets/images/conditional-variational-autoencoders/conditional-variational-autoencoders-2.png)
    

