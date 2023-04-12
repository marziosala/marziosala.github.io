---
layout: splash
permalink: /beta-distribution/
title: "Variational Autoencoders"
header:
  overlay_image: /assets/images/beta-distribution/beta-distribution-splash.png
excerpt: "Variational autoencoders applied to mathematical functions."
---

In the previous post we have seen how autoencoders work. As we noticed, in general the encoding space is a non-convex manifold and the codes have arbitrary scales. This makes basic autoencoders a poor choice for generative models. *Variational autoencoders* fix this issue by ensuring that the coding space follows a desirable distribution from which we can easily sample from. This distribution typically is the standard normal distribution. So, if we have $N_C$ encodings, our goal is to have $N_C$ independent and normally distributed encodings. In order to do that, we want to learn the mean $\mu$ and the standard deviation $\sigma$ that are close to that of a standard normal distribution, while at the same time having outputs that are close to the inputs.

A useful trick to achieve what we want is to write the $i-$th component of the encoding vector
$Z=(z_1, z_2, \ldots, z_C) $ as
$$
z_i = \mu_i  + \sigma_i \cdot \xi,
$$
where $\xi \sim N(0,1)$ is a random number coming from the standard normal distribution. Why do we do this? Because the random number generation is a non-differentiable operation, so if we did specify the mean and standard deviation of the distribution directly we would not be able to backpropagate the gradients.

The architecture is reported in the picture below. Note that the dense layer is connected to two separate blocks, one of which generates $\mu$ and the other $\sigma$. The part is the *encoder*. Once $Z$ has been generated, we enter the *decoder* that rebuilds the inputs from the encodings. 

![](variational-autoencoders-net.png)

The implementation is quite close to that of an autoencoder. The loss function is trickier though as now we have to goals:
- a good reconstruction of the inputs; and
- a coding space that is normally distributed.

The first goal is treated as for the autoencoder; the second requires us to compute the distance between two probability distributions, and this is done using the Kullback-Leibler divergence. If we assume that the encodings are independent we can simplify our analysis and work with univariate $\mu$ and $\sigma$.

The Kullback-Leibler divergence between two probability distributions $P$ and $Q$ is defined as

$$
D_{KL}(P || Q) = \int_\mathcal{X} p(x) \ln \frac{q(x)}{p(x)} dx
$$

where $p(x)$ and $q(x)$ are the probability density functions pf $P$ and $Q$, respectively, and the integral is taken over the sample space $\mathcal{X}$. For us, $P$ and $Q$ are Gaussians, so $\mathcal{X} = \mathbb{R}$, and

$$
\begin{align}
p(x) & = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( -\frac{(x - \mu)^2}{2 \sigma^2} \right) \\
q(x) & = \frac{1}{\sqrt{2 \pi}} \exp \left( -\frac{x^2}{2} \right).
\end{align}
$$

Plugging the definitions of $p(x)$ and $q(x)$ into the equation of the KL divergence gives

$$
\begin{align}
D_{KL}(P || Q) & = \int_\mathcal{R} (\ln p(x) -\ln q(x)) p(x) dx \\
& = \int_\mathcal{R} \left[
\ln \frac{1}{\sigma^2} - \frac{1}{2} \left( \frac{x - \mu}{\sigma} \right)^2 + \frac{1}{2} x^2
\right] p(x) dx \\
& = \mathbb{E}_P\left[ \ln \frac{1}{\sigma} + \frac{1}{2} X^2 - \frac{1}{2} 
\left( \frac{X - \mu}{\sigma} \right)^2
\right] \\
& = \ln \frac{1}{\sigma} + \frac{1}{2} (\sigma^2 + \mu^2) + \frac{1}{2},
\end{align}
$$

after noting that

$$
\begin{align}
\mathbb{E}_P[X^2] & = \mathbb{E}_P[X^2 - 2 \mu X + \mu^2 + 2 \mu X - \mu^2] \\
& = \mathbb{E}_P[(X - \mu)^2] + 2 \mu \mathbb{E}_P[X] - \mu^2 \\
& = \sigma^2 + 2 \mu^2 - \mu^2 \\
& = \sigma^2 + \mu^2.
\end{align}
$$

Therefore,

$$
D_{KL}(P || Q) = \frac{1}{2} \left( \mu^2 + \sigma^2 -1 - \ln \sigma^2 \right)
$$

which we can easily compute given $\mu$ and $\sigma$.


```python
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
from torch.utils.data import Dataset, DataLoader
```


```python
device = 'cpu' # if torch.cuda.is_available() else 'cpu'
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
def target_func(grid, α, β):
    # return the log of the probability, not the probability itself
    return torch.distributions.Beta(α, β).log_prob(grid)
```


```python
torch.manual_seed(0)
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




    <matplotlib.legend.Legend at 0x16e68326b80>




    
![png](/assets/images/beta-distribution/beta-distribution-1.png)
    



```python
normal_dist = torch.distributions.Normal(0, 1)
if device == 'cuda':
    normal_dist.loc = normal_dist.loc.cuda()  # hack to get sampling on the GPU
    normal_dist.scale = normal_dist.scale.cuda()
```


```python
class VariationalEncoder(nn.Module):

    def __init__(self, latent_dims, num_hidden):
        super().__init__()
        self.linear1 = nn.Linear(num_points, num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_hidden)
        self.linear3 = nn.Linear(num_hidden, num_hidden)
        self.linear_mu = nn.Linear(num_hidden, latent_dims)
        self.linear_logsigma2 = nn.Linear(num_hidden, latent_dims)

    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        mu =  self.linear_mu(x)
        logsigma2 = self.linear_logsigma2(x)
        sigma = torch.exp(logsigma2 / 2)
        z = mu + sigma * normal_dist.sample(mu.shape)
        integrand = (sigma**2 + mu**2 - logsigma2 - torch.ones_like(sigma)) / 2
        kl = torch.mean(torch.sum(integrand, axis=1), axis=0)
        return z, kl
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
        z = torch.tanh(self.linear1(z))
        z = torch.tanh(self.linear2(z))
        z = torch.tanh(self.linear3(z))
        return self.linear4(z)
```


```python
class VariationalAutoencoder(nn.Module):

    def __init__(self, latent_dims, num_hidden):
        super().__init__()
        self.encoder = VariationalEncoder(latent_dims, num_hidden)
        self.decoder = Decoder(latent_dims, num_hidden)

    def forward(self, x):
        z, kl = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, kl
    
    def compute_log_likelihood(self, X, X_hat, η):
        ξ = normal_dist.sample(X_hat.shape)
        X_sampled = X_hat + η.sqrt() * ξ
        return 0.5 * F.mse_loss(X, X_sampled) / η #+ η.log()
```


```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
```


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
data_loader = DataLoader(FuncDataset(X), batch_size=32, shuffle=True)
```


```python
def train(vae, data, epochs, lr, gamma, η, β, print_every):
    η = torch.tensor(η)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    history = []
    for epoch in range(epochs):
        last_lr = scheduler.get_last_lr()
        total_log_loss, total_kl_loss, total_loss = 0.0, 0.0, 0.0
        for x in data:
            x = x.to(device)  # GPU
            optimizer.zero_grad()
            x_hat, kl_loss = vae(x)
            log_loss = vae.compute_log_likelihood(x, x_hat, η)
            loss = log_loss + β * kl_loss
            total_log_loss += log_loss.item()
            total_kl_loss += kl_loss.item()
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % print_every == 0:
            print(f"Epoch: {epoch + 1:3d}, lr: {last_lr[0]:.4e}, " \
                  f"total log loss: {total_log_loss:.4f}, total KL loss: {total_kl_loss:.4f}, " \
                  f"total loss: {total_loss:.4e}")
        history.append((total_log_loss, total_kl_loss))
    return np.array(history)
```


```python
torch.manual_seed(0)
data_loader = DataLoader(FuncDataset(X), batch_size=256, shuffle=True)
latent_dims = 2
vae = VariationalAutoencoder(latent_dims, num_hidden=32).to(device)
vae.apply(init_weights)
history = train(vae, data_loader, epochs=1_000, lr=1e-3, gamma=0.99, η=0.1**2, β=1, print_every=50)
```

    Epoch:  50, lr: 6.1112e-04, total log loss: 172.4695, total KL loss: 216.1071, total loss: 3.8858e+02
    Epoch: 100, lr: 3.6973e-04, total log loss: 87.6573, total KL loss: 210.6503, total loss: 2.9831e+02
    Epoch: 150, lr: 2.2369e-04, total log loss: 73.5103, total KL loss: 209.3993, total loss: 2.8291e+02
    Epoch: 200, lr: 1.3533e-04, total log loss: 69.3126, total KL loss: 208.9647, total loss: 2.7828e+02
    Epoch: 250, lr: 8.1877e-05, total log loss: 66.7518, total KL loss: 210.3983, total loss: 2.7715e+02
    Epoch: 300, lr: 4.9536e-05, total log loss: 65.2317, total KL loss: 208.2934, total loss: 2.7353e+02
    Epoch: 350, lr: 2.9970e-05, total log loss: 65.0135, total KL loss: 208.5284, total loss: 2.7354e+02
    Epoch: 400, lr: 1.8132e-05, total log loss: 64.8618, total KL loss: 208.3012, total loss: 2.7316e+02
    Epoch: 450, lr: 1.0970e-05, total log loss: 65.0291, total KL loss: 208.7231, total loss: 2.7375e+02
    Epoch: 500, lr: 6.6369e-06, total log loss: 64.8451, total KL loss: 208.4879, total loss: 2.7333e+02
    Epoch: 550, lr: 4.0153e-06, total log loss: 64.3602, total KL loss: 208.6472, total loss: 2.7301e+02
    Epoch: 600, lr: 2.4293e-06, total log loss: 63.9771, total KL loss: 208.3997, total loss: 2.7238e+02
    Epoch: 650, lr: 1.4697e-06, total log loss: 64.4275, total KL loss: 208.3166, total loss: 2.7274e+02
    Epoch: 700, lr: 8.8920e-07, total log loss: 64.6008, total KL loss: 208.7841, total loss: 2.7338e+02
    Epoch: 750, lr: 5.3797e-07, total log loss: 64.3160, total KL loss: 208.5707, total loss: 2.7289e+02
    Epoch: 800, lr: 3.2548e-07, total log loss: 63.8710, total KL loss: 208.4223, total loss: 2.7229e+02
    Epoch: 850, lr: 1.9692e-07, total log loss: 64.5717, total KL loss: 208.6061, total loss: 2.7318e+02
    Epoch: 900, lr: 1.1914e-07, total log loss: 63.6378, total KL loss: 207.9878, total loss: 2.7163e+02
    Epoch: 950, lr: 7.2077e-08, total log loss: 64.7554, total KL loss: 208.2642, total loss: 2.7302e+02
    Epoch: 1000, lr: 4.3607e-08, total log loss: 63.8177, total KL loss: 208.0312, total loss: 2.7185e+02
    


```python
fig, (ax0, ax1) = plt.subplots(figsize=(10, 4), ncols=2)
ax0.semilogy(history[:, 0])
ax0.set_xlabel('Epoch')
ax0.set_ylabel('Loglikelyhood Loss')
ax1.semilogy(history[:, 1])
ax1.set_xlabel('Epoch')
ax1.set_ylabel('KL Loss')
fig.tight_layout()
```


    
![png](/assets/images/beta-distribution/beta-distribution-2.png)
    



```python
Z = []
for entry in X:
    Z.append(vae.encoder(entry.unsqueeze(0).to(device))[0].cpu().detach().numpy().squeeze(0))
Z = np.array(Z)
```


```python
fig, (ax0, ax1, ax2) = plt.subplots(figsize=(12, 4), ncols=3)
sns.histplot(Z[:, 0], ax=ax0, stat='density')
sns.histplot(Z[:, 1], ax=ax1, stat='density')
ax2.scatter(Z[:, 0], Z[:, 1], color='salmon')
ax0.set_xlabel('$z_1$')
ax1.set_xlabel('$z_2$')
ax2.set_xlabel('$z_1$'); ax2.set_ylabel('$z_2$')
for r in [1, 2, 3, 4, 5]:
    ax2.add_patch(Circle((0, 0), r, linestyle='dashed', color='crimson', alpha=1, fill=None))
fig.tight_layout()
```


    
![png](/assets/images/beta-distribution/beta-distribution-3.png)
    


Both latent dimensions $Z_1$ and $Z_2$ have a distribution resembling the one of a Normal variate; their joint distribution is nicely scattered around the origin are roughtly within 3 to 4 standard deviations, as the red circles (with a radius of 1, 2, 3 and 4, respectively) show.
It is then much easier to generate new values or look for the optimal ones that match a given, which is what we try now to do. First, we generate two random values for $\alpha$ and $\beta$ and compute the corresponding exact PDF; then we use `scipy` to find the values of $(Z_1, Z_2)$ that produce the closest match.

As an example of what each latent variable represents we plot, on a 10 by 10 grid, the result of the decoder for several values of $z_1$ and $z_2$, both ranging from -1 to 1, where can see the shape of the Beta distributions changing with the two parameters as we would expect. The dotted grey line indicates the zero axis; all plots share the same scale on the Y axis.


```python
n = 10
fig, axes = plt.subplots(nrows=n, ncols=n, figsize=(8, 8), sharex=True, sharey=True)
for i, z_1 in enumerate(np.linspace(-3, 3, n)):
    for j, z_2 in enumerate(np.linspace(-3, 3, n)):
        y = vae.decoder(torch.FloatTensor([z_1, z_2]).to(device)).detach().cpu().exp()
        axes[n - 1 - j, i].plot(grid, y)
        axes[n - 1 - j, i].plot(grid, np.zeros_like(grid), color='grey', linestyle='dashed')
        axes[n - 1 - j, i].axis('off')
        axes[n - 1 - j, i].set_title(f'{z_1:.2f},{z_2:.2f}', fontsize=6)
        axes[n - 1 - j, i].set_ylim(0, 3)
fig.tight_layout()
```


    
![png](/assets/images/beta-distribution/beta-distribution-4.png)
    



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

res = minimize(func, [0.0, 0.0], method='Nelder-Mead')
res
```

    Params for the test: α=3.4180, β=0.2701
    




           message: Optimization terminated successfully.
           success: True
            status: 0
               fun: 0.26235488057136536
                 x: [-1.498e+00  6.091e-01]
               nit: 69
              nfev: 132
     final_simplex: (array([[-1.498e+00,  6.091e-01],
                           [-1.498e+00,  6.091e-01],
                           [-1.498e+00,  6.091e-01]]), array([ 2.624e-01,  2.624e-01,  2.624e-01]))




```python
X_pred = vae.decoder(torch.FloatTensor(res.x).to(device)).cpu().detach().numpy()
fig, (ax0, ax1) = plt.subplots(figsize=(10, 4), ncols=2)
ax0.plot(grid, np.exp(X_target), label=f'α={α.item():.4f}, β={β.item():.4f}')
ax0.plot(grid, np.exp(X_pred), label=f'$z_1$={res.x[0].item():.4f}, $z_2$={res.x[1].item():.4f}')
ax0.legend()
ax1.plot(grid, np.exp(X_target) - np.exp(X_pred))
ax0.set_xlabel('x')
ax0.set_ylabel('β(x; α, β)')
ax0.set_title('PDF')
ax1.set_xlabel('x')
ax1.set_title('Error')
ax1.set_ylabel('$β(x; α, β) - \hat{β}(x; z_1, z_2)$')
fig.tight_layout()
```


    
![png](/assets/images/beta-distribution/beta-distribution-5.png)
    


The results isn't too bad -- true, the reconstructed curve oscillates a bit, but at a small scale. the oscillations can be resolved with more training data and more iterations, or by increasing the size of the neural networks. We can say that the encoder has managed to compress the input data to two parameters and the decoder to define how to build the PDF of the beta distribution from those.

To conclude, two references that have largely inspired this contribution: https://avandekleut.github.io/vae/ for the code and https://mathybit.github.io/auto-var/ for the math.
