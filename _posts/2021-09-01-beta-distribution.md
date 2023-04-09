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
import torch; torch.manual_seed(0)
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




    <matplotlib.legend.Legend at 0x2034239a3a0>




    
![png](/assets/images/beta-distribution/beta-distribution-1.png)
    



```python
normal_dist = torch.distributions.Normal(0, 1)
if device == 'cuda':
    normal_dist.loc = normal_dist.loc.cuda() # hack to get sampling on the GPU
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
        self.kl = 0

    def forward(self, x):
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        x = F.tanh(self.linear3(x))
        mu =  self.linear_mu(x)
        logsigma2 = self.linear_logsigma2(x)
        sigma = torch.exp(logsigma2 / 2)
        z = mu + sigma * normal_dist.sample(mu.shape)
        self.kl = ((sigma**2 + mu**2 - logsigma2 - 1) / 2).sum()
        return z
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


```python
class VariationalAutoencoder(nn.Module):

    def __init__(self, latent_dims, num_hidden):
        super().__init__()
        self.encoder = VariationalEncoder(latent_dims, num_hidden)
        self.decoder = Decoder(latent_dims, num_hidden)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    def compute_log_likelihood(self, X, X_hat, η):
        ξ = normal_dist.sample(X_hat.shape)
        X_sampled = X_hat + η * ξ
        return 0.5 * ((X - X_sampled)**2).sum() / η**2 + η.log()
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
def train(vae, data, epochs=20, lr=1e-3, gamma=0.95, η=0.1, print_every=1):
    η = torch.tensor(η)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    history = []
    for epoch in range(epochs):
        last_lr = scheduler.get_last_lr()
        total_log_loss, total_kl_loss = 0.0, 0.0
        for x in data:
            x = x.to(device) # GPU
            optimizer.zero_grad()
            x_hat = vae(x)
            log_loss = vae.compute_log_likelihood(x, x_hat, η)
            kl_loss = vae.encoder.kl
            total_log_loss += log_loss.item()
            total_kl_loss += kl_loss.item()
            loss = log_loss + kl_loss
            loss /= num_points * len(x)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % print_every == 0:
            print(f"Epoch: {epoch + 1:3d}, lr=: {last_lr[0]:.4e}, " \
                  f"total log loss: {total_log_loss:.4f}, total KL loss: {total_kl_loss:.4f}")
        history.append((total_log_loss, total_kl_loss))
    return np.array(history)
```


```python
data_loader = DataLoader(FuncDataset(X), batch_size=256, shuffle=True)
latent_dims = 2
vae = VariationalAutoencoder(latent_dims, num_hidden=32).to(device)
```


```python
history = train(vae, data_loader, epochs=2_000, lr=1e-4, gamma=0.99, η=1, print_every=50)
```

    Epoch:  50, lr=: 6.1112e-05, total log loss: 517560.9767, total KL loss: 53777.0405
    Epoch: 100, lr=: 3.6973e-05, total log loss: 517397.8387, total KL loss: 53511.1262
    Epoch: 150, lr=: 2.2369e-05, total log loss: 516222.6875, total KL loss: 53464.8774
    Epoch: 200, lr=: 1.3533e-05, total log loss: 515707.3129, total KL loss: 53421.4163
    Epoch: 250, lr=: 8.1877e-06, total log loss: 516783.6189, total KL loss: 53319.7858
    Epoch: 300, lr=: 4.9536e-06, total log loss: 516166.5024, total KL loss: 53320.4863
    Epoch: 350, lr=: 2.9970e-06, total log loss: 516000.9357, total KL loss: 53301.2755
    Epoch: 400, lr=: 1.8132e-06, total log loss: 517115.9525, total KL loss: 53286.8647
    Epoch: 450, lr=: 1.0970e-06, total log loss: 515438.4694, total KL loss: 53273.6666
    Epoch: 500, lr=: 6.6369e-07, total log loss: 515070.7604, total KL loss: 53273.0131
    Epoch: 550, lr=: 4.0153e-07, total log loss: 514981.3275, total KL loss: 53279.6528
    Epoch: 600, lr=: 2.4293e-07, total log loss: 515832.8436, total KL loss: 53274.6857
    Epoch: 650, lr=: 1.4697e-07, total log loss: 515978.0646, total KL loss: 53273.6939
    Epoch: 700, lr=: 8.8920e-08, total log loss: 516658.6315, total KL loss: 53271.9757
    Epoch: 750, lr=: 5.3797e-08, total log loss: 514245.8837, total KL loss: 53274.7419
    Epoch: 800, lr=: 3.2548e-08, total log loss: 516539.5763, total KL loss: 53271.7818
    Epoch: 850, lr=: 1.9692e-08, total log loss: 515993.5084, total KL loss: 53272.6352
    Epoch: 900, lr=: 1.1914e-08, total log loss: 514139.7813, total KL loss: 53272.5826
    Epoch: 950, lr=: 7.2077e-09, total log loss: 515917.3399, total KL loss: 53272.7743
    Epoch: 1000, lr=: 4.3607e-09, total log loss: 516738.8276, total KL loss: 53272.5610
    Epoch: 1050, lr=: 2.6383e-09, total log loss: 515481.4313, total KL loss: 53272.5339
    Epoch: 1100, lr=: 1.5962e-09, total log loss: 515403.8425, total KL loss: 53272.4816
    Epoch: 1150, lr=: 9.6569e-10, total log loss: 515286.4664, total KL loss: 53272.4920
    Epoch: 1200, lr=: 5.8425e-10, total log loss: 513969.0332, total KL loss: 53272.4972
    Epoch: 1250, lr=: 3.5347e-10, total log loss: 515212.8973, total KL loss: 53272.4982
    Epoch: 1300, lr=: 2.1385e-10, total log loss: 514161.1364, total KL loss: 53272.4985
    Epoch: 1350, lr=: 1.2938e-10, total log loss: 514198.2072, total KL loss: 53272.4977
    Epoch: 1400, lr=: 7.8278e-11, total log loss: 514654.1814, total KL loss: 53272.4980
    Epoch: 1450, lr=: 4.7358e-11, total log loss: 516021.1305, total KL loss: 53272.4986
    Epoch: 1500, lr=: 2.8652e-11, total log loss: 514222.7737, total KL loss: 53272.4985
    Epoch: 1550, lr=: 1.7335e-11, total log loss: 513725.6606, total KL loss: 53272.4987
    Epoch: 1600, lr=: 1.0488e-11, total log loss: 516049.3605, total KL loss: 53272.4984
    Epoch: 1650, lr=: 6.3451e-12, total log loss: 515978.9597, total KL loss: 53272.4981
    Epoch: 1700, lr=: 3.8388e-12, total log loss: 514534.2349, total KL loss: 53272.4987
    Epoch: 1750, lr=: 2.3225e-12, total log loss: 516498.9371, total KL loss: 53272.4989
    Epoch: 1800, lr=: 1.4051e-12, total log loss: 514640.8371, total KL loss: 53272.4980
    Epoch: 1850, lr=: 8.5011e-13, total log loss: 515816.7416, total KL loss: 53272.4977
    Epoch: 1900, lr=: 5.1432e-13, total log loss: 515970.1772, total KL loss: 53272.4981
    Epoch: 1950, lr=: 3.1117e-13, total log loss: 516229.1086, total KL loss: 53272.4984
    Epoch: 2000, lr=: 1.8826e-13, total log loss: 515762.4912, total KL loss: 53272.4978
    


```python
history = np.array(history)
```


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
    Z.append(vae.encoder(entry.to(device)).cpu().detach().numpy())
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
    


Both latent dimensions $z_1$ and $z_2$ have a distribution resembling the one of a Normal variate; their joint distribution is nicely scattered around the origin are roughtly within 3 to 4 standard deviations, as the red circles (with a radious of 1, 2, 3 and 4, respectively) show.
It is then much easier to generate new values or look for the optimal ones that match a given, which is what we try now to do. First, we generate two random values for $\alpha$ and $\beta$ and compute the corresponding exact PDF; then we use `scipy` to find the values of $(z_1, z_2)$ that produce the closest match.


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

    Params for the test: α=2.2155, β=2.5044
    




           message: Optimization terminated successfully.
           success: True
            status: 0
               fun: 0.13826389610767365
                 x: [-3.881e-01 -9.114e-02]
               nit: 54
              nfev: 101
     final_simplex: (array([[-3.881e-01, -9.114e-02],
                           [-3.880e-01, -9.117e-02],
                           [-3.880e-01, -9.109e-02]]), array([ 1.383e-01,  1.383e-01,  1.383e-01]))




```python
X_pred = vae.decoder(torch.FloatTensor(res.x).to(device)).cpu().detach().numpy()
fig, (ax0, ax1) = plt.subplots(figsize=(10, 4), ncols=2)
ax0.plot(grid, np.exp(X_target), label=f'α={α.item():.4f}, β={β.item():.4f}')
ax0.plot(grid, np.exp(X_pred), label=f'c1={res.x[0].item():.4f}, c_2={res.x[1].item():.4f}')
ax0.legend()
ax1.plot(grid, np.exp(X_target) - np.exp(X_pred))
ax0.set_xlabel('x')
ax0.set_ylabel('β(x; α, β)')
ax0.set_title('PDF')
ax1.set_xlabel('x')
ax1.set_title('Error')
ax1.set_xlabel('$β(x; α, β) - \hat{β}(x; z_1, z_2)$')
fig.tight_layout()
```


    
![png](/assets/images/beta-distribution/beta-distribution-4.png)
    


The results isn't too bad -- true, the reconstructed curve oscillates a bit, but at a small scale. the oscillations can be resolved with more training data and more iterations, or by increasing the size of the neural networks. We can say that the encoder has managed to compress the input data to two parameters and the decoder to define how to build the PDF of the beta distribution from those.

And finally a nice example of what each latent variable represents by plotting, on a 10 by 10 grid, the result of the decoder for several values of $z_1$ and $z_2$, both ranging from -1 to 1, where can see the shape of the Beta distributions changing with the two parameters as we would expect. The dotted grey line indicates the zero axis; all plots share the same scale on the Y axis.


```python
n = 10
fig, axes = plt.subplots(nrows=n, ncols=n, figsize=(8, 8), sharex=True, sharey=True)
for i, c_1 in enumerate(np.linspace(-3, 3, n)):
    for j, c_2 in enumerate(np.linspace(-3, 3, n)):
        y = vae.decoder(torch.FloatTensor([c_1, c_2]).to(device)).detach().cpu().exp()
        axes[n - 1 - j, i].plot(grid, y)
        axes[n - 1 - j, i].plot(grid, np.zeros_like(grid), color='grey', linestyle='dashed')
        axes[n - 1 - j, i].axis('off')
        axes[n - 1 - j, i].set_title(f'{c_1:.2f},{c_2:.2f}', fontsize=6)
fig.tight_layout()
```


    
![png](/assets/images/beta-distribution/beta-distribution-5.png)
    


To conclude, two references that have largely inspired this contribution: https://avandekleut.github.io/vae/ for the code and https://mathybit.github.io/auto-var/ for the math.
