---
layout: splash
permalink: /beta-distribution/
title: "Variational Autoencoders"
header:
  overlay_image: /assets/images/beta-distribution/beta-distribution-splash.png
excerpt: "Variational autoencoders applied to mathematical functions."
---

In the *autoencoder* post we have seen how to approximate the PDF of the Beta distribution. As we noticed, in general the encoding space is a non-convex manifold and the codes have arbitrary scales. This makes basic autoencoders a poor choice for generative models. *Variational autoencoders* fix this issue by ensuring that the coding space follows a desirable distribution from which we can easily sample from. This distribution typically is the standard normal distribution. So, if we have $N_C$ encodings, our goal is to have $N_C$ independent and normally distributed encodings. In order to do that, we want to learn the mean $\mu$ and the standard deviation $\sigma$ that are close to that of a standard normal distribution, while at the same time having outputs that are close to the inputs.

A useful trick to achieve what we want is to write the $i-$th component of the encoding vector
$Z=(z_1, z_2, \ldots, z_C) $ as
$$
z_i = \mu_i  + \sigma_i \cdot \xi,
$$
where $\xi \sim N(0,1)$ is a random number coming from the standard normal distribution. Why do we do this? Because the random number generation is a non-differentiable operation, so if we did specify the mean and standard deviation of the distribution directly we would not be able to backpropagate the gradients.

The architecture is reported in the picture below. Note that the dense layer is connected to two separate blocks, one of which generates $\mu$ and the other $\sigma$. The part is the *encoder*. Once $Z$ has been generated, we enter the *decoder* that rebuilds the inputs from the encodings. 

![](/assets/images/beta-distribution/variational-autoencoders-net.png)

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
p(x) & = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( \frac{(x - \mu)^2}{2 \sigma^2} \right) \\
q(x) & = \frac{1}{\sqrt{2 \pi}} \exp \left( \frac{x^2}{2} \right).
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




    <matplotlib.legend.Legend at 0x25ad7d15488>




    
![png](/assets/images/beta-distribution/beta-distribution-1.png)
    



```python
num_hidden = 256
```


```python
normal_dist = torch.distributions.Normal(0, 1)
if device == 'cuda':
    normal_dist.loc = normal_dist.loc.cuda() # hack to get sampling on the GPU
    normal_dist.scale = normal_dist.scale.cuda()
```


```python
class VariationalEncoder(nn.Module):

    def __init__(self, latent_dims):
        super().__init__()
        self.linear = nn.Linear(num_points, num_hidden)
        self.linear_mu = nn.Linear(num_hidden, latent_dims)
        self.linear_logsigma2 = nn.Linear(num_hidden, latent_dims)
        self.kl = 0

    def forward(self, x):
        x = F.relu(self.linear(x))
        mu =  self.linear_mu(x)
        logsigma2 = self.linear_logsigma2(x)
        sigma = torch.exp(logsigma2 / 2)
        z = mu + sigma * normal_dist.sample(mu.shape)
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
        return F.relu(self.linear2(z))  # can use F.relu() as this is a CDF
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
    
    def compute_log_likelihood(self, X, X_hat, η):
        ξ = normal_dist.sample(X_hat.shape)
        X_sampled = X_hat + η * ξ
        return 0.5 * ((X - X_sampled)**2).sum() / η**2 + η.log()
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
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % print_every == 0:
            print(f"Epoch: {epoch + 1:3d}, lr=: {last_lr[0]:.4e}, " \
                  f"total log loss: {total_log_loss:.4f}, total KL loss: {total_kl_loss:.4f}")
        history.append((total_log_loss, total_kl_loss))
    return history
```


```python
%%time
latent_dims = 2
vae = VariationalAutoencoder(latent_dims).to(device)
history = train(vae, X, epochs=100, lr=1e-4, gamma=0.975, η=0.1, print_every=10)
```

    Epoch:  10, lr=: 7.9624e-05, total log loss: 177319683.0049, total KL loss: 30413.3774
    Epoch:  20, lr=: 6.1814e-05, total log loss: 177252307.2591, total KL loss: 29225.2336
    Epoch:  30, lr=: 4.7988e-05, total log loss: 177263478.4586, total KL loss: 28762.0559
    Epoch:  40, lr=: 3.7255e-05, total log loss: 177262979.6462, total KL loss: 28350.7128
    Epoch:  50, lr=: 2.8922e-05, total log loss: 177279913.8852, total KL loss: 28218.2624
    Epoch:  60, lr=: 2.2453e-05, total log loss: 177261550.7046, total KL loss: 28144.7394
    Epoch:  70, lr=: 1.7431e-05, total log loss: 177286632.5582, total KL loss: 28067.6457
    Epoch:  80, lr=: 1.3532e-05, total log loss: 177275772.3955, total KL loss: 28106.7608
    Epoch:  90, lr=: 1.0505e-05, total log loss: 177301151.0656, total KL loss: 27862.5597
    Epoch: 100, lr=: 8.1556e-06, total log loss: 177284926.8205, total KL loss: 27868.6222
    Wall time: 25min 33s
    


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


    
![png](/assets/images/beta-distribution/beta-distribution-2.png)
    


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

    Params for the test: α=3.1240, β=4.7134
    


```python
res = minimize(func, [0.0, 0.0], method='Nelder-Mead')
res
```




     final_simplex: (array([[-2.19452863,  1.8495743 ],
           [-2.19450143,  1.8495532 ],
           [-2.19444038,  1.84950134]]), array([26.89578629, 26.89578629, 26.89578629]))
               fun: 26.89578628540039
           message: 'Optimization terminated successfully.'
              nfev: 87
               nit: 36
            status: 0
           success: True
                 x: array([-2.19452863,  1.8495743 ])




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


    
![png](/assets/images/beta-distribution/beta-distribution-3.png)
    


The results isn't too bad -- true, the reconstructed curve oscillates a bit, but at a small scale. We can say that the encoder has managed to compress the input data to two parameters and the decoder to define how to build the PDF of the beta distribution from those.

To conclude, two references that have largely inspired this contribution: https://avandekleut.github.io/vae/ for the code and https://mathybit.github.io/auto-var/ for the math.
