---
layout: splash
permalink: /swiss-roll/
title: "An Introduction to Diffusion Models"
header:
  overlay_image: /assets/images/swiss-roll/swiss-roll-splash.jpeg
excerpt: "Understanding how diffusion models work, on a simple two-dimensional dataset"
---

In this post we look at [diffusion models](https://en.wikipedia.org/wiki/Diffusion_model), a quite successful class of [generative models](https://en.wikipedia.org/wiki/Generative_model). A generative model is a model that, given samples from an unknown distribution $p^\star$, is capable of learning a good approximation of it. Once learned, the approximated distribution can be used to generate new samples, or to evaluate the likelihood of observed or sampled data.

In general $p^\star$ is too hard to find directly, so we seek a good approximation of it by defining a sufficiently large parametric family $\{ p_\vartheta \}_{\vartheta \in \Theta}$, then solve for

$$
\vartheta^\star = \argmin_{\vartheta \in \Theta} \mathcal{L}(p_\vartheta, p^\star)
$$

for some loss function $\mathcal{L}$.

We define $p_\vartheta$ not by using an explicit parametrized distribution family, as they are all too simple for our task, but we rather consider implicitly parametrized probability distributions. That is, we assume some latent variables $z$ and write

$$
\begin{aligned}
p_\vartheta(x) & = \int p_\vartheta(x, z) dz \\
& = \int p_\vartheta(x | z) p_\vartheta(z) dz.
\end{aligned}
$$

The distribution over the latent variables $p_\vartheta(z)$ is generally "simple", normal or uniform; the complexity of the transformation is contained in $p_\vartheta(z | z)$, which is generally based on some deep neural networks functions depending on $z$ for the definition of its coefficients.

With this approach it is easy to sample from $p_\vartheta$: first we sample $z \sim p_\vartheta(z)$, we compute the coefficients as a function of $z$ and finally sample $x \sim p_\vartheta(x | z)$. As such, we have defined a generative model.

We still need to define a procedure for computing the optimal parameters $\vartheta^\star$. The approach we follow is to maximize the log-likelihood of our data,

$$
\vartheta^\star = \argmax_{\vartheta \in \Theta} \log p_\vartheta(x),
$$

where $\log p_\vartheta(x)$ is called the *evidence*. Intuitively, if we have chosen the right $p_\vartheta$ and $\vartheta^\star$, we would expect a high probability of "seeing" our data, and therefore the likelihood will be a "large" number. Given another distribution $q(z)$, we have

$$
\begin{aligned}
\log p_\vartheta(x) & = \log \int p_\vartheta(x | z) p_\vartheta(z) dz \\
%
& = \log \int p_\vartheta(x | z) p_\vartheta(z) \frac{q(z)}{q(z)} dz \\
%
& \ge \int \log \left[
\frac{p_\vartheta(z)}{q(z)} p_\vartheta(x | z)
\right] q(z) dz \\
%
& = \mathbb{E}_q\left[ \log p_\vartheta(x | z) \right]
+ \mathbb{E}_q\left[ \log \frac{p_\vartheta(z)}{q(z)} \right] \\
%
& = \mathbb{E}_q\left[ \log p_\vartheta(x | z) \right] - D_{KL}(q(z) || p_\vartheta(z)).
\end{aligned}
$$

This is called the *evidence lower bound*, or ELBO. The first term describes the probability of the dat $x$ given the latent variables $z$, a quantity we want to maximize by picking those models $q(z)$ that better predict the data. The second term is the negative [Kullback-Leibler](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) divergence between $q(z)$ and $p_\vartheta(z)$, and we want to minimize this quantity by choosing $q(z)$ and $p_\vartheta(z)$ to be similar.

Diffusion models were first presented in [Sohl-Dickstein et al., 2015](https://arxiv.org/abs/1503.03585). The key idea of the paper is to use a Markov chain to gradually convert one distribution into another, with each step in the chain analytically tractable -- as such, the full chain can also be evaluated. That is, instead of looking for a single transformation that can be difficult to learn and evaluate, we use a composition of several small perturbations. The approach is extended in [Ho et al, 2020](https://arxiv.org/pdf/2006.11239.pdf), which we follow closely for the notation, and further extended in [Song et al., 2021](https://arxiv.org/pdf/2011.13456.pdf).

The basic idea is to start from a given sample, $x_0$, and using a *forward process* we define $x_1, x_2, \ldots, x_T$ for some $T > 0$. With a small abuse of notation, $x_0$ is our actual data while $x_1, \ldots, x_T$ are latent variables into which the data in transformed. At each step $t$ a bit or noise is added, until when, at $t=T$, the data becomes indistinguishable from pure Gaussian noise. A second process, the *reverse process*, will then generate data starting from pure Gaussian noise.

Given a data point from our distribution, $x_0 = x \sim p^\star(x)$, we define a forward diffusion process in which we add a small amount of Gaussian noise to the sample in $T$ steps, producing a sequence of noisy samples $x_1, \ldots, x_T$. Given a variance schedule $\{ \beta_t \in (0, 1) \}_{t=1}^T$, we have

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt(1 - \beta_t) x_{t-1}, \beta_t I)
$$

and

$$
q(x_1, \cdots, x_T | x_0) = \Pi_{t=1}^T q(x_t | x_{t-1}).
$$

The values of $T$ and $\beta_t$ ust be defined such that the distribution of $x_T$ is close to normal.

If we could sample from $q(x_{t - 1} | x_t)$, we could reverse the process and generate samples from Gaussian noise. Unfortunately we cannot easily estimate it but we can set up a model that learns such transformation. We define such reverse process as

$$
p_\vartheta(x_0, \cdots, x_T) = p(x_T) \Pi_{t=1}^T p_\vartheta(x_{t-1} | x_t)
$$

with

$$
p_\vartheta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1} | \mu_\vartheta(x_t, t), \Sigma_\vartheta(x_t, t))
$$

and

$$
p_x(_T) = \mathcal{N}(x_T; 0, I).
$$

To define the loss function for a given point $x_0$, we use the ELBO, which can be derived as follows:

$$
\begin{aligned}
-\log p_\vartheta(x_0) & = - \log \int p_\vartheta(x_0, x_1, x_2, \cdots, x_T) dx_1 dx_2 \cdots dx_T \\
%
& = - \log \int \frac{p_\vartheta(x_0, x_1, \cdots, x_T) q(x_1, \cdots, x_T | x_0)}{q(x_1, \cdots, x_T | x_0)} dx_{1:T} \\
%
& = -\log \mathbb{E}_{q(x_1, \cdots, x_T)} \left[
    \frac{p_\vartheta(x_0, x_1, \cdots, x_T)}{q(x_1, \cdots, x_T) | x_0)}
\right] \\
%
& = -\mathbb{E}_{q(x_1, \cdots, x_T)}\left[
- \log \frac{p_\vartheta(x_0, x_1, \cdots, x_T)}{q(x_1, \cdots, x_T | x_0)}
\right] \\
%
& = -\mathbb{E}_{q(x_1, \cdots, x_T)}\left[
\log \frac{
p(x_T) \Pi_{t=1}^T p_\vartheta(x_{t-1} | x_t)
}{
    \Pi_{t=1}^T q(x_t | x_{t-1})
}
\right].
\end{aligned}
$$

This is the most basic formulation of the method, which we will use in the code below.


```python
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import torch
import itertools
from tqdm.auto import tqdm
np.random.seed(42)
_ = torch.manual_seed(43)
```

We use the two-dimensional version of the [swiss roll](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_swiss_roll.html) dataset. This is often use as a test in nonlinear dimensionality reduction; here we are interested in the distribution of its points, which we want to be able to reproduce using diffusion methods. Each entry $x$ in our dataset is composed by two values, the X-axis component and the Y-axis component. We use 10'000 such points; the scatter plot explains the name of the dataset. The coordinates are scaled by a factor 10 to be more unitary in magnitude. 


```python
from sklearn.datasets import make_swiss_roll
num_samples = 10_000
dataset, _ = make_swiss_roll(num_samples, noise=0.2)
dataset = dataset[:, (0, 2)] / 10
plt.scatter(dataset[:, 0], dataset[:, 1], alpha=0.5, color='red', edgecolor='firebrick', s=5)
dataset = torch.Tensor(dataset).float()
```


    
![png](/assets/images/swiss-roll/swiss-roll-1.png)
    


We use $T=50$ and a constant $\beta_t=0.05^2$.


```python
T = 50
β_all = 0.05**2 * np.ones(T)
```

The forward process is used to move from the input $x_0 = x$ to the noisy version $x_T$. We store the distributions as well as the realizations to compute the log probabilities in the loss function.


```python
def compute_forward_process(x_0, T=T, β_all=β_all):
    q_all, x_all = [None], [x_0]
    x_t = x_0
    for t in range(T):
        β_t = β_all[t]
        q_t = torch.distributions.Normal(np.sqrt(1 - β_t) * x_t, np.sqrt(β_t))
        x_t = q_t.sample()
        q_all.append(q_t)
        x_all.append(x_t)
    return q_all, x_all
```

The function `plot_evolution()` allows us to see ten steps of the transformation from $x_0$ to $x_T$. We can see that our swiss roll becomes more or more Gaussian.


```python
def plot_evolution(x_all, is_reverse=False):
    fig, axes = plt.subplots(1, 11, figsize=(33, 3), sharex=True, sharey=True)
    for i in range(11):
        t = i * 5
        x_t = x_all[t]
        axes[i].scatter(x_t[:, 0], x_t[:, 1], color='grey', edgecolor='blue', s=5);
        axes[i].set_xticks([-1, 1])
        axes[i].set_yticks([-1, 1])
        label = T - t if is_reverse else t
        if label == T: label = 'T'
        axes[i].set_title(f't={label}')
        axes[i].axis('square')
    fig.tight_layout()
```


```python
_, x_all = compute_forward_process(dataset[:1_000])
plot_evolution(x_all)
```


    
![png](/assets/images/swiss-roll/swiss-roll-2.png)
    


The reverse process requires the definition of two models, one for $\mu_\vartheta(x_t, t)$ and the other for $\Sigma_\vartheta(x_t, t)$ for the standard deviation. The input size is three -- the two spatial dimensions plus time, while the output has two dimensions. We keep the same structure for both mean and standard deviation, with the only addition of a softplus function to the standard deviation one.


```python
μ_model = torch.nn.Sequential(
    torch.nn.Linear(2 + 1, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 2)
)

σ_model = torch.nn.Sequential(
    torch.nn.Linear(2 + 1, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 2),
    torch.nn.Softplus()
)
```

The computation of the reverse process is similar to the one of the forward process, but it uses the two models for mean and standard deviation at each step.


```python
def compute_reverse_process(μ_model, σ_model, count, T=T):
    p = torch.distributions.Normal(torch.zeros(count, 2), torch.ones(count, 2))
    x_t = p.sample()
    sample_history = [x_t]
    for t in range(T, 0, -1):
        xin = torch.cat((x_t, (t / T) * torch.ones(x_t.shape[0], 1)), dim=1)
        p = torch.distributions.Normal(μ_model(xin), σ_model(xin))
        x_t = p.sample()
        sample_history.append(x_t)
    return sample_history
```

As customary, we define a tiny class that wraps our dataset.


```python
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, X):
        super().__init__()
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index]
```

It is a good practice to initialize the weights of the model -- here we use Xavier initialization


```python
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

for model in [μ_model, σ_model]:
    model.apply(init_weights)
```

The `compute_loss()` function contains all the logic of the method. This implementation is not very efficient but it reflects the formulae we have seen above and it is a nice starting point to understand those methods. The inputs are all the steps of the forward process as well as all the corresponding distributions, plus the models for $\mu_\vartheta$ and $\Sigma_\vartheta$. Using those inputs we can easily compute the various terms we need: $\log p(x_T)$, $\log p_\vartheta(x_{t-1} | x_t)$ and $\log q(x_{t} | x_{t-1})$. The integrals are approximated by the average over all the samples.


```python
def compute_loss(q_all, x_all, μ_model, σ_model):
    p = torch.distributions.Normal(
        torch.zeros(x_all[0].shape),
        torch.ones(x_all[0].shape)
    )

    loss = -p.log_prob(x_all[-1]).mean()

    for t in range(1, T):
        x_t = x_all[t]
        x_t1 = x_all[t - 1]
        q_t = q_all[t]

        x_input = torch.cat((x_t, (t / T) * torch.ones(x_t.shape[0], 1)), dim=1)
        p_t = torch.distributions.Normal(μ_model(x_input), σ_model(x_input))

        loss -= torch.mean(p_t.log_prob(x_t1))
        loss += torch.mean(q_t.log_prob(x_t))

    return loss / T
```


```python
params = itertools.chain(μ_model.parameters(), σ_model.parameters())
optim = torch.optim.Adam(params, lr=1e-3)
```


```python
data_loader = trainloader = torch.utils.data.DataLoader(MyDataset(dataset), batch_size=512, shuffle=True)

loss_history = []
bar = tqdm(range(1, 1000 + 1))
for e in bar:

    for x_0 in data_loader:
        forward_distributions, forward_samples = compute_forward_process(x_0)

        optim.zero_grad()
        loss = compute_loss(forward_distributions, forward_samples, μ_model, σ_model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1)
        optim.step()
        bar.set_description(f'Loss: {loss.item():.4f}')
        loss_history.append(loss.item())

    if e in [1, 10, 20, 50] or e % 100 == 0:
        with torch.no_grad():
            print(f'iter: {e}, loss: {loss.item():.6e}')
            x_all = torch.stack(compute_reverse_process(μ_model, σ_model, 200))
            plot_evolution(x_all, True)
            plt.show()
```


      0%|          | 0/1000 [00:00<?, ?it/s]


    iter: 1, loss: 8.050011e-01
    


    
![png](/assets/images/swiss-roll/swiss-roll-3.png)
    


    iter: 10, loss: 2.388238e-02
    


    
![png](/assets/images/swiss-roll/swiss-roll-4.png)
    


    iter: 20, loss: 2.569752e-02
    


    
![png](/assets/images/swiss-roll/swiss-roll-5.png)
    


    iter: 50, loss: 2.346585e-02
    


    
![png](/assets/images/swiss-roll/swiss-roll-6.png)
    


    iter: 100, loss: 1.600004e-02
    


    
![png](/assets/images/swiss-roll/swiss-roll-7.png)
    


    iter: 200, loss: 1.569699e-02
    


    
![png](/assets/images/swiss-roll/swiss-roll-8.png)
    


    iter: 300, loss: 1.637200e-02
    


    
![png](/assets/images/swiss-roll/swiss-roll-9.png)
    


    iter: 400, loss: 5.513587e-03
    


    
![png](/assets/images/swiss-roll/swiss-roll-10.png)
    


    iter: 500, loss: 7.712739e-03
    


    
![png](/assets/images/swiss-roll/swiss-roll-11.png)
    


    iter: 600, loss: 8.991304e-03
    


    
![png](/assets/images/swiss-roll/swiss-roll-12.png)
    


    iter: 700, loss: 5.626636e-03
    


    
![png](/assets/images/swiss-roll/swiss-roll-13.png)
    


    iter: 800, loss: 9.135569e-03
    


    
![png](/assets/images/swiss-roll/swiss-roll-14.png)
    


    iter: 900, loss: 4.601617e-03
    


    
![png](/assets/images/swiss-roll/swiss-roll-15.png)
    


    iter: 1000, loss: 3.721235e-03
    


    
![png](/assets/images/swiss-roll/swiss-roll-16.png)
    



```python
plt.semilogy(loss_history)
plt.xlabel('Batch Iteration')
plt.ylabel('Log-Loss');
```


    
![png](/assets/images/swiss-roll/swiss-roll-17.png)
    


Results aren't bad, even if they are not too good neither, with the generated dataset being a bit more diffused than the original one. In my tests different learning rates, number of steps $T$ or number of points in the neural networks didn't change the results substantially.


```python
fig, (ax0, ax1) = plt.subplots(figsize=(12, 6), ncols=2)
n = 1_000
ax0.scatter(dataset[:n, 0], dataset[:n, 1], alpha=0.5, color='blue', edgecolor='grey', s=5)
ax0.set_title('Original dataset')
x_all = torch.stack(compute_reverse_process(μ_model, σ_model, n))
ax1.scatter(x_all[-1][:, 0], x_all[-1][:, 1], alpha=0.5, color='crimson', edgecolor='grey', s=5)
ax1.set_title('Generated dataset');
```


    
![png](/assets/images/swiss-roll/swiss-roll-18.png)
    


We conclude this post with a note for two good blogs in the subject: [Lil'Lol](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) post on the topic is one of the best introductions that can be found on the web, while [Emilio Dorigatti](https://e-dorigatti.github.io/math/deep%20learning/2023/06/25/diffusion.html) post inspired the code. Very noteworthy is also [Calvin Luo](https://arxiv.org/pdf/2208.11970.pdf) paper on diffusion models.
