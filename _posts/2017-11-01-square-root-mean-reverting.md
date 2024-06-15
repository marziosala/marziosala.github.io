---
layout: splash
permalink: /square-root-mean-reverting/
title: "Square-Root Mean-Reverting Process"
header:
  overlay_image: /assets/images/square-root-mean-reverting/square-root-mean-reverting-splash.jpeg
excerpt: "A quick overview of the square-root mean-reverting stochastic differential equation"
---

In this article we study the stochastic differential equation

$$
dX(t) = \kappa (\theta - X(t))dt + \sigma \sqrt{X(t)} dW(t),
$$

where $\kappa, \theta, \sigma > 0$ and $X(0) = X_0$. In finance, it was proposed by [Cox, Ingersoll and Ross](https://en.wikipedia.org/wiki/Cox%E2%80%93Ingersoll%E2%80%93Ross_model) as an interest rate model and forms the stochastic volatility component of Heston's asset price model. The SDE is nonlinear and non-Lipschitzian. It was shown by Feller that a unique strong solution exists; this solution preserves the nonnegativity of the initial data, that is $X(t) \ge 0 \forall t$ with probability one.


To find the solution, we rearrange the terms and multiply by $e^{\kappa t}$,

$$
\begin{aligned}
dX(t) - \kappa X(t) & = \kappa \theta dt + \sigma \sqrt{X(t)} dW(t) \\
d(e^{\kappa t} X(t)) & = e^{\kappa t} \kappa \theta dt + e^{\kappa t} \sigma \sqrt{X(t)} dW(t),
\end{aligned}
$$

which integrated between 0 and $t$ gives

$$
\begin{aligned}
e^{\kappa t} X(t) & = X_0 + \int_0^t e^{\kappa s} \kappa \theta ds + \int_0^t e^{\kappa s} \sigma \sqrt{X(s)} ds \\
& = X_0 + \theta \left( e^{\kappa t} - 1 \right)+ \int_0^t e^{\kappa s} \sigma \sqrt{X(s)} dW(s),
\end{aligned}
$$

giving

$$
X(t) = e^{-\kappa t}X_0 + \theta(1 - e^{-\kappa t}) + e^{-\kappa t}\int_0^t e^{\kappa s} \sigma \sqrt{X(s)} dW(s).
$$

Taking expectations, we obtain

$$
\mathbb{E}[X(t)] = e^{-\kappa t} X_0 + \theta(1 - e^{-\kappa t}).
$$

Since

$$
\lim_{t\rightarrow\infty} \mathbb{E}[X(t)] = \theta,
$$

the value $\theta$ represents the long-term mean value, with the process moving (in expectations) from $X_0$ at $t \approx 0$ to $\theta$ as $t \rightarrow\infty$.

To compute the variance, we need $\mathbb{E}[X(t)^2]$. From easy computations we have

$$
\begin{aligned}
\mathbb{V}[X(t)] & = 2 \left( e^{-\kappa t} X_0 + \theta(1 - e^{-\kappa t}) \right)
\mathbb{E}\left[
\int_0^t e^{\kappa s} \sigma\sqrt{X(s)} dW(s)
\right] +
\sigma^2 e^{-2 \kappa t} \mathbb{E}\left[ \left(
\int_0^t e^{\kappa s} \sqrt{X(s)} dW(s) \right)^2
\right] \\
%
& = \sigma^2 e^{-2 \kappa t} \mathbb{E}\left[
    \int_0^t e^{2\kappa s} X(s) ds
\right] \\
%
& = \sigma^2 e^{-2 \kappa t} \int_0^t e^{2\kappa s} \mathbb{E}\left[ X(s) \right] ds \\
%
& = \sigma^2 e^{-2 \kappa t} \int_0^t e^{2\kappa s} \left( e^{-\kappa s} X_0 + \theta(1 - e^{-\kappa s}) \right) ds \\
%
& = \sigma^2 e^{-2 \kappa t} \int_0^t \left( e^{\kappa s} X_0 + e^{2\kappa s} \theta - e^{\kappa s} \theta \right) ds \\
%
& = \frac{\sigma^2}{\kappa}X_0 \left( e^{-\kappa t} - e^{-2\kappa t} \right)
   + \frac{\theta \sigma^2}{2 \kappa} \left( 1 - e^{-\kappa t} \right)^2.
\end{aligned}
$$

(See also this [stackexchange equation](https://math.stackexchange.com/questions/944181/) since
the diffusion is non-Lipschitz.)
The long-term variance is therefore

$$
\lim_{t \rightarrow\infty}\mathbb{V}[X(t)] = \frac{\theta \sigma^2}{2 \kappa}.
$$

It is possible to compute the probability density $p(x, t)$ associated with our process by solving the [Fokker-Planck equation](https://en.wikipedia.org/wiki/Fokker%E2%80%93Planck_equation)

$$
\begin{aligned}
\frac{\partial}{\partial t}p(x, t) & =
-\frac{\partial}{\partial x}(\kappa(\theta - x)p(x, t))
+ \frac{1}{2} \frac{\partial^2}{\partial x^2}(\sigma^2 p(x, t)) \\
%
p(x, 0) & = \delta(x - X_0)
\end{aligned}
$$

As it is well-known, the distribution of $X(t)$ is a scaled non-central $\chi^2$ distribution,

$$
X(t) \sim \frac{Y}{c}
$$

with

$$
c = \frac{2 \kappa}{(1 - e^{-\kappa t}) \sigma^2}
$$

and the probability distribution function of $Y$ given by

$$
f_Y(y, t) = \frac{1}{2}
e^{-\frac{\lambda + y}{2}}
\left( \frac{y}{\lambda} \right)^{\frac{k - 2}{4}}
I_{\frac{k - 2}{2}}(\sqrt{\lambda y})
$$

with $k = \frac{4 \kappa \theta}{\sigma^2}$ degrees of freedom and non-centrality parameter $\lambda = 2 c X_0 e^{-\kappa t}$, and $I$ a modified Bessel function of the first kind.

To undestand why we end up with a $\chi^2$ distribution, consider the $d$ processes

$$
dX_i(t) = -\frac{1}{2} \alpha X(t) dt + \sqrt{\alpha} dW_i(t),
$$

each of which is normal,

$$
X_i(t) \sim \mathcal{N} \left(0, 1 - e^{-\alpha t} \right).
$$

The sum of the squares

$$
X(t) = \sum_{i=1}^d \left( X_i(t) \right)^2
$$

satisfies the SDE

$$
\begin{aligned}
dX(t) & = \sum_{i=1}^d 2X_i(t) dX_i(t) + d \alpha dt \\
%
& = \sum_{i=1}^d X_i(t) \left( -\alpha X_i(t) + 2 \sqrt{\alpha} dW_i(t) \right) + d \alpha dt \\
%
& = \alpha(d - X(t)) dt + \sqrt{4 \alpha} \sum_{i=1}^d X_i(t) dW_i(t) \\
%
& = \alpha(d - X(t)) dt + \sqrt{4 \alpha} \sqrt{X(t)} dW(t),
\end{aligned}
$$

which is our process. 

If the Feller condition $\kappa \theta \ge \frac{1}{2}\sigma^2$ is not satisfied, the process will visit the origin recurrently but not stay there -- that is, the zero boundary is strongly reflecting. This was first shown by William Feller in a [1950 paper](https://www.jstor.org/stable/1969318), giving the name to the condition.

We discretize using the Euler-Maruyama method. Given a set of time points

$$
0 = t_0 < t_1 < t_2 < \ldots < t_n = T
$$

assumed for simplicity to be linearly spaced, that is $t_{i+1} - t_i = \Delta t = \frac{T}{n}$, we use

$$
X_{i+1} = X_i + \kappa (\theta - X_i) \Delta t + \sigma \sqrt{X_i} \sqrt{\Delta t} Z, 
$$

with $Z \sim \mathcal{N}(0, 1)$ and, with a slight abuse of notation, $X_i$ the numerical approximation of $X(t_i)$. Clearly, because of the square root, we need $X_i \geq 0 \forall i$, but with the above discretization we could obtain negative $X_i$. In fact, since $Z$ is normal, we have

$$
\mathbb{P}[_{i+1} < 0] = \Phi\left(
-\frac{X_i + \kappa(\theta - X_i)\Delta t}{\sigma \sqrt{X_i}\sqrt{\Delta t}}
\right),
$$

assuming $X_i \ge 0$. The problem is generally resolve by either replacing $\sqrt{X_i}$ by
$\sqrt{|X_i|}$ or by $\sqrt{\max{X_i, 0}}$. The first approach is called reflection, meaning that the process is reflected back into the positivity, while the latter absorption, as the boundary takes in the process at $x=0$. By doing that we have only solved the problems with the square root can get still negative values; alternatively we either impose

$$
X_{i+1} \leftarrow |X_{i+1}|
$$

or

$$
X_{i+1} \leftarrow \max{0, X_{i+1}}
$$

as a post-step process.
