---
layout: splash
permalink: /ornstein-uhlenbeck/
title: "Ornstein-Uhlenbeck Process"
header:
  overlay_image: /assets/images/ornstein-uhlenbeck/ornstein-uhlenbeck-splash.jpeg
excerpt: "A quick overview of the square-root mean-reverting stochastic differential equation"
---

In this post we consider the [Ornstein-Uhlenbeck](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) process, described by the SDE

$$
\begin{aligned}
dX(t) & = \kappa (\theta - X(t)) dt + \sigma dW(t) \\
%
X(0) & = X_0
\end{aligned}
$$

with $\kappa, \theta, \sigma > 0$ three parameters. $\kappa$ defines the mean reversion coefficient, $\theta$ is the long-term mean and $\sigma$ is the volatility coefficient.

The above SDE can be solved exactly after rearranging the terms and multiplying by $e^{\kappa t}$:

$$
\begin{aligned}
dX(t) + \kappa X(t) dt & = \kappa \theta dt + \sigma dW(t) \\
%
d(e^{\kappa t} X(t)) & = \kappa \theta e^{\kappa t} dt + \sigma e^{\kappa t} dW(t) \\
%
e^{\kappa t}X(t) - X_0 & = \theta \int_0^t e^{\kappa s} \kappa ds + 
    \sigma \int_0^t e^{\kappa s} dW(s) \\
%
e^{\kappa t}X(t) - X_0 & = \theta(e^{\kappa t} - 1) + \sigma \int_0^t e^{\kappa s} dW(s) \\
%
X(t) & = \theta + e^{-\kappa t} (X_0 - \theta) + \sigma \int_0^t e^{\kappa (s - t)} dW(s).
\end{aligned}
$$

It is easy to see that the mean is

$$
\mathbb{E}[X(t)] = \theta + (X_0 - \theta) e^{-\kappa t}.
$$

The covariance can be computed as follows, for $s < t$:

$$
\begin{aligned}
\mathbb{C}[X(s), X(t)] & = \mathbb{C}\left[
    \sigma e^{-\kappa s} \int_0^s e^{\kappa s'} dW(s'), 
    \sigma e^{-\kappa t} \int_0^t e^{\kappa t'} dW(t') 
\right] \\
%
& = \sigma^2 e^{-\kappa(s + t)} \mathbb{E} \left[
    \int_0^s e^{\kappa s'} dW(s') \,\,
    \int_0^t e^{\kappa t'} dW(t')
\right] \\
%
& = \sigma^2 e^{-\kappa(s + t)} \mathbb{E} \left[
    \int_0^s e^{\kappa s'} dW(s')
    \left( \int_0^s e^{\kappa s'} dW(s') + \int_s^t e^{\kappa t'} dW(t') \right)
\right] \\
%
& = \sigma^2 e^{-\kappa(s + t)} \mathbb{E} \left[
    \int_0^s e^{\kappa s'} dW(s')
    \int_0^s e^{\kappa s'} dW(s')
\right] \\
% 
& = \sigma^2 e^{-\kappa(s + t)} \int_0^s e^{2 \kappa s'} ds' \\
%
& = \sigma^2 e^{-\kappa(s + t)} \frac{1}{2 \kappa}(e^{2\kappa s} - 1).
\end{aligned}
$$

The variance is therefore

$$
\mathbb{V}[X(t)] = \frac{\sigma^2}{2 \kappa}(1 - e^{-2\kappa t}).
$$
