---
layout: splash
permalink: /functional/
title: "Exploring Markov-Functional Methods"
header:
  overlay_image: /assets/images/functional/functional.jpeg
excerpt: "A short introduction to new a class of SDEs."
---

In this article we explore the Markovian functional approach for single-underlying models. We start with a local volatility model, then build a Blach-Scholes type model that:

- is a one-factor Markov model, with a Brownian driver and a simple structure; and
- reproduces exactly the ame one-dimensional distributions of the local volatility model, that is all European options are priced exactly as in the local volatility model.

The starting point is the local volatility model. We assume the risk-neutral dynamics

$$
\frac{dS(t)}{S(t)} = \mu dt + \sigma(S(t), t) dW(t),
$$

with $S(0) = S_0$ and, with standard notation, $W(t)$ is a Wiener process. It is well-known that, at a given maturity $T$, the probability distribution is fully determined by the European call prices,

$$
f_{S}(s, T) = \left.\frac{\partial^2 C(K, T)}{\partial K^2}\right|_{K=s}.
$$

We assume we are interested at a set of times $\mathcal{T} = \{ T_1, T_2, \ldots, T_n \}$. On each $T_i, i = 1, \ldots, n$,
we can compute the cumulative density function $F_{S}(s, T_i)$ from $f_{S}(s, T_i)$. These $n$ cumulative distribution functions
are the only components we need to keep from the local volatility model; everything else can be discarded.

To proceed, we need to define the (simpler) process that will drive our model. For simplicity here we use the simple process

$$
dX(t) = \nu(t) dW(t),
$$

with $X(0) = 0$ and a deterministic volatility $\nu(t) > 0$. The distribution of $X(t)$ is Gaussian,

$$
X(t) \sim \mathcal{N}(0, v(t)^2),
$$

with

$$
v(t) = \int_0^t \nu(\tau)^2 d\tau.
$$

The process $X(t)$ is Markov by construction; the idea is to define a functional mapping $\hat{S}(t) = f(X(t), t)$ such that $S(T)$ and $\hat{S}(T)$ have the same distributions for all $T \in \mathcal{T}$. For all times not included in $\mathcal{T}$, instead, the two processes will have different distributions. This means that the matching is only at *discrete* points, not for a continuous interval.

The mapping is defined as follows. For any $T_i \in mathcal{T}$, we impose the equality of the cumulative density functions,

$$
F_S(s, T_i) = F_X(x, T).
$$

Because

At this point, for each maturity $T_i \in \mathcal{T}$, we define a monotone mapping $g$ through the cumulative density function matching,

$$
F_S(s, T_i) = F_X(x, T_i),
$$

from which we can define a function $g$ such that $s = g(T_i, x)$. Both $F_S$ and $F_X$ are monotonically increases and ranging from 0 to 1, so for every $x$ we can find one and only point $s$ such that the equation above is satisfied. That is, a function $g$ exists such that

$$
s = g(x, T_i).
$$

Because of our choice of $X(t)$, we have $F_X(x, T_i) = \Phi(x / v(T_i))$ is known analytically; also $F_S(s, T_i)$ is known, perhaps numerically. Therefore

$$
g(x, T_i) = \left( F_S \right)^{-1}\Phi \left( \frac{x}{v(T_i)} \right).
$$
