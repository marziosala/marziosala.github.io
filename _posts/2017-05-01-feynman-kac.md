---
layout: splash
permalink: /feynman-kac/
title: "The Feynman-Kac Equation"
header:
  overlay_image: /assets/images/feynman-kac/feynman-kac.png
excerpt: "Solving the Feynman-Kac equation using finite differences"
---

In this long article we explore the connection between stochastic differential equations (SDEs) and partial differential equations (PDEs). Our starting point is an SDE of type

$$
dX(t) = \mu(X(t), t) dt + \sigma(X(t), t) dW(t)
$$

for $t \in (t_0, T)$ and $X(t_0) = X_0$. We want to compute

$$
\mathbb{E}\left[ \Phi(X(T)) \right]
$$

for some function $\Phi(x)$. To solve numerically with Monte Carlo we define a time grid $0 = t_0 < t_1 < \ldots < t_m = T$, construct approximated paths

$$
\hat X^{(\ell)}_i \approx X^{(\ell)}(t_i), \ell = 1, \ldots, N_{paths},
$$

and use the unbiased estimator

$$
\mathbb{E}[\Phi(X(T))] \approx \frac{1}{N_{paths}} \sum_{\ell=1}^{N_{paths}} \Phi(\hat X_m^{(\ell)}).
$$

Another way for computing 

$$
\mathbb{E}[\Phi(X(T))]
$$

is to use partial differential equations. Let

$$
u(x, t) = \mathbb{E}_{X(t)=x}[\Phi(X(T))].
$$

Then, $u(x, t)$ solves

$$
\begin{cases}
\displaystyle
\frac{\partial{u(x, t)}}{\partial{t}}
    + \mu(x, t) \frac{\partial{u(x, t)}}{\partial{x}}
    + \frac{1}{2} \sigma(x, t)^2 \frac{\partial^2{u(x, t)}}{\partial{x^2}} = 0 
    & \text{ in } \mathbb{R} \times (0, T) \\
%
    u(x, T) = \Phi(x) & \text{ on } \mathbb{R}.
\end{cases}
$$

This is known as the [Feynman-Kac formula](https://en.wikipedia.org/wiki/Feynman%E2%80%93Kac_formula), which we will explore here.

To see this, we apply Itô's lemma to $u(X(t), t)$:

$$
d[u(X(t), t)] = \left(
\frac{\partial u}{\partial t}
    + \mu(x, t) \frac{\partial{u}}{\partial{x}}
    + \frac{1}{2} \sigma(x, t)^2 \frac{\partial^2{u}}{\partial{x^2}} 
\right) + \sigma(x, t) \frac{\partial u}{\partial x}dW(t),
$$

and integrate over time,

$$
\begin{align}
\displaystyle
u(X(T), T) - u(X(t), t) = & \int_t^T
\left(
\frac{\partial u}{\partial t}
    + \mu(x, t') \frac{\partial{u}}{\partial{x}}
    + \frac{1}{2} \sigma(x, t')^2 \frac{\partial^2{u}}{\partial{x^2}} 
\right) dt'\\
&     + \int_t^T
\sigma(x, t') \frac{\partial u}{\partial x}dW(t').
\end{align}
$$

Taking expectations and using the final conditions yields $\mathbb{E}_{X(t)=x}[\Phi(X(T))] - u(x, t) = 0$, which is what we want to compute.

In general we are interested in some discounted payoff,

$$
u(x, t) = \mathbb{E}_{X(t)=x}
\left
    [e^{-\int_t^T r(X(s), s) ds }\Phi(X(T))
\right]
$$

for some known function $r(x, t)$, or if $r$ is not stochastic (and constant),

$$
u(x, t) = e^{-r(T - t)} \mathbb{E}_{X(t)=x}
\left
    [\Phi(X(T))
\right].
$$

Then $u(x, t)$ solves

$$
\begin{cases}
\displaystyle
\frac{\partial{u(x, t)}}{\partial{t}}
    + \mu(x, t) \frac{\partial{u(x, t)}}{\partial{x}}
    + \frac{1}{2} \sigma(x, t)^2 \frac{\partial^2{u(x, t)}}{\partial{x^2}} -r(x, t) u = 0 
    & \text{ in } \mathbb{R} \times (0, T) \\
%
    u(x, T) = \Phi(x) & \text{ on } \mathbb{R}.
\end{cases}
$$

An easy extension is for

$$
u(x, t) = \mathbb{E}_{X(t) = x}\left[
    \int_t^T \Psi(X(s), s) ds
\right]
$$

for a specified function $\Psi$. Then $u(x, t)$ solves

$$
\begin{cases}
\displaystyle
\frac{\partial{u(x, t)}}{\partial{t}}
    + \mu(x, t) \frac{\partial{u(x, t)}}{\partial{x}}
    + \frac{1}{2} \sigma(x, t)^2 \frac{\partial^2{u(x, t)}}{\partial{x^2}} + \Psi(x, t)= 0 
    & \text{ in } \mathbb{R} \times (0, T) \\
%
    u(x, T) = 0 & \text{ on } \mathbb{R}.
\end{cases}
$$

This can be shown using as before Ito's lemma:

$$
\begin{align}
d[u(x, t)] & = 
\underbrace{
    \left(
        \frac{\partial{u}}{\partial{t}}
    + \mu(x, t) \frac{\partial{u}}{\partial{x}}
    + \frac{1}{2} \sigma(x, t)^2 \frac{\partial^2{u}}{\partial{x^2}}
    \right)
}_{=-\Psi(X(t), t)} + \sigma(X(t), t) \frac{\partial u}{\partial x} dW(t) \\
& = -\Psi(X(t), t) dt + \sigma(X(t), t) \frac{\partial u}{\partial x} dW(t).
\end{align}
$$

Integrating and taking expectations gives

$$
\underbrace{u(X(T), T)}_{=0} - u(x, t) + \mathbb{E}_{X(t) = t}\left[
    \int_t^T \Psi(X(s), s) ds
\right] = 0.
$$
