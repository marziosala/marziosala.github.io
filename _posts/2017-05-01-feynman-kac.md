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

is to use partial differential equations (PDEs). Let

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

This is known as the **Feynman-Kac equation**.
