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

where $\kappa, \theta, \sigma > 0$ and $X(0) = X_0$. In finance, it was proposed by Cox, Ingersoll and Ross as an interest rate model and forms the stochastic volatility component of Heston's asset price model. The SDE is nonlinear and non-Lipschitzian. It was shown by Feller that a unique strong solution exists; this solution preserves the nonnegativity of the initial data, that is $X(t) \ge 0 \forall t$ with probability one.


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
& = X_0 + \theta \left( e^{\kappa t} - 1 \right)+ \int_0^t e^{\kappa s} \sigma \sqrt{X(s)} ds,
\end{aligned}
$$

giving

$$
X(t) = e^{-\kappa t}X_0 + \theta(1 - e^{-\kappa t}) + e^{-\kappa t}\int_0^t e^{\kappa s} \sigma \sqrt{X(s)} ds.
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

The long-term variance is therefore

$$
\lim_{t \rightarrow\infty}\mathbb{V}[X(t)] = \frac{\theta \sigma^2}{2 \kappa}.
$$
