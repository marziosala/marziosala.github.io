---
layout: splash
permalink: /bachelier/
title: "The Bachelier Model"
header:
  overlay_image: /assets/images/bachelier/bachelier-splash.jpeg
excerpt: "Computing the price of an option using the Bachelier model."
---

In this article we look at the [Bachelier model](https://en.wikipedia.org/wiki/Bachelier_model).

We assume the dynamics

$$
dS(t) = \mu S(t) dt + \sigma dW(t),
$$

with $S(0) = S_0$ given and $W(t)$ a standard Wiener process. The drift can be removed with
the transformation

$$
X(t) = e^{\mu (T - t)} S(t),
$$

which brings

$$
dX(t) = -\mu e^{\mu(T - t)}S(t) dt + e^{\mu(T - t)}dS(t),
$$

that is

$$
\begin{aligned}
e^{-\mu (T-t)}dX(t) + \mu S(t) dt & = dS(t) \\
 & = \mu S(t)dt + \sigma dW(t),
\end{aligned}
$$

and therefore

$$
dX(t) = e^{\mu(T - t)} \sigma dW(t).
$$

This equation can be easily integrated, giving

$$
X(T) - X(0) = \int_0^Te^{\mu(T - \tau)}\sigma dW(\tau),
$$

that is,

$$
S_T = e^{\mu T}S_0 + \int_0^Te^{\mu (T - \tau) \sigma dW(\tau)}.
$$

Using [Ito's isometry](https://en.wikipedia.org/wiki/It%C3%B4_isometry), it follows that

$$
S_T \sim \mathcal{N}\left(
  e^{\mu T} S_0,
  \frac{\sigma^2}{2\mu} \left( e^{2\mu T} - 1 \right)
\right),
$$

meaning that the spot distribution at time $T$ follows a normal distribution with mean
$e^{\mu T} S_0$ and variance $\frac{\sigma^2}{2\mu} \left( e^{2\mu T} - 1 \right)$.

At this point we have all the ingredients to compute the price of a call option:

$$
\begin{aligned}
C & = e^{-rT} \mathbb{E}[(S_T - K)^+] \\
& = e^{-rT} \mathbb{E}\left[(e^{\mu T} S_0 +
  \sqrt{\frac{\sigma^2}{2\mu} \left( e^{2\mu T} - 1 \right)} Z - K)^+\right] \\
& = \sqrt{\frac{\sigma^2}{2\mu} \left( e^{2\mu T} - 1 \right)} \mathbb{E}
\left[
  \left(
    Z - \frac{K - e^{\mu T}S_0}{\sqrt{\frac{\sigma^2}{2\mu} \left( e^{2\mu T} - 1 \right)}}
  \right)
\right],
\end{aligned}
$$

where $Z \sim \mathcal{N}(0, 1)$ is a standard normal. Our task reduces to the computation,
for $a \in \mathbb{R}$, of

$$
\begin{aligned}
\mathbb{E}[(Z - a)^+] & = \mathbb{E}[(Z - a) \mathbb{1}_{Z > a}] \\
& =  \mathbb{E}[Z\mathbb{1}_{Z > a}] - a \mathbb{P}[Z > a] \\
& = \frac{1}{2 \pi} \int_a^\infty z e^{-z^2/2} dz - a (1 - \Phi(a)),
\end{aligned}
$$

where $\Phi$ is the cumulative density function of the standard normal distribution. Using $\zeta = z^2/2$, we obtain

$$
\begin{aligned}
\mathbb{E}[(Z - a)^+] & = \frac{1}{2 \pi} \int_{a^2 /2}^\infty e^{-\zeta} d\zeta - a \Phi(-a) \\
& = - \frac{1}{2 \pi} \left. e^{-\zeta} \right|_{a^2/2}^\infty - a \Phi(-a) \\
& = - \frac{1}{2 \pi} e^{-a^2/2} - a \Phi(-a) \\
& = \varphi(a) - a \Phi(-a) \\
& = \varphi(-a) - a \Phi(-a),
\end{aligned}
$$

where $\varphi$ is the probability density function of the normal distribution. The call price follows by replacing $a$ with its value above.
