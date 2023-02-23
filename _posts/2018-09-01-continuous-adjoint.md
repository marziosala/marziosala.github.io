---
layout: splash
permalink: /continuous-adjoint/
title: "Computing Sensitivities with the Continuous Adjoint Method"
header:
  overlay_image: /assets/images/continuous-adjoint/continuous-adjoint-splash.jpeg
excerpt: "How to compute option sensitivities when pricing with PDEs using adjoint methods."
---

In this article we look at the applications of the continuous adjoint method to finance. The approach is described in this [2015 Risk.net paper](https://www.risk.net/derivatives/2415103/greeks-with-continuous-adjoints-fast-to-code-fast-to-run).

The focus is on the pricing of [financial derivatives](https://en.wikipedia.org/wiki/Derivative_(finance)), for which one often needs to compute the  value as well as a (possibly quite large) number of risk sensitivities, also known as [Greeks](https://en.wikipedia.org/wiki/Greeks_(finance)), that is the partial derivative of the value with respect to a model parameter. For example, one needs to know how much the option value will change when spot changes, or when the volatility changes, as so on. 

We will go through the meain reasoning, the equations, and discuss how this method compares with other techniques for computing risk sensitivities.

We have already covered financial modeling in other articles, for example for the [Black-Scholes model](/black-scholes). The notation of this article is slightly difference, since we want to emphasize the dependency on the parameters on which we need to compute the derivatives. The stochastic differential equation reads

$$
dX(t, \alpha) = (r(t,\alpha) - q(t,\alpha) - \frac{1}{2}\sigma_{BS}(t,\alpha)^2)dt + \sigma_{BS}(t, \alpha) dW(t)
$$

where $r$ is the continuously componded interest rate, $q$ the continuous dividend, $\sigma_{BS}$ the volatility, $X(0) = x_0$ the initial conditions, $X(t) = \log(S(t))$ the logarithm of the spot price,  $\alpha \in \mathbb{R}^n$ a vector if parameters, and $\alpha^\star$ the market conditions around which we want to compute the sensitivities.

The PDE pricing problem reads as follows: Find $u(x, t, \alpha)$ such that

$$
\left\{
\begin{array}{r c l l}
\displaystyle 
- \frac{\partial u(x, t, \alpha)}{\partial t} - \mathcal{L}_{BS}(x, t, \alpha) u(x, t, \alpha) & = & 0 & \text{ in } \mathbb{R} \times (0, T) \times \mathbb{R}^n \\
u(x, T, \alpha) & = & g(x) & \text{ on } \mathbb{R} \times \mathbb{R}^n \\
\end{array}
\right.
$$

with the elliptic operator

$$
\textstyle
\mathcal{L}_{BS}(x, t, \alpha) = 
\frac{1}{2} \sigma_{BS}(t, \alpha)^2 \frac{\partial^2}{\partial x^2} + 
\left(r(t, \alpha) - q(t, \alpha) - \frac{1}{2}\sigma_{BS}(t, \alpha)^2 \right)
\frac{\partial}{\partial x}  - r_d(t, \alpha).
$$

Once $u(x, t, \alpha^\star)$ is computed, our goal is compute the risk sensitivities

$$
R_\ell(\alpha^\star) = \left.
\frac{\partial TV(\alpha)}{\partial \alpha_\ell}
\right|_{\alpha = \alpha^\star}, \quad \quad \ell = 1, \ldots, n
$$

as efficiently as possible.

The formulation of the paper is not limited to Black-Scholes; indeed, it can be applied to the general $d-$dimensional backward PDE

$$
\left\{
\begin{array}{r c l l}
\displaystyle 
- \frac{\partial u(x, t, \alpha)}{\partial t} - \mathcal{L}(x, t, \alpha) u(x, t, \alpha) & = & 0 & \text{ in } \mathbb{R}^d \times (0, T) \times \mathbb{R}^n \\
u(x, T, \alpha) & = & g(x) & \text{ on } \mathbb{R}^d \times \mathbb{R}^n \\
\end{array}
\right.
$$

with

$$
\textstyle
\mathcal{L}(x, t, \alpha) = 
\sum_{i,j=1}^d \sigma_{i, j}(x, t, \alpha)\frac{\partial^2}{\partial x_i \partial x_j}
+ \sum_{i=1}^d \mu_i(x, t, \alpha) \frac{\partial}{\partial x_i}
+ \beta(x, t, \alpha).
$$

in which case our objective becomes the following: for a given $\alpha^\star$, solve the PDE, then compute the option value

$$
TV(\alpha^\star) = u(x_0, 0, \alpha^\star) = \int_{\mathbb{R}^d} \delta(x - x_0)u(x, 0, \alpha^\star) dx
$$

with $\delta(x-x_0)$ the Dirac delta operator centered in $x_0$.

Borrowing from optimization methods, we see our problem as if we were computing

$$
R_\ell(\alpha^\star) = \left.
\frac{\partial TV(\alpha)}{\partial \alpha_\ell}
\right|_{\alpha = \alpha^\star}, \quad \quad \ell = 1, \ldots, n
$$

with

$$
TV(\alpha^\star) = \int_{\mathbb{R}^d} \delta(x - x_0)u(x, 0, \alpha) dx
$$

subjected to the PDE constraint

$$
\left\{
\begin{array}{r c l l}
\displaystyle 
- \frac{\partial u(x, t, \alpha)}{\partial t} - \mathcal{L}(x, t, \alpha) u(x, t, \alpha) & = & 0 & \text{ in } \mathbb{R}^d \times (0, T) \times \mathbb{R}^n \\
u(x, T, \alpha) & = & g(x) & \text{ on } \mathbb{R}^d \times \mathbb{R}^n \\
\end{array}
\right.
$$

The value of the derivative $TV(\alpha)$ is then replaced by

$$
TV(\alpha) - \int_0^T \int_{\mathbb{R}^n} \lambda(x, t, \alpha^\star)
\underbrace{
\left\{
-\frac{\partial u(x, t, \alpha)}{\partial t} -
\mathcal{L}(x, t, \alpha) u(x, t, \alpha)
\right\} 
}_{\text{identically zero } \forall \alpha}
\,dx\,dt
$$

where $\lambda(x, t, \alpha)$ is the Lagrangian multiplier (to be chosen); the risk sensitivities are instead

$$
\begin{aligned}
R_\ell(\alpha^\star) & =
\left.\frac{\partial TV(\alpha)}{\partial \alpha_\ell}\right|_{\alpha = \alpha^\star} + \\
&&  \frac{\partial}{\partial \alpha_\ell}
\left.
\int_0^T \int_{\mathbb{R}^n} \lambda(x, t, \alpha^\star)
\left\{
-\frac{\partial u}{\partial t} -
\mathcal{L}(x, t, \alpha) u(x, t, \alpha)
\right\} dx dt
\right|_{\alpha = \alpha^\star}
\end{aligned}
$$

Having reformulated the problem as a constraint optimization, the procedure is as follows:

1. we introduce a generic Lagrangian multiplier $\lambda(x, t \alpha)$;
2. we integrate by parts;
3. we define define a "good" $\lambda$ such that most terms disappears and we can compute the risk sensitivities $R_\ell$ efficiently.

To do that, we start from the definition of the price itself, written as an integral of the solution at $t=0$ multiplied by the [Dirac delta function](https://en.wikipedia.org/wiki/Dirac_delta_function) in $x_0$,

$$
TV(\alpha) = \int_{\mathbb{R}^d} \delta(x - x_0)u(x, 0, \alpha) dx.
$$

After several easy manipulations, we obtain

$$
R_\ell(\alpha^\star) = \int_0^T \int_{\mathbb{R}^d} \lambda(x, t, \alpha^\star)
\left.\frac{\partial \mathcal{L}(x, t, \alpha)}{\partial \alpha}\right|_{\alpha = \alpha^\star}
u(x, t, \alpha^\star) \,dx \,dt
$$

where $\lambda$ is the solution of the (forward) problem

$$
\left\{
\begin{array}{rcll}
\displaystyle
\frac{\partial \lambda(x, t, \alpha^\star)}{\partial t} -
\mathcal{L}'(x, t, \alpha^\star) \lambda(x, t, \alpha^\star) & = & 0 &
\text{ in } \mathbb{R}^d \times (0, T) \\
\lambda(x, t, \alpha^\star) & = & \delta(x - x_0) & \text{ on } \mathbb{R}^d
\end{array}
\right.
$$

where $\mathcal{L}'$ is the adjoint of $\mathcal{L}$,

$$
\mathcal{L}'(x, t, \alpha) \lambda = \sum_{i, j=1}^d \frac{\partial^2}{\partial x_i \partial x_j} (\sigma_{i,j}(x, t, \alpha)  \lambda) -
\sum_{i=1}^d \frac{\partial }{\partial x_i} (\mu_i(x, t, \alpha) \lambda(x, t)) + \beta(x, t, \alpha) \lambda,
$$

and

$$
\frac{\partial \mathcal{L}(x, t, \alpha)}{\partial \alpha_\ell} = \sum_{i, j=1}^d
\frac{\partial \sigma_{i,j}(x, t, \alpha)}{\partial \alpha_\ell}
\frac{\partial^2}{\partial x_i \partial x_j}
+ \sum_{i=1}^d \frac{\partial \mu_i(x, t, \alpha)}{\partial \alpha_\ell}\frac{\partial }{\partial x_i}
+ \frac{\partial \beta(x, t, \alpha)}{\partial \alpha_\ell}.
$$

This is the procedure for the continuous adjoint.

*Step 1*: Solve the backward pricing equation:

$$
\left\{
\begin{array}{r c l l}
\displaystyle 
- \frac{\partial u(x, t, \alpha^\star)}{\partial t} - \mathcal{L}(x, t, \alpha^\star) u(x, t, \alpha^\star) & = & 0 & \text{ in } \mathbb{R}^d \times (0, T) \\
u(x, T, \alpha^\star) & = & g(x) & \text{ on } \mathbb{R}^d \\
\end{array}
\right.
$$

to compute $u(x, t, \alpha^\star)$ and store all solutions.

*Step 2*: Solve the forward adjoint equation:
$$
\left\{
\begin{array}{r c l l}
\displaystyle 
\frac{\partial \lambda(x, t, \alpha^\star)}{\partial t} - \mathcal{L}'(x, t, \alpha^\star) u(x, t, \alpha^\star) & = & 0 & \text{ in } \mathbb{R}^d \times (0, T) \\
\lambda(x, T, \alpha^\star) & = & \delta(x - x_0) & \text{ on } \mathbb{R}^d\\
\end{array}
\right.
$$
to compute $\lambda(x, t, \alpha^\star)$ and store all solutions.

*Step 3*: For $\ell = 1, \ldots, n$ compute the integral

$$
R_\ell(\alpha^\star) = \int_0^T \int_{\mathbb{R}^d}
\lambda(x, t, \alpha^\star) \left.
\frac{\partial \mathcal{L}(x, t, \alpha)}{\partial \alpha_\ell}
\right|_{\alpha = \alpha^\star} u(x, t, \alpha^\star) \,dx \,dt.
$$

Note that $\lambda$ can be reinterpreted as a probability distribution.

We can apply integration by parts directly to

$$
\begin{aligned}
TV(\alpha^\star) & =
\int_{\mathbb{R}^d} \delta(x - x_0) u(x, 0, \alpha^\star) dx \\
& - \int_0^T \int_{\mathbb{R}^d} \lambda(x, t, \alpha^\star)
\left\{
-\frac{\partial u}{\partial t} - \mathcal{L}(x, t, \alpha^\star) u(x, t, \alpha^\star)
\right\} \, dx dt
\end{aligned}
$$

Because of our choice of $\lambda$, it easily follows that

$$
TV(\alpha^\star) = \int_{\mathbb{R}^d}
\underbrace{\lambda(x, T, \alpha^\star)}_{\text{prob distribution }}
\underbrace{g(x)}_{\text{ payoff}} dx.
$$

Besides,

$$
TV(\alpha^\star) = \int_{\mathbb{R}^d} \lambda(x, t, \alpha^\star) u(x, t, \alpha^\star) dx ,
\quad \quad \forall t \in [0, T].
$$

We could also compute $\lambda$ directly and use it to price. The adjoint of $\lambda$ will then be $u$; the integrals for $R_\ell(\alpha^\star)$ will be similar.

Differentiating the pricing equation

$$
\left\{
\begin{array}{r c l}
\displaystyle 
- \frac{\partial u(x, t, \alpha)}{\partial t} - \mathcal{L}(x, t, \alpha) u(x, t, \alpha) & = & 0 \\
u(x, T, \alpha) & = & g(x)
\end{array}
\right.
$$

wrt $\alpha_\ell$ gives

$$
\left\{
\begin{array}{r c l}
\displaystyle 
- \frac{\partial \hat R_\ell(x, t, \alpha)}{\partial t} - \mathcal{L}(x, t, \alpha)\hat R_\ell(x, t, \alpha) & = & \displaystyle \frac{\partial \mathcal{L}(x, t, \alpha)}{\partial \alpha_\ell}
u(x, t, \alpha)\\
\hat R_\ell(x, T, \alpha) & = & 0.
\end{array}
\right.
$$

Our risk is then

$$
R_\ell(\alpha^\star) = \hat R_\ell(x_0, 0, \alpha^\star).
$$

By using the (discounted) probability distribution $\lambda(x, t, \alpha^\star)$ and the Feynman-Kac formula, we obtain the same result as before.

The general formulation takes into account a generic functional $J$ defined as

$$
J(\alpha) = \int_\Omega P(x, \alpha, u(x, 0, \alpha) dx 
+ \int_0^T \int_\Omega Q(x, t, \alpha, u(x, t, \alpha)) \, dx\, dt,
$$

on a bounded domain $\Omega \subseteq \mathbb{R}^d$ and
$\alpha-$dependency in the final conditions $g$.  

After solving a (slightly modified) adjoint problem, we get:

$$
\begin{aligned}
\left.\frac{\partial J(\alpha)}{\partial \alpha_\ell}\right|_{\alpha=\alpha^\star} & =
\int_\Omega \left. \frac{\partial P}{\partial \alpha_\ell}(x, \alpha, u(x, 0, \alpha)) \right|_{\alpha = \alpha^\star}dx +
\\
& \int_0^T \int_\Omega \left.\frac{\partial Q}{\partial \alpha_\ell}(x, t, \alpha, u(x, t, \alpha)) \right|_{\alpha = \alpha^\star} dx \, dt +
\\
& \int_\Omega \left. \lambda(x, T, \alpha^\star) \frac{\partial g(x, \alpha)}{\partial \alpha_\ell} \right|_{\alpha = \alpha^\star} dx +
\\
& \int_0^T \int_\Omega \left.\lambda(x, t, \alpha^\star)
\frac{\partial \mathcal{L}(x, t, \alpha)}{\partial \alpha_\ell} u(x, t, \alpha)
\right|_{\alpha = \alpha^\star} dx dt
\end{aligned}
$$

A first example is this. We consider $P(u) = u^2, Q = 0$, resulting in the functional

$$
J(\alpha) = \int_0^\pi u(x, 0, \alpha)^2 dx
$$

where $u$ is the solution of the final boundary value problem

$$
\left\{
\begin{array}{rcl}
- \displaystyle \frac{\partial u(x, t, \alpha)}{\partial t} - \alpha_1
\frac{\partial^2 u(x, t, \alpha)}{\partial x^2} & = & 0 \\
x(0, t, \alpha) & = & 0 \\
x(\pi, t, \alpha) & = & 0 \\
u(x, T, \alpha) & = & \frac{1}{2} \sin(x)e^{-\alpha_1 T} \alpha_2 \\
\end{array}
\right.
$$

for $x \in (0, \pi), t \in (0, T)$, whose the exact solution is

$$
u(x, t, \alpha) = \frac{1}{2} \sin(x)e^{-\alpha_1 t} \alpha_2.
$$

Then

$$
J(\alpha) = \frac{\pi \alpha_2^2}{8}, \quad \quad
R_1 = 0, \quad \quad R_2 = \frac{\pi \alpha_2}{4}.
$$

Finally, the adjoint solution for the Black-Scholes PDE we started this article with is

<!--
Using sympy:

sigma, r, mu = symbols('sigma, r, mu')
lam = 1 / (sqrt(2 * pi * t)* sigma) * exp(r*t - (x - (mu - 1/2*sigma**2)*t)**2 / (2 * sigma**2 * t))
(lam.diff(t) + (mu - 1/2*sigma**2)*lam.diff(x) - 1/2*sigma**2*lam.diff(x, x) - r*lam).simplify()

-->

$$
\lambda(x, t) = \frac{1}{\sigma \sqrt{2 \pi t}} \exp\left\{
r_d t - \frac{(x - (r - q - \frac{1}{2}\sigma^2) t)^2}{2 \sigma^2 t}
\right\}.
$$
