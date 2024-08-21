---
layout: splash
permalink: /feynman-kac/
title: "The Feynman-Kac Equation"
header:
  overlay_image: /assets/images/feynman-kac/feynman-kac.jpeg
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

for some function $\Phi(x)$. To solve numerically with Monte Carlo we define a time grid $0 = t_0 < t_1 < \ldots < t_m = T$, construct approximated paths  $\hat X^{(\ell)}_i \approx X^{(\ell)}(t_i)$, $\ell = 1, \ldots, N_{paths}$, and use the unbiased estimator

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


```python

```


```python

```


```python
from abc import ABC, abstractmethod
import matplotlib.pylab as plt
from matplotlib import cm
import numpy as np
import numpy.typing as npt
from numpy.polynomial.polynomial import polyfit, polyval
from scipy.linalg import solve_banded
```

 All the quantities for such problem are contained in the `SDE` class. 


```python
class SDE(ABC):

    t_0: float

    T: float
    
    X_0: float

    def compute_μ(self, x, t):
        pass

    def compute_σ(self, x, t):
        pass
```

The corresponding backward PDE, instead, requires a few classes. We start from the space and time domain, with time going from $t_0$ to $T$ and the domain from a value $x_{min}$ to an $x_{max}$.


```python
class Domain:

    t_0: float

    T: float

    x_min: float

    x_max: float

    def __init__(self, t_0, T, x_min, x_max):
        self.t_0 = t_0
        self.T = T
        self.x_min = x_min
        self.x_max = x_max
```

The partial 


```python
class InitialValueProblem(ABC):
    
    domain: Domain
    
    @abstractmethod
    def compute_a(self, x, t):
        "coefficient of the second-order derivative"
        pass
    
    @abstractmethod
    def compute_b(self, x, t):
        "coefficient of the first-order derivative"
        pass
    
    @abstractmethod
    def compute_c(self, x, t):
        "coefficient of the source term"
        pass

    @abstractmethod
    def compute_f(self, x, t):
        "force term"
        pass

    @abstractmethod
    def compute_g(self, x, t):
        "boundary conditions"
        pass

    @abstractmethod
    def compute_u_0(self, x):
        "initial condition"
        pass

    @abstractmethod
    def compute_u_exact(self, x, t):
        "the exact solution"
        pass
```


```python
class ConstantCoeffsInitialValueProblem(InitialValueProblem):

    def __init__(self, domain, a, b, c, exact_solution, f):
        self.domain = domain
        self.a = a
        self.b = b
        self.c = c
        self.exact_solution = exact_solution
        self.f = f

    def compute_a(self, x, t):
        return np.ones_like(x) * self.a
    
    def compute_b(self, x, t):
        return np.ones_like(x) * self.b
    
    def compute_c(self, x, t):
        return np.ones_like(x) * self.c

    def compute_f(self, x, t):
        return self.f(x, t)

    def compute_g(self, x, t):
        return self.exact_solution(x, t)

    def compute_u_0(self, x):
        return self.exact_solution(x, self.domain.t_0)

    def compute_u_exact(self, x, t):
        return self.exact_solution(x, t)
```


```python
class SpaceDiscretizationMethod(ABC):
    
    problem: InitialValueProblem
    
    x_all: npt.NDArray
    
    @abstractmethod
    def assemble(self, t):
        pass

    @abstractmethod
    def impose_bc(self, t):
        pass
```


```python
params = dict(
    num_points=128,
    num_steps=64,
    time_integrator='Crank-Nicolson',
)
```


```python
class SecondOrderFiniteDifferenceMethod(SpaceDiscretizationMethod):

    def __init__(self, problem: InitialValueProblem, params: dict):
        self.x_all = np.linspace(problem.domain.x_min, problem.domain.x_max, params['num_points'])
        self.problem = problem

    def assemble(self, t):
        num_points = len(self.x_all)
        A = np.zeros(num_points * 3)

        a = self.problem.compute_a(self.x_all, t)
        b = self.problem.compute_b(self.x_all, t)
        c = self.problem.compute_c(self.x_all, t)
        f = self.problem.compute_f(self.x_all, t)

        for i in range(1, num_points - 1):
            x_prev, x_i, x_next = self.x_all[i - 1], self.x_all[i], self.x_all[i + 1]
            h_prev, h_i = x_i - x_prev, x_next - x_i

            den = h_prev * h_i * (h_prev + h_i)

            D2_prev = 2 * h_i / den
            D2_i = -2 * (h_prev + h_i) / den
            D2_next = 2 * h_prev / den 
            D1_prev = -1.0 / (h_prev + h_i)
            D1_next = 1.0 / (h_prev + h_i)
            
            A[i -1 + 2 * num_points] = a[i] * D2_prev + b[i] * D1_prev
            A[i + num_points] = a[i] * D2_i + c[i]
            A[i + 1] = a[i] * D2_next + b[i] * D1_next
        
        return A, f

    def impose_bc(self, t, A, f):
        num_points = len(self.x_all)

        A[1], A[num_points] = 0.0, 1.0
        A[-num_points - 1], A[-2] = 1.0, 0.0
        
        f[0] = self.problem.compute_g(self.x_all[0], t)
        f[-1] = self.problem.compute_g(self.x_all[-1], t)
```


```python
class ThetaMethod:

    def __init__(self, space_disc: SpaceDiscretizationMethod, params: dict):
        self.space_disc = space_disc
        self.t_all = np.linspace(space_disc.problem.domain.t_0, space_disc.problem.domain.T, params['num_steps'])
        if params['time_integrator'] == 'Crank-Nicolson':
            self.ϑ_all = np.ones_like(self.t_all) * 0.5
        elif params['time_integrator'] == 'Backward Euler':
            self.ϑ_all = np.ones_like(self.t_all)
        else:
            raise ValueError("time_integrator not recognized")

    @staticmethod
    def matmult(A, b):
        n = len(b)
        retval = A[n:2 * n] * b
        retval[:-1] += A[1:n] * b[1:]
        retval[1:] += A[2 * n:-1] * b[:-1] 
        return retval
    
    def solve(self):
        n = len(self.space_disc.x_all)
        I = np.concatenate((np.zeros(n), np.ones(n), np.zeros(n)))

        u = self.space_disc.problem.compute_u_0(self.space_disc.x_all)
        solutions = [u]

        for k in range(len(self.t_all) - 1):
            t_k = self.t_all[k]
            t_k_plus_1 = self.t_all[k + 1]
            Δt = t_k_plus_1 - t_k

            ϑ = self.ϑ_all[k]

            A_k, f_k = self.space_disc.assemble(t_k)
            A_k_plus_1, f_k_plus_1 = self.space_disc.assemble(t_k_plus_1)

            Z = I - ϑ * Δt * A_k_plus_1
            b = self.matmult(I + (1 - ϑ) * Δt * A_k, u) + ϑ * Δt * f_k_plus_1 + (1 - ϑ) * Δt * f_k

            self.space_disc.impose_bc(t_k_plus_1, Z, b)

            u = solve_banded((1, 1), Z.reshape(3, -1), b)
            solutions.append(u)
        self.solutions = np.array(solutions)

    def get_solutions(self):
        return self.solutions

    def compute_max_error(self):
        max_error = 0.0
        u_T = self.solutions[-1]
        T = self.space_disc.problem.domain.T
        u_exact_T = self.space_disc.problem.compute_u_exact(self.space_disc.x_all, T)
        max_error = max(max_error, max(abs(u_T - u_exact_T)))
        return max_error
```


```python
ω = 4
α, β = 1, 10
a, b, c = 1.0, -2.0, 0.5
exact_solution = lambda x, t: np.sin(α * x + β * t)
der_t = lambda x, t: β * np.cos(α * x + β * t)
der_x = lambda x, t: α * np.cos(α * x + β * t)
der_xx = lambda x, t: -α**2 * np.sin(α * x + β * t)
f = lambda x, t: der_t(x, t) - a * der_xx(x, t) - b * der_x(x, t) - c * exact_solution(x, t)
domain = Domain(t_0=0, T=1.0, x_min=-np.pi, x_max=np.pi)
test = ConstantCoeffsInitialValueProblem(domain=domain, a=a, b=b, c=c, exact_solution=exact_solution, f=f)
```


```python
hs = []
max_errors = []
base = 32
for η in [1, 2, 4, 8]:
    hs.append(1 / η)
    params = dict(num_points=16 * η, num_steps=16 * η, time_integrator='Crank-Nicolson')
    space_disc = SecondOrderFiniteDifferenceMethod(test, params)
    t_all = np.linspace(test.domain.t_0, test.domain.T, base * η)
    theta_method = ThetaMethod(space_disc, params)
    theta_method.solve()
    max_errors.append(theta_method.compute_max_error())
p = polyfit(np.log(hs), np.log(max_errors), 1)[1]
max_errors, p
```




    ([0.03692492661553992,
      0.008744121217710177,
      0.002130038980338589,
      0.0005247700322816851],
     2.044773129350085)




```python

```


```python

```


```python
ω = 4
# exact_solution = lambda x, t: np.sin(ω * x) * np.exp(-t)
# f = lambda x, t: np.zeros_like(x)
α, β = 1, 10
a, b, c = 1.0, -2.0, 0.5
exact_solution = lambda x, t: np.sin(α * x + β * t)
der_t = lambda x, t: β * np.cos(α * x + β * t)
der_x = lambda x, t: α * np.cos(α * x + β * t)
der_xx = lambda x, t: -α**2 * np.sin(α * x + β * t)
f = lambda x, t: der_t(x, t) - a * der_xx(x, t) - b * der_x(x, t) - c * exact_solution(x, t)
domain = Domain(t_0=0, T=1.0, x_min=-np.pi, x_max=np.pi)
test = ConstantCoeffsInitialValueProblem(domain=domain, a=a, b=b, c=c, exact_solution=exact_solution, f=f)
```


```python
hs = []
max_errors = []
base = 32
for η in [1, 2, 4, 8]:
    hs.append(1 / η)
    x_all = np.linspace(test.domain().x_min, test.domain().x_max, base * η)
    t_all = np.linspace(test.domain().t_0, test.domain().T, base * η)
    solutions = solve(test, x_all, t_all, 0.5)
    max_errors.append(compute_max_error(test, x_all, solutions))
p = polyfit(np.log(hs), np.log(max_errors), 1)[1]
max_errors, p
```




    ([0.008744121217710177,
      0.002130038980338589,
      0.0005247700322816851,
      0.00013019225590193972],
     2.0229916169518876)




```python
XX, YY = plt.meshgrid(x_all, t_all)
fig, (ax0, ax1) = plt.subplots(figsize=(10, 5), ncols=2,subplot_kw=dict(projection='3d'))
ax0.plot_surface(XX, YY, solutions)
ax1.plot_surface(XX, YY, np.array([test.u_exact(x_all, t) for t in t_all]))
```




    <mpl_toolkits.mplot3d.art3d.Poly3DCollection at 0x2035307bf10>




    
![png](/assets/images/feynman-kac/feynman-kac-1.png)
    



```python
k = -1
plt.plot(x_all, solutions[k], label='numerical solution')
plt.plot(x_all, solutions[k] - test.u_exact(x_all, t_all[k]), label='error')
plt.legend();
```


    
![png](/assets/images/feynman-kac/feynman-kac-2.png)
    



```python
class FinalValueProblem(InitialValueProblem):

    def __init__(self, sde: SDE, u_0, x_min, u_min, x_max, u_max, f):
        self.sde = sde
        self.t_0 = sde.t_0()
        self.T = sde.T()
        self.domain_ = Domain(self.t_0, self.T, x_min, x_max)
        self.u_0_ = u_0
        self.u_min = u_min
        self.u_max = u_max
        self.f_ = f
    
    def to_τ(self, t):
        return self.T - t
    
    def to_t(self, τ):
        return self.T - τ
    
    def domain(self):
        return self.domain_

    def a(self, x, t):
        τ = self.to_τ(t)
        return 0.5 * self.sde.σ(x, τ)**2
    
    def b(self, x, t):
        τ = self.to_τ(t)
        return (self.sde.μ(x, τ) - 0.5 * self.sde.σ(x, τ)**2)
    
    def c(self, x, t):
        return np.zeros_like(x)

    def f(self, x, t):
        return self.f_(x, t)

    def g(self, x, t):
        if x == self.domain_.x_min():
            return self.u_min
        elif x == self.domain_.x_max():
            return self.u_max
        else:
            raise Exception("9oundany condition not recognized")

    def u_0(self, x):
        return self.u_0_(x)

    def u_exact(self, x, t):
        raise Exception("not implemented yet")
```


```python
class WienerProcess(SDE):

    def t_0(self) -> float:
        return 0.0

    def T(self) -> float:
        return 1.0
    
    def X_0(self) -> float:
        return 0.0

    def μ(self, x, t) -> float:
        return np.zeros_like(x)

    def σ(self, x, t) -> float:
        return np.ones_like(x)
```


```python
u_0 = lambda x: np.ones_like(x)
f = lambda x, t: np.zeros_like(x)
problem = FinalValueProblem(WienerProcess(), u_0, -1.0, 0.0, 1.0, 0.0, f)
x_all = np.linspace(problem.domain().x_min(), problem.domain().x_max(), 32)
t_all = np.linspace(problem.domain().t_0(), problem.domain().T(), 32)
solutions_no_touch = solve(problem, x_all, t_all, 1.0)
```


```python
u_0 = lambda x: np.zeros_like(x)
problem = FinalValueProblem(WienerProcess(), u_0, -1.0, 1.0, 1.0, 1.0, f)
x_all = np.linspace(problem.domain().x_min(), problem.domain().x_max(), 32)
t_all = np.linspace(problem.domain().t_0(), problem.domain().T(), 32)
solutions_touch = solve(problem, x_all, t_all, 1.0)
```


```python
XX, YY = plt.meshgrid(x_all, t_all)
fig, (ax0, ax1) = plt.subplots(figsize=(10, 5), ncols=2, subplot_kw=dict(projection='3d'),
                               sharex=True, sharey=True)
ax0.plot_surface(XX, YY, solutions_no_touch[::-1], cmap=cm.coolwarm, alpha=0.9)
ax0.scatter([0.0], [0.0], [np.interp(0.0, x_all, solutions_no_touch[-1])], color='red', s=50)
ax0.set_title('No touch probability')
ax0.scatter(np.zeros(31), np.zeros(31), np.linspace(0., 1, 31), s=10, color='grey') 
ax1.plot_surface(XX, YY, solutions_touch[::-1], cmap=cm.coolwarm, alpha=0.9)
ax1.scatter([0.0], [0.0], [np.interp(0.0, x_all, solutions_touch[-1])], color='red', s=50)
ax1.scatter(np.zeros(31), np.zeros(31), np.linspace(0., 1, 31), s=10, color='grey') 
ax1.set_title('Touch probability')
for ax in (ax0, ax1):
    ax.set_xlabel('x')
    ax.set_ylabel('t')
```


    
![png](/assets/images/feynman-kac/feynman-kac-3.png)
    



```python
u_0 = lambda x: np.zeros_like(x)
f = lambda x, t: np.where(np.logical_and(x >= -1, x <= 1), 1.0, 0.0)
sde = WienerProcess()
problem = FinalValueProblem(sde, u_0, -5.0, 0.0, 5.0, 0.0, f)
x_all = np.linspace(problem.domain().x_min(), problem.domain().x_max(), 512)
t_all = np.linspace(problem.domain().t_0(), problem.domain().T(), 128)
solutions_band = solve(problem, x_all, t_all, 1.0)
```


```python
xxx = []
for i, t in enumerate(t_all):
    xxx.append(np.interp(t, x_all, solutions_band[i]))
```


```python
XX, YY = plt.meshgrid(x_all, t_all)
fig, (ax0, ax1) = plt.subplots(figsize=(10, 5), ncols=2)
c = ax0.contourf(YY, XX, solutions_band[::-1], cmap=cm.coolwarm, alpha=0.9)
fig.colorbar(c, ax=ax0, location='left')
ax1.plot(t_all, list(reversed(xxx)))
ax0.set_xlabel('t')
ax0.set_ylabel('x')
ax1.set_xlabel('t')
ax0.set_title('Feynman-Kac Solution')
ax1.set_title('Expected Time in [-1 1] for a point (0, t)');
```


    
![png](/assets/images/feynman-kac/feynman-kac-4.png)
    



```python
class MonteCarlo:
    
    def __init__(self, sde, problem, num_paths, num_steps):
        self.t_all = np.linspace(0, problem.T, num_steps)
        X = sde.X_0() * np.ones(num_paths)
        X_all = [X]
        for i in range(num_steps - 1):
            t = self.t_all[i]
            Δt = self.t_all[i + 1] - t
            μ = sde.μ(X, t)
            σ = sde.σ(X, t)
            W = np.random.randn(num_paths)
            X = X + μ * Δt + σ * W * np.sqrt(Δt)
            X_all.append(X)
        self.X_all = np.array(X_all)
```


```python
mc = MonteCarlo(sde, problem, 1_000, 50)
```


```python
num_inside = 0
for i in range(1_000):
    path = mc.X_all[:, i]
    num_inside += sum(np.logical_and(path >= -1, path <= 1)) / 50
num_inside / 1_000
```




    0.8439




```python

```


```python
u_0 = lambda x: np.where(x <= 0, 1.0, 0.0)
f = lambda x, t: np.zeros_like(x)
sde = WienerProcess()
problem = FinalValueProblem(sde, u_0, -5.0, 1.0, 5.0, 0.0, f)
x_all = np.linspace(problem.domain().x_min(), problem.domain().x_max(), 512)
t_all = np.linspace(problem.domain().t_0(), problem.domain().T(), 128)
solutions = solve(problem, x_all, t_all, 1.0)
```


```python
mc = MonteCarlo(sde, problem, 100, 128)
```


```python
XX, YY = plt.meshgrid(x_all, t_all)
fig, ax0 = plt.subplots(figsize=(10, 6), ncols=1, subplot_kw=dict(projection='3d'))
ax0.plot_surface(XX, YY, solutions[::-1], cmap=cm.coolwarm, alpha=0.5)

idx = 43
X = mc.X_all[:, idx]
Y = mc.t_all
Z = []
path = mc.X_all[:, idx]
for i, solution in zip(path, solutions[::-1]):
    Z.append(np.interp(i, x_all, solution))

ax0.plot(X, Y, Z, 'black', linewidth=4, antialiased=True)
ax0.set_xlabel('X')
ax0.set_ylabel('t')
ax0.set_zlabel('conditional expectation')
```




    Text(0.5, 0, 'conditional expectation')




    
![png](/assets/images/feynman-kac/feynman-kac-5.png)
    



```python
min([min(mc.X_all[:, i]) for i in range(100)])
```




    -2.811876227832887




```python
plt.plot(solutions[0])
```




    [<matplotlib.lines.Line2D at 0x2036029c520>]




    
![png](/assets/images/feynman-kac/feynman-kac-6.png)
    



```python
mc.X_all[:, 43].shape
```




    (128,)




```python
plt.plot(x_all, solutions[-1, :])
```




    [<matplotlib.lines.Line2D at 0x2035c7052b0>]




    
![png](/assets/images/feynman-kac/feynman-kac-7.png)
    



```python

```


```python

```


```python
u_0 = lambda x: np.where(x <= 0, 1.0, 0.0)
f = lambda x, t: np.zeros_like(x)
sde = WienerProcess()
problem = FinalValueProblem(sde, u_0, -1.0, 1.0, 1.0, 0.0, f)
x_all = np.linspace(problem.domain().x_min(), problem.domain().x_max(), 512)
t_all = np.linspace(problem.domain().t_0(), problem.domain().T(), 16)
solutions_euler = solve(problem, x_all, t_all, 1)
solutions_cn = solve(problem, x_all, t_all, 0.5)
```


```python
import seaborn as sns
```


```python
t_all[-11]
```




    0.3333333333333333




```python
plt.plot(solutions_euler[-11,:], linewidth=4, label='Implicit Euler')
plt.plot(solutions_cn[-11,:], linewidth=4, label='Crank-Nicolson')
plt.legend()
plt.xlabel('x')
plt.ylabel('solution')
plt.title('Solution at t=1/3')
```




    Text(0.5, 1.0, 'Solution at t=1/3')




    
![png](/assets/images/feynman-kac/feynman-kac-8.png)
    



```python
XX, YY = plt.meshgrid(x_all, t_all)
fig, ax0 = plt.subplots(figsize=(10, 6), ncols=1, subplot_kw=dict(projection='3d'))
ax0.plot_surface(XX, YY, solutions_cn[::-1], cmap=cm.coolwarm)
# ax0.view_init(elev=30, azim=-45,)
```




    <mpl_toolkits.mplot3d.art3d.Poly3DCollection at 0x2036c1c3a00>




    
![png](/assets/images/feynman-kac/feynman-kac-9.png)
    



```python

```
