---
layout: splash
permalink: /factorial-decomposition/
title: "Factorial Decomposition"
header:
  overlay_image: /assets/images/factorial-decomposition/factorial-decomposition.jpg
excerpt: "A coding exercise to decompose a factorial."
---

The aim of the exercise is to decompose the factorial of n, $n!$, into its prime factors.

Examples:

- for $n = 12$, the decomposition should give `2^10 * 3^5 * 5^2 * 7 * 11`;
- for $n = 22$, the decomposition should give `2^19 * 3^9 * 5^4 * 7^3 * 11^2 * 13 * 17 * 19`;
- for $n = 25$, the decomposition should give `2^22 * 3^10 * 5^6 * 7^3 * 11^2 * 13 * 17 * 19 * 23`.

Prime numbers should be in increasing order. When the exponent of a prime is 1 don't put the exponent.

The first part of this exercise is to be able to compute all the factors of a number $i$ using a function
`prime_factors()`; computing the
factors of $n! = 1 \times 2 \times 3 \times \cdots \times n$ is then just applying the factorization for each $i$ in  for loop with some bookkeeping. 

The factorization of a number is quite easy and can be found in lots of places, like this [stackoverflow answer](https://stackoverflow.com/questions/15347174/python-finding-prime-factors). Just applying this function naïvely, though, can be inefficient for large values of $n$. We solve the problem by making sure that `prime_factors()` goes through all the numbers rercursively and by adding a cache to avoid repeated calls.


```python
from collections import defaultdict
from functools import lru_cache
import matplotlib.pylab as plt
import numpy as np
```


```python
@lru_cache(maxsize=None)
def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors += prime_factors(i)
    if n > 1:
        factors.append(n)
    return factors
```

Before solving the exercise, we can have some fun and plot the number of prime numbers in the interval $[1, n]$,
usually indicated with $\pi(n)$. It is well-known that $\pi(n) \approx \frac{n}{\ln n + 1}$.


```python
x = list(range(2, 100_000))
y = []
for i in x:
    factors = prime_factors(i)
    y.append(0 if len(factors) > 1 else 1)
```


```python
plt.figure(figsize=(10, 4))
plt.plot(np.cumsum(y), label='exact')
plt.plot(x, x / (np.log(x) - 1), label='approximation')
plt.legend()
plt.xlabel('$n$')
plt.ylabel('$\pi(n)$');
```




    Text(0, 0.5, '$\\pi(n)$')




    
![png](/assets/images/factorial-decomposition/factorial-decomposition-1.png)
    


The solution of the exercise is now quite simple and is implemented below. The function `factorial_decomposition()` has two parts: in the first we compute the decomposition; in the second we assemble it in the required format.


```python
def factorial_decomposition(n):
    decomp = defaultdict(lambda: 0)
    for i in range(2, n + 1):
        factors = prime_factors(i)
        for factor in factors:
            decomp[factor] += 1

    keys = sorted(decomp.keys())
    retval = []
    for key in keys:
        value = decomp[key]
        if value == 1:
            retval.append(str(key))
        else:
            retval.append(f'{key}^{value}')
    return ' * '.join(retval)
```

We can check the examples at the top:


```python
assert factorial_decomposition(12) == '2^10 * 3^5 * 5^2 * 7 * 11'
assert factorial_decomposition(22) == '2^19 * 3^9 * 5^4 * 7^3 * 11^2 * 13 * 17 * 19'
assert factorial_decomposition(25) == '2^22 * 3^10 * 5^6 * 7^3 * 11^2 * 13 * 17 * 19 * 23'
```

The code is quite fast, also for large factorials. Calling `factorial_decomposition(40_000)` is a fraction of a second; on the slow computer I am using `factorial_decomposition(400_000)` is just above 5 seconds, which is still not bad.
