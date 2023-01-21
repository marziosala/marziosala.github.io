---
layout: splash
permalink: /roman-numerals/
title: "Roman Numerals"
header:
  overlay_image: /assets/images/roman-numerals/roman-numerals-splash.png
excerpt: "Converting a number to Roman numerals."
---

[Roman numerals](https://en.wikipedia.org/wiki/Roman_numerals) are a numeral system that originated in ancient Rome and remained the usual way of writing numbers throughout Europe well into the Late Middle Ages. The basic rules are well-known: numbers are written with combinations of letters from the Latin alphabet, each letter with a fixed integer value.
Although different style have evoled during the millenia, the modern style uses only these seven
letters: `I` for 1, `V` for 5, `X` for 10, `L` for 50, `C` for 100, `D` for 500 and `M` for 1,000. There is no zero and different letters are used for each power of ten (up to 1,000) instead of the positional system of [Arabic numbers](https://en.wikipedia.org/wiki/Arabic_numerals).

In general, writing a number in Roman numerals is quite easy: a letter repeats its value that many times (`XXX` is 30 and `CC` is 200, for example).  Letters go from the highest amount to the lower, so the `X` goes before the `I`, so with letters placed after another letter of greater value, we add that amount. For example, the number 22 is `XXII`, with the two `X` for 20 and the two `I` for the 2.

There are exceptions, however. A letter can only be repeated three times, so
the numerals for 4 (`IV`) and 9 (`IX`) are written using *subtractive notation*, where the first symbol (`I`) is subtracted from the larger one (`V`, or `X`), thus avoiding the clumsier IIII and VIIII. Subtractive notation is also used for 40 (`XL`), 90 (`XC`), 400 (`CD`) and 900 (`CM`). These are the only subtractive forms in standard use; subtractive notation is not used for the numbers 99 or 999, for example: we do not subtract a number from one that is more than 10 times greater (that is, you can subtract 1 from 10 but not 1 from 20, there is no such number as `IXX`.)

Note that the subtractive notation means that if a letter is placed before another letter of greater value, we subtract that amount.

Below there is a simple function that converts a number $n \subset \mathbb{N}$ such that
$n \in [1, 3 999]$ from Arabic to Roman. The function is quite simple and makes use of generators to go through the different steps.


```python
def to_roman(n):
    def inner(n):
        if n >= 1000:
            yield 'M' * (n // 1000)
            n %= 1_000
        if n >= 900:
            yield 'CM'
            n -= 900
        if n >= 500:
            yield 'D'
            n -= 500
        if n >= 400:
            yield 'CD'
            n -= 400
        if n >= 100:
            yield 'C' * (n // 100)
            n %= 100
        if n >= 90:
            yield 'XC'
            n -= 90
        if n >= 50:
            yield 'L'
            n -= 50
        if n >= 40:
            yield 'XL'
            n -= 40
        if n >= 10:
            yield 'X' * (n // 10)
            n %= 10
        if n >= 9:
            yield 'IX'
            n -= 9
        if n >= 5:
            yield 'V'
            n -= 5
        if n == 4:
            yield 'IV'
            n -= 4
        yield 'I' * n
    
    if n < 1 or n > 3_999:
        raise ValueError('input n must be in the interval [1, 3999]')

    return ''.join(inner(n))
```

A few tests, mostly taken from the Wikipedia page dedicated to Roman numerals, plus a few corner cases.


```python
assert to_roman(39) == 'XXXIX'
assert to_roman(246) == 'CCXLVI'
assert to_roman(789) == 'DCCLXXXIX'
assert to_roman(2421) == 'MMCDXXI'
assert to_roman(160) == 'CLX'
assert to_roman(207) == 'CCVII'
assert to_roman(1009) == 'MIX'
assert to_roman(1066) == 'MLXVI'
assert to_roman(1776) == 'MDCCLXXVI'
assert to_roman(1918) == 'MCMXVIII'
assert to_roman(1944) == 'MCMXLIV'
assert to_roman(2023) == 'MMXXIII'
assert to_roman(999) == 'CMXCIX'
assert to_roman(3999) == 'MMMCMXCIX'
assert to_roman(90) == 'XC'
assert to_roman(400) == 'CD'
assert to_roman(900) == 'CM'
```
