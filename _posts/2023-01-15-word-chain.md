---
layout: splash
permalink: /word-chain/
title: "Word Chain"
header:
  overlay_image: /assets/images/word-chain/word-chain-splash.jpg
excerpt: "A coding exercise that solves word-chain puzzles."
---

Another interesting coding exercise is the [word chain](http://codekata.com/kata/kata19-word-chains/):
given a corpus of words, the challenge is to build a chain of words, starting with one particular word and ending with another. Successive entries in the chain must all be words in the same corpus, and each can differ from the previous word by just one letter. For example, one can get from `black` to `white` using the following chain:

- `white` $\rightarrow$ `while`
- `while` $\rightarrow$ `whilk`
- `whilk` $\rightarrow$ `whick`
- `whick` $\rightarrow$ `whack`
- `whack` $\rightarrow$ `shack`
- `shack` $\rightarrow$ `slack`
- `slack` $\rightarrow$ `black`

The Python environment is quite simple and only requires a few packages. The list of words can be obtained in several ways; here we use the `words` corpus from the [nltk](https://www.nltk.org/) package. The corpus requires downloading before its first use.

```powershell
$ python -mv venv venv
$ ./venv/Scripts/activate
$ pip install nltk ipykernel numpy scipy
```


```python
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import nltk
nltk.download('words', quiet=True)
from nltk.corpus import words
```

This corpus is quite large, with more than 230,000 words. We don't use them all and rather focus on a subset of given length and limit ourselves to words of length 6 -- much longer words don't give rise to interesting chains, shorter words are more fun.


```python
dictionary = words.words()
print(f"Found {len(dictionary):,} words in the dictionary.")
```

    Found 236,736 words in the dictionary.
    


```python
WORD_LENGTH = 6
```


```python
subset = set()
for word in dictionary:
    if len(word) != WORD_LENGTH:
        continue
    if word.lower() != word:
        continue
    subset.add(word)
subset = list(subset)
print(f"Kept {len(subset)} lowercase words of length {WORD_LENGTH}.")
```

    Kept 15068 lowercase words of length 6.
    

We are ready to start. We need a function that computes the distance of two words, for which we assuem that the inputs have the same lenght (that is, the value `WORD_LENGHT` defined above). The logic is failry simple: we loop over all letters of both words, and track the number of different values. So `one` and `any` have a distance 2, for example. This logic is symmetric and as such it is a proper mathematical function of a distance between two objects.


```python
def compute_distance(word1, word2):
    assert len(word1) == len(word2)
    d = 0
    for c1, c2 in zip(word1, word2):
        if c1 != c2:
            d += 1
    return d
```

The next step is to compute the graph connecting all the words with a distance-1 transformation. Finding the word chain is then equivalent to find the shortest path between two nodes, which can be easily done using a variety of algorithms, including [Dijkstra's](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm). The advantage of this approach is the clear separation between the method used to defined the distance, which is the function above, and the solution, which can be delegated to well-tested software. Here we use scipy's implementation. (Using a dense matrix to start with is a bit of a waste of memory, but for relatively small number of words it can be done on today's computers without problems.) The construction of the matrix can take a moment or two depending on how large the corpus is, but it must be computed only once for any word chain of the same word length.


```python
n = len(subset)
D = np.zeros((n, n))
for i, w1 in enumerate(subset):
    for j, w2 in enumerate(subset):
        # skip the diagonal element
        if i <= j:
            continue
        d = compute_distance(w1, w2)
        # only keep the elements with distance 1 and ignore the others
        if d == 1:
            D[j, i] = D[i, j] = d
```

The computation of the shortest path used `scipy`'s method `shortest_path()`, which is quite fast.


```python
dist_matrix, predecessors = shortest_path(D, return_predecessors=True)
```

With words of length 6 we can easily connect words that are quite uncorrelated. For example, the shortest path to connect `likely` and `engage` has 24 steps:


```python
begin_word = 'likely'
end_word = 'engage'
assert begin_word in subset and end_word in subset

begin_index = subset.index(begin_word)
end_index = subset.index(end_word)
distance = dist_matrix[begin_index, end_index]
print(f'Distance between {begin_word} and {end_word} is {distance}.')

if distance < np.inf:
    path = []
    current_index = end_index
    while current_index != begin_index and current_index >= 0:
        path.append(current_index)
        current_index = predecessors[begin_index, current_index]
    path.append(current_index)
    path = list(reversed(path))
    for i, j in zip(path[:-1], path[1:]):
        word_i, word_j = subset[i], subset[j]
        print(f'{word_i} -> {word_j} (distance={compute_distance(word_i, word_j)})')
else:
    print('No word chain found, the two words are not connected.')
```

    Distance between likely and engage is 24.0.
    likely -> lively (distance=1)
    lively -> livery (distance=1)
    livery -> rivery (distance=1)
    rivery -> revery (distance=1)
    revery -> revere (distance=1)
    revere -> revete (distance=1)
    revete -> revote (distance=1)
    revote -> remote (distance=1)
    remote -> remove (distance=1)
    remove -> relove (distance=1)
    relove -> belove (distance=1)
    belove -> belive (distance=1)
    belive -> belite (distance=1)
    belite -> pelite (distance=1)
    pelite -> polite (distance=1)
    polite -> podite (distance=1)
    podite -> iodite (distance=1)
    iodite -> indite (distance=1)
    indite -> incite (distance=1)
    incite -> uncite (distance=1)
    uncite -> uncate (distance=1)
    uncate -> uncage (distance=1)
    uncage -> encage (distance=1)
    encage -> engage (distance=1)
    
