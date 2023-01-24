---
layout: splash
permalink: /poker-hand/
title: "Poker Hand"
header:
  overlay_image: /assets/images/poker-hand/poker-hand-splash.jpg
excerpt: "A coding exercise that identifies the poker hand."
---

Recently I stumbled upon a nice coding exercise that requested to output the type of Poker hand. Specifically, the request is to return the following strings that indicate the highest level of the hand:

- **Royal Flush** - Special version of straight flush with A, K, Q, J, 10
- **Straight Flush** - Five cards in sequence of the same suit.
- **Four of a Kind** - four cards of same rank.
- **Full House** - three cards of same rank + two cards of same rank.
- **Flush** - five cards of the same suit.
- **Straight** - five cards in sequence of different suits.
- **Three of a Kind** - three cards of same rank.
- **Two Pair** - two sets of two cards of same rank.
- **One Pair** - one set of two cards of same rank.
- (High Card) - return highest ranking card in hand.

The implemetation uses standard Python classes and requires Python 3.10 at least because of the pattern matching syntax.

The idea is the following:
- an enum defines the rank (13 possibilities);
- another enum defines the suit (4 possibilities);
- a card is a defined by a rank and a suit (52 possibilities);
- a hand is defined by 5 cards (from the same deck, so they must all be different).

The `rank()` function below returns the required string. To do that, it first orders the hand
using a specific logic, as defined in the sorted_hand() method. Cards are ordered
by the rank (and the suit is ignored), however cards with identical rank are
always consecutive and ordered such they are always at the beginning.
So if we have four of a kind, they will always be in positions [1, 2, 3, 4];
the same for three of a kind. Two of a kind either follow a three of a kind,
or another two of a kind, or are first. Cards are otherwise ordered in ascending
order from 1 (the ace) to 13 (the king). The ordering reduces the complexity
of the pattern matching, also making testing much simpler.


```python
from enum import IntEnum, Enum
from collections import namedtuple
from dataclasses import dataclass
```


```python
class Rank(IntEnum):
    """Use IntEnum such that we can check that TWO = ACE + 1"""
    ACE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    J = 11
    Q = 12
    K = 13
```


```python
class Suit(Enum):
    """The order of the suit if not relevant for this exercise, so just use Enum"""
    HEARTS = 1
    DIAMONDS = 2
    CLUBS = 3
    SPADES = 4
```

A `Card` is the combination of the `Rank` and the `Suit`.


```python
Card = namedtuple('Card', 'rank suit')
```

An additional data structure, called `Hand` is introduced to identify 5 cards. We use `dataclass` instead of `namedtuple` because we have to have a custom initialization, which isn't possible with `namedtuple`. The custom initialization is used to check that all the five cards come from the same deck, and as such we can't have identical cards. This is because the pattern matching requires different cards.


```python
@dataclass
class Hand:
    c1: Card
    c2: Card
    c3: Card
    c4: Card
    c5: Card
    
    def __post_init__(self):
        cards = [self.c1, self.c2, self.c3, self.c4, self.c5]
        if len(set(cards)) != 5:
            raise ValueError('The hand contains at least two identical cards')
    
    def __iter__(self):
        yield self.c1
        yield self.c2
        yield self.c3
        yield self.c4
        yield self.c5
```

The `sorted_hand()` function is the key one, because it massively simplifies the pattern matching below. What it does is to sort the hand by the rank in the following way:
- cards with identical ranks are always consecutive
- if we have four cards with the same rank, they are always returned in position [#1, #2, #3, #4],
  and the remaining card is in position #5
- if we have three cards with the same rank, they are always returned in position [#1, #2, #3],
   with the remaining cards in position #4 and #5
- if we have two cards with the same rank, they are returned either after the ones 
  of the point above, or at position [#1, #2], or at position [#3, #4] if we also have another
  group of two cards with the same rank
- otherwise, cards are ordered from 1 (the ace) to K, following the order of the enum
 
The suits are ignored, but can be easily inserted as well. 


```python
def sorted_hand(hand: Hand):
    hand = sorted(hand, key=lambda x: x.rank)
    result = [[hand[0]]]
    for i in hand[1:]:
        if i.rank == result[-1][0].rank:
            result[-1].append(i)
        else:
            result.append([i])
    # the number 1000 gives priority to the number of identical ranks, with the second
    # component ordering by the rank
    retval = sorted(result, key=lambda x: -len(x) * 1000 + x[0].rank)
    # flatten the list
    return sum(retval, [])
```

We also have a conveniency class, `HandRank`, which is not really needed but makes the code a bit cleaner. It will also be quite convenient later on to test the outcome by comparing the same string.


```python
class HandRank:
    """Return values as requested by the exercise"""
    ROYAL_FLUSH = "Royal Flush"
    STRAIGHT_FLUSH = "Straight Flush"
    FOUR_OF_A_KIND = "Four of a Kind"
    FULL_HOUSE = "Full House"
    FLUSH = "Flush"
    STRAIGHT = "Straight"
    THREE_OF_A_KIND = "Three of a Kind"
    TWO_PAIR = "Two Pair"
    ONE_PAIR = "One Pair"
```

Finally, the `rank()` that takes in input the hand and returns the corresponding string. 


```python
def rank(hand: Hand):
    """Ranks a hand and returns one of values in HandRank, or the value of the
    highest card if no other combination is present"""

    # small utility function to check that all the arguments are the same
    all_same = lambda *args: len(set(args)) == 1
    
    # first we first the hand as specified in the comments of sorted_hand(), to
    # reduce the patterns we have to match
    hand = sorted_hand(hand)

    # look at the Poker patterns
    match hand:
        # Special version of straight flush with A, K, Q, J, 10
        case (Rank.ACE, s1), (Rank.TEN, s2), (Rank.J, s3), (Rank.Q, s4), (Rank.K, s5) \
                if all_same(s1, s2, s3, s4, s5):
            return HandRank.ROYAL_FLUSH
        # Five cards in sequence of the same suit
        case (r1, s1), (r2, s2), (r3, s3), (r4, s4), (r5, s5) \
                if all_same(s1, s2, s3, s4, s5) \
                and r2 == r1 + 1 and r3 == r2 + 1 and r4 == r3 + 1 and r5 == r4 + 1:
            return HandRank.STRAIGHT_FLUSH
        # four cards of same rank
        case (r1, s1), (r2, s2), (r3, s3), (r4, s4), (r5, s5) \
                if all_same(r1, r2, r3, r4):
            return HandRank.FOUR_OF_A_KIND
        # three cards of same rank + two cards of same rank
        case (r1, s1), (r2, s2), (r3, s3), (r4, s4), (r5, s5) if all_same(r1, r2, r3) and r4 == r5:
            return HandRank.FULL_HOUSE
        # five cards of the same suit
        case (_, s1), (_, s2), (_, s3), (_, s4), (_, s5) if all_same(s1, s2, s3, s4, s5):
            return HandRank.FLUSH
        # five cards in sequence of different suits
        case (r1, _), (r2, _), (r3, _), (r4, _), (r5, _) \
                if r2 == r1 + 1 and r3 == r2 + 1 and r4 == r3 + 1 and r5 == r4 + 1:
            return HandRank.STRAIGHT
        # as above, with the ace at the end
        case (Rank.ACE, _), (Rank.TEN, _), (Rank.J, _), (Rank.Q, _), (Rank.K, _):
            return HandRank.STRAIGHT
        # three cards of same rank.
        case (r1, _), (r2, _), (r3, _), _, _ if all_same(r1, r2, r3):
            return HandRank.THREE_OF_A_KIND
        # two sets of two cards of same rank
        case (r1, _), (r2, _), (r3, _), (r4, _), (_, _) if r1 == r2 and r3 == r4:
            return HandRank.TWO_PAIR
        # one set of two cards of same rank
        case (r1, _), (r2, _), _, _, _ if r1 == r2:
            return HandRank.ONE_PAIR
        # return highest ranking card in hand.
        case _, _, _, _, c5:
            return c5
```

Let's test now. To make things simpler, a few helper functions are defined to 
create a Hand object from a string. Those functions are not part of the main code
and are reported here as they would rather be part of the tests. 


```python
def rank_from_string(s: str):
    match s:
        case '1' | 'ACE':
            return Rank.ACE
        case 'J':
            return Rank.J
        case 'Q':
            return Rank.Q
        case 'K':
            return Rank.K
        case '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' | '10':
            return Rank(int(s))
        case _:
            raise ValueError(f'input string {s} not recognized')
```


```python
def suit_from_string(s: str):
    match s:
        case '♡' | 'H':
            return Suit.HEARTS
        case '♢' | 'D':
            return Suit.DIAMONDS
        case '♧' | 'C':
            return Suit.CLUBS
        case '♤' | 'S':
            return Suit.SPADES
        case _:
            raise ValueError(f'iinput string {s} not recognized')
```


```python
def card_from_string(s: str):
    rank = rank_from_string(s[:-1].strip())
    suit = suit_from_string(s[-1].strip())
    return Card(rank, suit)
```


```python
def hand_from_string(s: str):
    cards = []
    for card in s.split(','):
        cards.append(card_from_string(card))
    return Hand(*cards)
```

The actual tests, with at least one per pattern to match.


```python
import random
random.seed(42)
```


```python
all_ranks = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
all_suits = ['♡', '♢', '♧', '♤']
```


```python
for suit in all_suits:
    hand = hand_from_string('10 {0}, J {0}, Q {0}, K {0}, ACE {0}'.format(suit))
    assert rank(hand) == HandRank.ROYAL_FLUSH
```


```python
for suit in all_suits:
    hand = hand_from_string('9 {0}, 10 {0}, J {0}, Q {0}, K {0}'.format(suit))
    assert rank(hand) == HandRank.STRAIGHT_FLUSH
```


```python
for r1 in all_ranks:
    for r2 in all_ranks:
        if r2 != r1:
            cards = [f'{r1} ♧', f'{r1} ♡', f'{r1} ♢', f'{r1} ♤', f'{r2} ♡']
            random.shuffle(cards)
            cards = map(card_from_string, cards)
            hand = Hand(*cards)
            assert rank(hand) == HandRank.FOUR_OF_A_KIND
```


```python
hand = hand_from_string('10 ♧, J ♡, Q ♡, K ♡, ACE ♧')
assert rank(hand) == HandRank.STRAIGHT
```


```python
hand = hand_from_string('3 ♤, 3 ♡, Q ♡, 3 ♧, Q ♧')
assert rank(hand) == HandRank.FULL_HOUSE
```


```python
hand = hand_from_string('ACE ♤, 2 ♡, 3 ♡, 4 ♧, 5 ♧')
assert rank(hand) == HandRank.STRAIGHT
```


```python
hand = hand_from_string('6 ♤, 2 ♡, 3 ♡, 4 ♧, 5 ♧')
assert rank(hand) == HandRank.STRAIGHT
```


```python
hand = hand_from_string('6 ♤, 2 ♡, 6 ♡, 6 ♧, J ♧')
assert rank(hand) == HandRank.THREE_OF_A_KIND
```


```python
hand = hand_from_string('6 ♤, 2 ♡, 10 ♡, 6 ♧, 2 ♧')
assert rank(hand) == HandRank.TWO_PAIR
```


```python
hand = hand_from_string('6 ♤, 2 ♡, 10 ♡, 6 ♧, 3 ♧')
assert rank(hand) == HandRank.ONE_PAIR
```


```python
hand = hand_from_string('Q ♤, 2 ♡, 10 ♡, 6 ♧, 3 ♧')
assert rank(hand) == Card(Rank.Q, Suit.SPADES)
```
