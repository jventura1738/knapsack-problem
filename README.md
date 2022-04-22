# knapsack-problem
Python Genetic Algorithm Solution by Justin Ventura

---
## How it works:

We have a knapsack that has the capability to hold a certain weight of treasures, with each treasure having a weight and a value, and the goal is to maximize the value in the knapsack after collecting some subset of a finite amount of treaures.

### Example:
Consider a knapsack `K(cap=15)` which has a maximum weight capacity of 15.

Now consider the set of treasures
```
    T = [
        t1(w=5, v=8),
        t2(w=7, v=7),
        t3(w=1, v=5),
        t4(w=7, v=3),
        t5(w=2, v=1)
    ]
```
We wish to find the combination of treasures in `T` that maximizes the sum of the values of the treaures.

Potential solutions take the form:
```
    [0, 1, 1, 1, 0],
    [1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0]
```
Where a 0 at index i means treasure i is not in the solution sum, and a 1 at index i means treasure i is in the solution sum.

Note that `[1, 1, 0, 1, 0]` is not a possible solution since the sum of the weights corresponding to the treasures implied by this list exceeds 15.