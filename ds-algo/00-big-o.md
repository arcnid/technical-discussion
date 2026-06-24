# 00 — Big-O Notation (Preface)

Every file in this lesson plan tosses around things like "O(n)", "O(log n)", "O(1) average." If those notations feel like jargon, this page demystifies them in about 10 minutes. If they already feel obvious, skim and move on.

The big idea: **Big-O is a way to describe how an algorithm's cost grows as the input grows.** Nothing more. It's not about clock time, not about machine specs, not about how good your code is at micro-optimizations. It's about the *shape* of the cost curve.

---

## Why we need it

Imagine two functions that both sort a list. On a list of 10 items, both finish in under a millisecond on your laptop. Which is "faster"?

Wrong question — at small sizes, almost everything is fast. The right question is **what happens when the list has 10 million items**:

- Function A: takes 50 ms.
- Function B: takes 3 hours.

Big-O is the language we use to predict this *without actually running the code*. It tells you the **growth rate**, which is the only thing that matters at scale.

---

## The notation

`O(f(n))` means: "as the input size `n` grows, the running time grows roughly proportional to `f(n)`."

Two important conventions:
1. **Drop the constants.** `O(2n)` and `O(500n)` are both written as `O(n)`. The point is the shape, not the multiplier.
2. **Drop the lower-order terms.** `O(n² + n + 100)` becomes `O(n²)`. At large n the squared term dominates everything else.

So Big-O is a coarse but very useful classification. It lumps a 50-ms algorithm and a 200-ms algorithm into the same bucket (`O(n)`) because *both* will scale the same way as n grows.

---

## The complexity classes you'll see this week

From fastest to slowest:

```
O(1)        constant         doesn't depend on n at all
O(log n)    logarithmic      halving each step
O(n)        linear           one pass
O(n log n)  linearithmic     sort-shaped
O(n²)       quadratic        nested loops
O(2ⁿ)       exponential      "tries every subset" — usually a red flag
O(n!)       factorial        permutations — almost always too slow
```

### A picture of how brutal the difference is

For n = 1,000,000:

```
O(1)        ─                                        1 step
O(log n)    ──                                       ~20 steps
O(n)        ──────                                   1 million steps
O(n log n)  ───────────────                          ~20 million steps
O(n²)       ─────────────────────────────────...    1 trillion steps  ← won't finish
O(2ⁿ)       (heat death of the universe)
```

That gap is why "O(n²) vs O(n)" matters more than any micro-optimization. You're not picking between fast and faster — you're picking between "ships" and "doesn't ship."

---

## Examples in code

### O(1) — constant

The cost doesn't depend on the input size at all.

```js
const first = (arr) => arr[0];
const isEven = (n) => n % 2 === 0;
const map = new Map();  map.get("key");
```

Direct array index, hashmap lookup, arithmetic — all O(1).

### O(log n) — logarithmic

Each step throws away a fixed fraction (usually half) of the remaining work.

```js
const binarySearch = (sorted, target) => {
  let l = 0, r = sorted.length - 1;
  while (l <= r) {
    const mid = (l + r) >> 1;
    if (sorted[mid] === target) return mid;
    if (sorted[mid] < target) l = mid + 1;
    else r = mid - 1;
  }
  return -1;
};
```

For a million elements, ~20 comparisons total. Searching a balanced BST is O(log n) for the same reason.

### O(n) — linear

You touch each element once.

```js
const sum = (arr) => {
  let total = 0;
  for (const x of arr) total += x;
  return total;
};

arr.includes(x);
arr.indexOf(x);
arr.filter(...);
```

Walking the array, counting things, transforming each element — all O(n).

### O(n log n) — linearithmic

The "sorting" complexity. Most general-purpose sorts (Merge sort, Heap sort, Timsort — which is what JS `.sort()` uses) are O(n log n).

```js
arr.sort();              // O(n log n)
arr.sort((a, b) => a - b);
```

You can't sort comparison-based faster than this in the general case. Proven.

### O(n²) — quadratic

Nested loop where both go over the input.

```js
const hasDuplicates = (arr) => {
  for (let i = 0; i < arr.length; i++) {
    for (let j = i + 1; j < arr.length; j++) {
      if (arr[i] === arr[j]) return true;
    }
  }
  return false;
};
```

For each element, scan the rest. n × n = n². Acceptable for n ≤ a few thousand. Unacceptable for n in the millions.

Note: the duplicate check above can be done in O(n) with a `Set`. **Almost every nested loop you write has an O(n) hashmap version waiting to be discovered.** That's literally the topic of file `02`.

### O(2ⁿ) — exponential

Usually shows up when you naively try every subset / every combination.

```js
// Compute the n-th Fibonacci number with naive recursion.
const fib = (n) => n < 2 ? n : fib(n - 1) + fib(n - 2);
fib(40);   // takes a few seconds
fib(50);   // takes minutes
fib(60);   // good luck
```

Each call branches into two. The call tree has ~2ⁿ nodes. Fixing this — using memoization or iteration — is the prototypical "dynamic programming" trick.

---

## Reading Big-O off code (the eyeball method)

You don't need calculus. Just:

1. **One pass through `n` items?** → contributes `O(n)`.
2. **Loop inside a loop, both over `n`?** → multiply → `O(n²)`.
3. **Halving each step (binary search, divide and conquer)?** → `O(log n)`.
4. **Hashmap lookup, array index, arithmetic?** → `O(1)`.
5. **Call yourself twice per call?** → typically `O(2ⁿ)` unless memoized.
6. **Add up all the parts, keep the biggest, drop constants.**

Worked example:

```js
const f = (arr) => {
  const n = arr.length;
  for (let i = 0; i < n; i++) {            // O(n)
    arr.sort();                            // O(n log n) — INSIDE a loop!
  }
  return arr[0];                           // O(1)
};
```

`n` iterations × `n log n` per iteration = `O(n² log n)`. Catastrophic. (Yes, this is a contrived example. People still write things this shape.)

---

## Average vs worst vs amortized

This trips people up. Three different ways to talk about cost.

### Worst case

The pathological input. "How slow can this go?" Useful when you can't control the input (user data, adversarial input, real-world distributions you don't understand).

### Average case

What happens on "typical" input. Useful but slippery — depends on your definition of typical. Hashmap lookup is `O(1) average` because most keys land in different buckets, but `O(n) worst case` if everything collides into the same bucket.

### Amortized

The *long-run average per operation*. Even if a single operation occasionally costs a lot, if the **average over many operations** stays low, we call it amortized cheap.

Classic example: `arr.push(x)`.

- Usually `O(1)` — append to the end.
- Occasionally `O(n)` — the underlying array filled up and has to be resized (allocate a bigger array, copy everything over).
- But resizes are *rare* (the array typically doubles in capacity), so over a million pushes the average per push stays `O(1)`.

We say `push` is `O(1) amortized` — meaning if you average over many calls, you pay constant time per call, even though individual calls can spike.

---

## Space complexity

Same idea, different resource. Instead of "how many operations," it's "how much extra memory."

```js
const sum = (arr) => arr.reduce((a, b) => a + b, 0);
// time: O(n), space: O(1) — we only hold one running total
```

```js
const doubled = (arr) => arr.map(x => x * 2);
// time: O(n), space: O(n) — we allocate a whole new array
```

Two-pointer palindrome (file 01): `O(n)` time, `O(1)` space.
Hashmap two sum (file 02): `O(n)` time, `O(n)` space.

There's almost always a time/space tradeoff. The hashmap solution to two sum is faster than the nested-loop version *because we spent some memory to remember things we've seen*. That's the deal you're making, explicit.

---

## What Big-O doesn't capture

Two algorithms with the same Big-O can have wildly different real-world performance. Things Big-O ignores:

- **Constants.** A `5n` algorithm is 5× slower than a `n` algorithm in practice. Both are `O(n)`.
- **Cache locality.** Arrays beat linked lists at "linear scan" even though both are `O(n)` — because modern CPUs are way faster at sequential memory than at chasing pointers.
- **Branch prediction, SIMD, parallelism.** Real CPUs have tricks. Big-O assumes a hypothetical machine.
- **Input distribution.** Quicksort is `O(n²)` worst-case but in practice almost always behaves `O(n log n)` because pathological inputs are rare.

So Big-O is your **first-order tool**. It tells you which algorithms are even worth considering. Past that, you measure.

---

## FAQ

**Q: What's the difference between O, Θ (theta), and Ω (omega)?**
Formally: Big-O is an *upper* bound, Big-Ω is a *lower* bound, Big-Θ is *both* (tight bound). In practice in industry, people say "O" when they mean "Θ" — i.e., "this is exactly how it scales," not "this scales no worse than." Don't sweat the distinction unless you're writing a paper.

**Q: Is `O(n)` always faster than `O(n²)`?**
At large n, yes. At tiny n, no — the constants matter. A 1000n algorithm beats a 5n² algorithm only when n > 200. For n=10, the quadratic version wins. This is why JavaScript's `.sort()` uses insertion sort for tiny arrays even though it's `O(n²)`.

**Q: Is `O(log n)` base-2 or base-10?**
Doesn't matter — Big-O ignores constant factors, and `log₂(n) = log₁₀(n) × constant`. They're the same complexity class. We just say "log n."

**Q: What's `O(n + m)`?**
Means the cost depends on *two* inputs of different sizes. Example: merging two sorted arrays of lengths n and m costs `O(n + m)`. Don't combine them into a single variable — they can vary independently.

**Q: Why do hashmaps say "O(1) average" instead of just "O(1)"?**
Because the worst case is O(n) — if every key collides into the same bucket, lookup degrades to a linear scan. With good hash functions and reasonably distributed keys, this basically never happens in practice. The hedge is technically correct and you can usually ignore it.

**Q: I see "O(n^k)" sometimes — what's k?**
A variable polynomial degree. "k" is a parameter the algorithm or problem takes. For example, "k-sum" problems (find k numbers in an array that sum to a target) are typically `O(n^(k-1))`. Two-sum is k=2 → `O(n)`. Three-sum is k=3 → `O(n²)`. And so on.

**Q: Is `O(1)` always better than `O(log n)`?**
Asymptotically, yes. Practically, often no. A hashmap lookup is `O(1)` but involves hashing the key, possibly handling collisions, etc. A BST lookup is `O(log n)` but each step is just a comparison and a pointer follow. For small n, BST can win. For large n, hashmap wins. **Measure when it matters.**

**Q: What complexity is "JavaScript code that calls an API"?**
Network calls dominate everything. Big-O of your local code is irrelevant compared to the round-trip latency. This is a reminder that Big-O is a model — it's about CPU operations, not real-world end-to-end latency.

---

## Quick self-check

Before moving on, you should be able to answer these without looking back:

1. What's the Big-O of `arr.includes(x)`?
   → `O(n)`. Linear scan.
2. What's the Big-O of `map.get(key)`?
   → `O(1)` average.
3. What's the Big-O of `arr.sort()`?
   → `O(n log n)`.
4. What's the Big-O of a binary search on a sorted array?
   → `O(log n)`.
5. What's the Big-O of two nested loops, each going over `n` items?
   → `O(n²)`.

If those felt natural: you're set. The Big-O language in every other file will land.

---

## Next up

→ [01-arrays.md](01-arrays.md) — start with the data structure you already know best, with a Big-O lens this time.
