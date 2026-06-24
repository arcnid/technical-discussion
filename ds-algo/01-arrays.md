# 01 — Arrays

The grandparent data structure. Everything else in this lesson is in some way trying to fix something arrays are bad at.

---

## Mental model

A **contiguous block of memory** where each slot holds one value and each slot has a numeric index. The "contiguous" part is what makes arrays so fast for index access: the runtime can compute *exactly* where slot `n` lives by doing `start_address + n * slot_size`. One arithmetic step, no searching.

```
index:   0     1     2     3     4     5
       ┌─────┬─────┬─────┬─────┬─────┬─────┐
value: │  7  │  3  │  9  │  1  │  4  │  6  │
       └─────┴─────┴─────┴─────┴─────┴─────┘
        ↑                                 ↑
       arr[0]                          arr[5]
```

JS arrays are not *really* contiguous memory under the hood — V8 uses a few representations and can degrade to hashmap-like storage if you do weird things (`arr[1e9] = 1`). For lesson purposes: pretend they're contiguous. The performance characteristics match.

---

## Operations and Big-O

| Operation | Big-O | Why |
|-----------|-------|-----|
| `arr[i]` (read by index) | O(1) | Direct memory address |
| `arr[i] = x` (write by index) | O(1) | Same |
| `arr.push(x)` (append) | O(1)* | *Amortized — occasionally has to resize |
| `arr.pop()` (remove last) | O(1) | No shifting needed |
| `arr.unshift(x)` (prepend) | O(n) | Every other element shifts right by 1 |
| `arr.shift()` (remove first) | O(n) | Every other element shifts left by 1 |
| `arr.includes(x)` / linear search | O(n) | Has to check every slot |
| `arr.indexOf(x)` | O(n) | Same |
| `arr.sort()` | O(n log n) | Timsort |

**Key takeaway:** index access is free. Anything that touches the front of the array (or searches without an index) is linear.

---

## Code: the basics

```js
// Creating
const arr = [7, 3, 9, 1, 4, 6];
const empty = [];
const filled = new Array(5).fill(0);   // [0, 0, 0, 0, 0]

// Reading
arr[0];           // 7
arr[arr.length - 1]; // 6  -- last element
arr.at(-1);       // 6  -- also last element, cleaner

// Mutating
arr.push(11);     // [7, 3, 9, 1, 4, 6, 11]
arr.pop();        // [7, 3, 9, 1, 4, 6]    returns 11
arr[2] = 99;      // [7, 3, 99, 1, 4, 6]

// Iterating
for (let i = 0; i < arr.length; i++) { ... }
for (const x of arr) { ... }
arr.forEach((x, i) => { ... });
arr.map(x => x * 2);
arr.filter(x => x > 3);
arr.reduce((acc, x) => acc + x, 0);
```

---

## The two-pointer pattern (advanced — but easy)

This is the trick we'll use for the palindrome problem. It's the most useful array pattern to know.

Idea: instead of one index walking the array, use **two** — usually one at each end, moving toward each other.

```
[ a, b, c, d, c, b, a ]
  ↑                 ↑
  L                 R     compare arr[L] with arr[R]
                          if equal, move both inward
                          if not equal, not a palindrome
```

```js
const isPalindrome = (arr) => {
  let l = 0;
  let r = arr.length - 1;
  while (l < r) {
    if (arr[l] !== arr[r]) return false;
    l++;
    r--;
  }
  return true;
};
```

**Why it's fast:** one pass, O(n). No extra memory. Compare with the obvious "reverse the array and check equality" approach, which also works but allocates a whole second array.

Other places two-pointer shows up: removing duplicates from a sorted array, "container with most water," merging two sorted arrays, sliding-window problems.

---

## FAQ

**Q: Why is `unshift` so slow?**
Because everything else has to move. If you insert at index 0, indices 1 through n-1 all need new positions. The runtime literally walks through and shifts each one. If you find yourself doing `arr.unshift` in a loop, you're doing O(n²) work for what could be O(n) (collect into the array, then reverse at the end).

**Q: Should I ever care about the difference between an array and a linked list?**
For the day job: rarely. Arrays win for almost everything because cache locality is *huge* — modern CPUs are way faster at sequential memory access than at chasing pointers around the heap. Linked lists win for "I need O(1) insertion in the middle and I already have a reference to the node," which is a fairly niche situation.

**Q: What does `arr.length = 0` do?**
Truncates the array to empty. Yes, it works. Yes, it's a little cursed. `arr = []` is more obvious unless you specifically need to mutate the same reference (e.g., other code holds a reference to the same array).

**Q: Are JS arrays really arrays or are they secretly objects?**
Both, kind of. `typeof []` returns `"object"`. Under the hood V8 has multiple representations and will pick fast ones (packed-smi, packed-double, packed-elements) when possible, degrading to dictionary mode if you do things like assign at huge indices, delete properties, etc. For 99% of code: pretend it's a real array.

**Q: Why does `arr[1e9] = 1` make my program slow?**
You just hinted to V8 that this is a sparse array, which kicks it out of the fast representation and into dictionary mode (basically a hashmap). All subsequent operations are slower. Lesson: keep arrays dense.

**Q: What about `TypedArray` (Uint8Array, etc)?**
These are *actually* contiguous fixed-size memory. Useful for binary data, performance-critical inner loops, or when you need to interop with WebGL / WASM / network buffers. Not something you'll reach for in app code often.

---

## Discussion prompts

- "Is `arr[5]` faster than `obj.x`?" (Both O(1), but arr is often faster because of cache locality + simpler internal representation.)
- "If I want a sorted collection with fast insert, is an array a good choice?" (No — insert is O(n). Use a BST or a heap depending on the access pattern.)
- "What's the difference between `arr.slice()` and `arr.splice()`?" (Classic interview gotcha. `slice` is non-mutating, returns a copy. `splice` mutates in place. Names are confusingly similar.)

---

## Next up

→ [02-hashmaps-objects.md](02-hashmaps-objects.md) — when O(n) array search isn't fast enough.
