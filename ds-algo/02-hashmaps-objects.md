# 02 — Hashmaps (Objects + Map)

The "what if array lookup was O(1) by **value** instead of by index" data structure. Single most useful tool on this list.

---

## Mental model

A hashmap is a **lookup table where the key can be anything** (well — almost anything). You give it a key, it gives you a value. The trick is that internally it converts the key into an array index via a **hash function**, so reads and writes are O(1) instead of O(n).

```
   key                hash function          internal array
  ─────                ─────────────         ───────────────
 "alice"  ──────►   hash("alice") = 3 ─────► slot 3: ─────► "alice → 95"
 "bob"    ──────►   hash("bob")   = 0 ─────► slot 0: ─────► "bob   → 88"
 "carol"  ──────►   hash("carol") = 5 ─────► slot 5: ─────► "carol → 92"
                                              slot 1: empty
                                              slot 2: empty
                                              slot 4: empty
                                              ...
```

The hash function turns the key into a bucket number. As long as the function is good (spreads keys evenly across buckets) and the table is sized right, lookups are effectively constant-time.

**Collisions** happen when two keys hash to the same bucket. The map handles this internally (usually by chaining — storing a small list at each bucket). You don't think about it. But it's why hashmap operations are *average* O(1), not *guaranteed* O(1).

---

## JS gives you two: Object and Map

```js
// Object — the old-school way
const ages = { alice: 30, bob: 25 };
ages.alice;         // 30
ages["bob"];        // 25
ages.carol = 28;    // add
delete ages.bob;    // remove
"alice" in ages;    // true
Object.keys(ages);  // ['alice', 'carol']

// Map — the modern, recommended way
const m = new Map();
m.set("alice", 30);
m.set("bob", 25);
m.get("alice");     // 30
m.has("bob");       // true
m.delete("bob");    // true
m.size;             // 1
for (const [k, v] of m) { ... }
```

### When to use which

| Use **Object** when | Use **Map** when |
|---|---|
| Keys are always known strings (config, JSON) | Keys come from user input or vary in type |
| You want JSON serialization for free | You need `.size` |
| You're describing a "record" (a thing with named fields) | You need guaranteed insertion order iteration |
| | Keys might be objects, numbers, etc — not just strings |
| | You're doing lots of add/remove |

For algo problems: **almost always reach for `Map`**. It's faster for large numbers of insertions/deletions, has cleaner semantics, and avoids the prototype-pollution footguns of plain objects (e.g., `obj.toString` exists even on `{}` because of the prototype chain).

---

## Operations and Big-O

| Operation | Big-O | Note |
|-----------|-------|------|
| `m.get(k)` | O(1) avg | Worst case O(n) with bad hash/many collisions |
| `m.set(k, v)` | O(1) avg | |
| `m.has(k)` | O(1) avg | |
| `m.delete(k)` | O(1) avg | |
| Iterating all entries | O(n) | |

The "average" caveat is real but in practice almost never matters. JS engine hash functions are good.

---

## The "use a hashmap to skip a nested loop" pattern

This is the trick for two sum. It's the most useful hashmap pattern.

**Naive two sum** — for each element, scan the rest of the array for a partner. O(n²).

```js
// O(n²) — slow
const twoSumNaive = (nums, target) => {
  for (let i = 0; i < nums.length; i++) {
    for (let j = i + 1; j < nums.length; j++) {
      if (nums[i] + nums[j] === target) return [i, j];
    }
  }
};
```

**Hashmap two sum** — for each element `x`, the partner we want is `target - x`. So as we walk the array, remember every number we've seen in a map keyed by value → index. At each step, check if the complement is already in the map.

```js
// O(n) — fast
const twoSum = (nums, target) => {
  const seen = new Map();
  for (let i = 0; i < nums.length; i++) {
    const need = target - nums[i];
    if (seen.has(need)) return [seen.get(need), i];
    seen.set(nums[i], i);
  }
};
```

We traded a tiny bit of memory (the map) for a giant time win. **This is the move:** any time you're tempted to write a nested loop, ask "can I precompute the inner thing into a hashmap?"

Same pattern shows up in: anagram grouping, contains-duplicate, valid-sudoku, "first non-repeating character," `groupBy`-style aggregations.

---

## Visualizing two sum running

Walking through `twoSum([2, 7, 11, 15], 9)`:

```
i=0  nums[i]=2   need = 9-2 = 7    seen = {}                ← 7 not in map
                                   seen ← {2: 0}

i=1  nums[i]=7   need = 9-7 = 2    seen = {2: 0}            ← 2 IS in map! at index 0
                                   return [0, 1]
```

Done in one pass.

---

## FAQ

**Q: Why use `Map` instead of `{}`?**
A few reasons:
1. **No prototype pollution.** `{}.toString` is a function. `new Map().get("toString")` is `undefined`. With plain objects you have to guard every lookup with `Object.hasOwn(obj, key)` to be safe. With Map you don't.
2. **Any key type.** Map keys can be objects, functions, etc. Object keys are coerced to strings (so `obj[1]` and `obj["1"]` are the same key).
3. **Iteration order is guaranteed insertion order.** Objects also preserve insertion order for string keys in modern JS, but integer-looking keys are a footgun.
4. **`.size`.** Objects need `Object.keys(obj).length`, which is O(n).

**Q: What is the hash function for strings?**
Implementation-defined but typically something like FNV or a variant: walk the chars, mix them into a running 32-bit integer with shifts and XORs. The goal is "small change in input → big change in output" so similar strings don't collide.

**Q: What happens if two keys hash to the same bucket?**
Collision. Standard handling is **chaining** — the bucket holds a small linked list (or array) of `(key, value)` pairs, and lookup compares actual keys. Some hashmaps use **open addressing** instead (probe to the next bucket). V8 uses a hybrid.

**Q: Can I use an object as a Map key in plain `{}`?**
No — it gets coerced via `.toString()`, which returns `"[object Object]"` for every object. So all objects collapse into one key. This is why `Map` exists.

**Q: When does the "average O(1)" promise break?**
If your keys are crafted to all hash to the same bucket (adversarial input → "hash flooding attack"). Modern engines randomize hash seeds to make this hard. In normal app code: doesn't happen.

**Q: What about Set?**
`Set` is a hashmap where you only care about the keys, not values. Use it for "have I seen this before?" / dedupe / membership tests. `new Set([1, 1, 2, 3, 3])` → Set of `{1, 2, 3}`.

**Q: Difference between Map and WeakMap?**
WeakMap keys must be objects, and the map doesn't keep them alive — if the object is garbage collected, the entry disappears. Useful for caching metadata on objects without leaking memory. Rare in app code.

---

## Discussion prompts

- "When would you choose `Object` over `Map` in modern JS?" (Config-shaped data; JSON serialization; describing a record. Otherwise `Map` is the default.)
- "Could you implement a Map using only arrays? How big would the internal array be?" (Yes — table size usually a prime or power of two; load factor ~0.7 before resizing. Resizing is the expensive operation that gets amortized.)
- "Why does `new Set(arr)` dedupe an array in one line?" (Because Set rejects duplicate keys. `[...new Set(arr)]` is the canonical dedupe idiom.)

---

## Next up

→ [03-stacks-queues.md](03-stacks-queues.md) — two specialized array shapes you've already used without naming.
