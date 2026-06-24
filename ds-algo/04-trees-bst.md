# 04 — Trees and BSTs

The first data structure on this list that's **not just a fancy array**. Trees are a family; binary search trees are one specific member of that family with a useful invariant.

---

## Mental model: tree

A tree is a bunch of **nodes** connected by directed edges, where:
- Exactly one node is the **root** (no parent)
- Every other node has **exactly one parent**
- No cycles (no node is its own ancestor)

```
                    ┌─────┐
                    │  8  │   ← root
                    └──┬──┘
              ┌────────┴────────┐
              ▼                 ▼
           ┌─────┐           ┌─────┐
           │  3  │           │ 10  │
           └──┬──┘           └──┬──┘
        ┌────┴────┐              └─────┐
        ▼         ▼                    ▼
     ┌─────┐  ┌─────┐               ┌─────┐
     │  1  │  │  6  │               │ 14  │   ← leaves
     └─────┘  └──┬──┘               └─────┘
              ┌──┴──┐
              ▼     ▼
            ┌─────┐ ┌─────┐
            │  4  │ │  7  │
            └─────┘ └─────┘
```

**Vocabulary you'll hear:**
- **Root** — top node. No parent.
- **Leaf** — bottom node. No children.
- **Internal node** — has at least one child.
- **Depth** of a node — distance from root (root is depth 0).
- **Height** of the tree — max depth of any leaf.
- **Subtree** — a node plus everything below it.
- **Binary tree** — every node has *at most* two children (left and right).

---

## The BST invariant

A **Binary Search Tree** is a binary tree where for *every* node:

> Everything in the **left** subtree is **less than** the node.
> Everything in the **right** subtree is **greater than** the node.

This single rule — applied recursively — gives you O(log n) search on a balanced tree.

```
            ┌─────┐
            │  8  │
            └──┬──┘
          ┌───┴───┐
        ┌─▼─┐   ┌─▼─┐
        │ 3 │   │10 │
        └─┬─┘   └─┬─┘
        ┌─┴─┐     └─┐
      ┌─▼─┐ ┌▼┐    ┌▼─┐
      │ 1 │ │6│    │14│
      └───┘ └─┘    └──┘
```

Walking through: is `7` in this tree?
- At 8: 7 < 8 → go left.
- At 3: 7 > 3 → go right.
- At 6: 7 > 6 → go right.
- No right child of 6 → 7 is not in the tree.

Three comparisons in a tree of 7 nodes. **Each step throws away half the remaining tree** — same logic as binary search on a sorted array.

### Why "search tree"

Because lookup, insertion, and deletion all use the BST property to do binary-search-like descent:
- Look at current node.
- If target is less → go left.
- If greater → go right.
- If equal → found it.

Recurse until you fall off the tree (not found) or hit the value.

---

## Code: the node and basic operations

```js
class TreeNode {
  constructor(value) {
    this.value = value;
    this.left = null;
    this.right = null;
  }
}

// Insert (returns the (possibly new) root)
const insert = (node, value) => {
  if (node === null) return new TreeNode(value);
  if (value < node.value) node.left  = insert(node.left,  value);
  else if (value > node.value) node.right = insert(node.right, value);
  // equal → already present, do nothing (or count duplicates)
  return node;
};

// Search
const contains = (node, value) => {
  if (node === null) return false;
  if (value === node.value) return true;
  return value < node.value
    ? contains(node.left,  value)
    : contains(node.right, value);
};

// Build the tree above
let root = null;
for (const v of [8, 3, 10, 1, 6, 14, 4, 7]) {
  root = insert(root, v);
}
contains(root, 7);   // true
contains(root, 5);   // false
```

The recursion mirrors the picture. Every call is one step deeper into the tree.

---

## Big-O

| Operation | Balanced | Worst case (unbalanced) |
|-----------|----------|-------------------------|
| search | O(log n) | O(n) |
| insert | O(log n) | O(n) |
| delete | O(log n) | O(n) |

**The big asterisk:** all of this assumes the tree is balanced (height ≈ log n). If you insert sorted values into a plain BST, you build a linked list and get O(n) everything:

```
insert order: 1, 2, 3, 4, 5

      ┌─────┐
      │  1  │
      └──┬──┘
         └─►┌─────┐
            │  2  │
            └──┬──┘
               └─►┌─────┐
                  │  3  │   ← this is a linked list with extra steps
                  └──┬──┘
                     └─►...
```

This is why **self-balancing BSTs** exist (red-black, AVL, splay) — they rotate nodes during insert/delete to keep the tree shallow. You don't need to implement these. You need to know they exist, and you need to know that real-world maps/sets (Java `TreeMap`, C++ `std::map`, most database indexes) use them under the hood.

---

## Bonus exposure: balanced-tree relatives

| Name | One-line summary |
|------|------------------|
| AVL tree | BST that re-balances aggressively. Heights of subtrees differ by ≤ 1. |
| Red-black tree | BST that re-balances looser than AVL. Most language standard libraries use this. |
| B-tree | Tree where nodes hold *many* keys, not just one. Database indexes use B-trees / B+ trees because each node fits on a disk page. |
| Heap | Tree where parent ≤ (or ≥) children. Used for priority queues. *Not* a BST. |
| Trie | Tree keyed by string-prefix instead of by value. See `06-tries.md`. |

We're not implementing any of these. The point is when someone says "we should use a B-tree here," you know what conversation you're in.

---

## FAQ

**Q: How do I tell if a tree is balanced?**
For every node, the heights of its left and right subtrees differ by at most 1 (definition varies slightly — AVL uses 1, red-black uses a more relaxed "twice as deep"). For a quick eyeball check: if it looks lopsided, it's lopsided.

**Q: What's the difference between a tree and a graph?**
A tree is a special case of a graph: connected, acyclic, with one node designated as root. Any tree is a graph; not every graph is a tree.

**Q: What's a "binary heap" — is that a BST?**
No. A heap is also a binary tree, but the invariant is different: parent ≤ children (min-heap) or parent ≥ children (max-heap). Siblings have no ordering between them. Heaps are great for "give me the min/max" but terrible for "is X in here?" — that's O(n).

**Q: How do you delete from a BST?**
Three cases by number of children:
1. Leaf → just remove it.
2. One child → splice the child up.
3. Two children → find the **in-order successor** (smallest value in the right subtree), copy its value into the current node, then delete the successor (which has 0 or 1 child).

Worth understanding conceptually; rarely worth implementing from scratch.

**Q: What does "in-order successor" mean?**
The node that would come immediately after the current node if you walked the tree in-order (sorted order). In a BST: smallest value in the right subtree. Concrete: in the tree above, the in-order successor of `3` is `4`.

**Q: Are BSTs used in production code?**
The data structure conceptually, yes — *everywhere* (any sorted map/set is one). The hand-rolled implementation, almost never. You use the language's `TreeMap`/`SortedDict`/`std::map`.

**Q: Are recursion and trees inseparable?**
Mostly, yes. You can convert any tree traversal to iterative using an explicit stack — but the recursive version reads almost identically to the mental model, so it's the default. Just watch the call stack on huge trees.

---

## Quiz (verbal, no coding)

Use these to drive the BST discussion during the session.

1. **What is the BST invariant?** ("For every node, left subtree < node < right subtree.")
2. **Why is search O(log n)?** ("Each step eliminates half the remaining tree.")
3. **What breaks the O(log n) guarantee?** ("Imbalance — worst case is sorted insertion → linked-list-shaped tree → O(n).")
4. **In the example tree, where would 5 be inserted?** Walk it together. (5 < 8 → left. 5 > 3 → right. 5 < 6 → left. 5 > 4 → right. Insert as right child of 4.)
5. **Is a heap a BST?** ("No — different invariant. Parent ≤ children, no left/right ordering.")
6. **Name two production uses of trees.** (Database indexes, filesystem hierarchies, DOM, JSON, abstract syntax trees, prefix-based routing tables…)

---

## Discussion prompts

- "If you needed a sorted map in JS today, what would you use?" (No built-in. Either an npm sorted-map / red-black tree package, or — for small sizes — `[...map.entries()].sort()`.)
- "If your BST is shaped like a linked list, can you fix it without rebuilding?" (Yes — rotations, the building block of all self-balancing BSTs. Implementation is fiddly.)
- "Filesystems are trees. Web URLs are trees. The DOM is a tree. Why does the same shape keep appearing?" (Hierarchical containment is the most common natural relationship — "this thing contains those things." Trees encode it directly.)

---

## Next up

→ [05-tree-traversal.md](05-tree-traversal.md) — three orderings everyone hears about, demystified.
