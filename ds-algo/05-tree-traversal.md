# 05 — Tree Traversal: Pre / In / Post / Level Order

Four ways to **visit every node in a tree**. They all touch every node once (O(n)). The difference is **when you process the current node** relative to its children.

Once you've internalized this, every tree-walking algorithm will feel like a variation of one of these four templates.

---

## The example tree

We'll use the same tree for all four traversals so the difference is clear.

```
                    ┌─────┐
                    │  A  │
                    └──┬──┘
              ┌────────┴────────┐
              ▼                 ▼
           ┌─────┐           ┌─────┐
           │  B  │           │  C  │
           └──┬──┘           └──┬──┘
        ┌────┴────┐         ┌───┴───┐
        ▼         ▼         ▼       ▼
     ┌─────┐  ┌─────┐   ┌─────┐  ┌─────┐
     │  D  │  │  E  │   │  F  │  │  G  │
     └─────┘  └─────┘   └─────┘  └─────┘
```

---

## The three "depth-first" orders

All three are recursive and all three visit children depth-first. The only difference is *when* the current node is processed.

```
preorder:    process self  →  recurse left  →  recurse right
inorder:     recurse left  →  process self  →  recurse right
postorder:   recurse left  →  recurse right →  process self
```

That's it. Three words = three orders.

### Preorder — A, B, D, E, C, F, G

"Process self first, then descend." Reads like a top-down outline.

```
                    1. A
                       ├── 2. B
                       │   ├── 3. D
                       │   └── 4. E
                       └── 5. C
                           ├── 6. F
                           └── 7. G
```

**When you'd use it:** copying a tree (you need the parent before its children), serializing a tree to a flat list (a → b → c → ...), generating a directory listing.

```js
const preorder = (node, out = []) => {
  if (node === null) return out;
  out.push(node.value);    // process
  preorder(node.left, out);
  preorder(node.right, out);
  return out;
};
```

### Inorder — D, B, E, A, F, C, G

"Recurse left, process, recurse right." On a BST this gives you **sorted order** — that's the killer feature.

If we used the BST from the previous file (8, 3, 10, 1, 6, 14, 4, 7), inorder would yield 1, 3, 4, 6, 7, 8, 10, 14.

**When you'd use it:** getting sorted output from a BST, range queries, in-order successor lookups.

```js
const inorder = (node, out = []) => {
  if (node === null) return out;
  inorder(node.left, out);
  out.push(node.value);    // process
  inorder(node.right, out);
  return out;
};
```

### Postorder — D, E, B, F, G, C, A

"Recurse first, process last." You only process a node once you know everything about its children.

```
                    7. A
                       ├── 3. B
                       │   ├── 1. D
                       │   └── 2. E
                       └── 6. C
                           ├── 4. F
                           └── 5. G
```

**When you'd use it:** deleting a tree (children before parent so we don't lose references), evaluating expression trees (`3 + 4 * 2` → evaluate `4 * 2` before the `+`), computing the size/height of a tree, anything where the parent's value depends on its children's results.

```js
const postorder = (node, out = []) => {
  if (node === null) return out;
  postorder(node.left, out);
  postorder(node.right, out);
  out.push(node.value);    // process
  return out;
};
```

---

## The fourth: level-order (BFS)

Visit by **depth**: all depth-0 nodes, then all depth-1 nodes, then all depth-2 nodes, etc.

```
        ┌─────┐
   0:   │  A  │                          Order: A, B, C, D, E, F, G
        └─────┘
        ┌─────┐  ┌─────┐
   1:   │  B  │  │  C  │
        └─────┘  └─────┘
        ┌───┐ ┌───┐ ┌───┐ ┌───┐
   2:   │ D │ │ E │ │ F │ │ G │
        └───┘ └───┘ └───┘ └───┘
```

Not depth-first → not recursive (or at least, not naturally). You use a **queue**: enqueue the root, then loop: dequeue → process → enqueue children.

```js
const levelOrder = (root) => {
  if (root === null) return [];
  const out = [];
  const queue = [root];
  let head = 0;            // index-trick queue (avoid O(n) shift)
  while (head < queue.length) {
    const node = queue[head++];
    out.push(node.value);
    if (node.left)  queue.push(node.left);
    if (node.right) queue.push(node.right);
  }
  return out;
};
```

**When you'd use it:** "find the shortest path in an unweighted graph/tree," BFS on a graph (this is exactly the algorithm — trees are just graphs with no cycles), printing a tree level-by-level for visualization.

---

## All four side-by-side

For the same tree (A as root, B/C as children of A, D/E as children of B, F/G as children of C):

| Order | Sequence |
|-------|----------|
| Preorder | A B D E C F G |
| Inorder | D B E A F C G |
| Postorder | D E B F G C A |
| Level-order | A B C D E F G |

Look at where `A` appears in each. That's the mnemonic:
- **Pre**order: A is *first* (root first).
- **In**order: A is *in the middle*.
- **Post**order: A is *last*.
- **Level**order: A is *first by depth*.

---

## Iterative versions (using an explicit stack)

Recursion is fine for sane-depth trees but blows the call stack on huge trees. Iterative preorder for reference:

```js
const preorderIter = (root) => {
  if (root === null) return [];
  const out = [];
  const stack = [root];
  while (stack.length) {
    const node = stack.pop();
    out.push(node.value);
    // push right first so left pops first
    if (node.right) stack.push(node.right);
    if (node.left)  stack.push(node.left);
  }
  return out;
};
```

Notice: this is BFS code (level-order) but with a **stack** instead of a **queue**. Swap stack/queue → swap DFS/BFS. That's the relationship in one sentence.

---

## FAQ

**Q: How do I remember which is which?**
The middle word tells you when the parent is processed:
- **Pre**: parent **before** children
- **In**: parent **between** children
- **Post**: parent **after** children

**Q: Why does inorder traversal of a BST yield sorted output?**
Because of the BST invariant. Left subtree is strictly smaller, right strictly larger. "Left, then self, then right" → smallest values, then current, then larger values. Recursively, this expands to fully sorted order.

**Q: Can I do inorder on a non-binary tree?**
Not naturally — "in" between which children? It's specifically a binary-tree thing. Preorder and postorder generalize fine: process self, then recurse over all children (preorder) or vice versa (postorder).

**Q: Is BFS the same as level-order traversal?**
Yes for trees. BFS = "visit nearest first," and in a tree all nodes at depth N are equally near the root, so BFS naturally proceeds level-by-level. On a graph BFS is the general algorithm; on a tree it specializes to level-order.

**Q: Is DFS the same as preorder?**
Close. DFS = "go deep before wide," which describes all three of preorder/inorder/postorder. They differ only in *when* the current node is processed during the descent. Most often "DFS" implicitly means preorder.

**Q: When do I need an explicit stack vs recursion?**
Whenever the tree is too deep for the call stack. JS typically allows 10k-ish nested calls. For deep tries, parse trees, or skewed BSTs, convert to iterative. For most app code: recursion is fine.

**Q: Postorder and "evaluating an expression tree" — what's that?**
If you parse `(3 + 4) * 2` into a tree:

```
        ┌───┐
        │ * │
        └─┬─┘
      ┌───┴───┐
    ┌─▼─┐   ┌─▼─┐
    │ + │   │ 2 │
    └─┬─┘   └───┘
   ┌──┴──┐
 ┌─▼─┐ ┌─▼─┐
 │ 3 │ │ 4 │
 └───┘ └───┘
```

Postorder traversal visits `3, 4, +, 2, *`. That's reverse Polish notation, which a stack can evaluate directly: push 3, push 4, on `+` pop two and push 7, push 2, on `*` pop two and push 14. Done.

---

## Discussion prompts

- "If you serialize a BST using preorder, can you reconstruct it?" (Yes — the order encodes the structure. For non-BSTs you need preorder + inorder both, or you need to encode nulls.)
- "Why does the iterative preorder push right before left?" (Because the stack is LIFO — last pushed pops first. To process left first, push it last.)
- "If I want to print a tree with indentation showing depth, which traversal?" (Preorder. Print self at current indent, then recurse with depth+1.)

---

## Next up

→ [06-tries.md](06-tries.md) — a tree where the structure encodes string prefixes instead of value order.
