# 03 — Stacks and Queues

Two **access patterns** more than two data structures. Both are usually implemented on top of an array. The point isn't the structure — it's the discipline of *only* using certain operations.

---

## Stack — LIFO (last in, first out)

Picture a stack of plates. You add a plate on top, and the next plate you take off is the one you most recently added.

```
                push ─┐         ┌─ pop
                      ▼         ▲
                    ┌─────────────┐
              top → │     5       │
                    ├─────────────┤
                    │     4       │
                    ├─────────────┤
                    │     2       │
                    ├─────────────┤
                    │     9       │  ← bottom (oldest)
                    └─────────────┘
```

### Operations

| Op | Big-O | JS array equivalent |
|----|-------|---------------------|
| push (add to top) | O(1) | `arr.push(x)` |
| pop (remove top) | O(1) | `arr.pop()` |
| peek (look at top, don't remove) | O(1) | `arr[arr.length - 1]` |
| isEmpty | O(1) | `arr.length === 0` |

### Code

```js
const stack = [];
stack.push(9);     // [9]
stack.push(2);     // [9, 2]
stack.push(4);     // [9, 2, 4]
stack.push(5);     // [9, 2, 4, 5]
stack.pop();       // returns 5, stack = [9, 2, 4]
stack[stack.length - 1]; // 4  (peek)
```

That's it. **A stack is just an array where you've promised yourself you'll only use `push` and `pop`.**

### When you reach for a stack

- **Undo / redo.** Each action is pushed. Undo pops the last one.
- **Function call stack.** Every function call is "pushed" by the runtime; `return` pops.
- **Matching delimiters.** `({[]})` — push openers, pop and check on closers.
- **Depth-first search.** Either explicit stack, or implicit via recursion.
- **Expression evaluation.** Shunting yard, postfix parsing, etc.

---

## Queue — FIFO (first in, first out)

Picture a line at a coffee shop. First person in line is the first one served.

```
       enqueue                                      dequeue
       (add at back)                            (remove from front)
              │                                        ▲
              ▼                                        │
         ┌─────────────────────────────────────┐
         │   9   →   2   →   4   →   5         │
         └─────────────────────────────────────┘
          ↑                                 ↑
         front (oldest)                    back (newest)
```

### Operations

| Op | Big-O | Naive JS array equivalent |
|----|-------|---------------------------|
| enqueue (add at back) | O(1) | `arr.push(x)` |
| dequeue (remove from front) | **O(n)** ⚠ | `arr.shift()` — every element shifts left |
| peek front | O(1) | `arr[0]` |

### Watch out: `shift()` is O(n)

For tiny queues it doesn't matter. For BFS over a 100k-node graph, it absolutely does. Two ways to get O(1) dequeue:

**Two-pointer / index trick.**
Use a regular array but keep a `head` index. Don't actually remove elements — just advance the pointer. Periodically compact.

```js
class Queue {
  constructor() { this.items = []; this.head = 0; }
  enqueue(x) { this.items.push(x); }
  dequeue() {
    if (this.head >= this.items.length) return undefined;
    const x = this.items[this.head];
    this.head++;
    if (this.head > 32 && this.head * 2 > this.items.length) {
      this.items = this.items.slice(this.head);
      this.head = 0;
    }
    return x;
  }
  get size() { return this.items.length - this.head; }
}
```

**Doubly-linked list.**
Each node has a `prev` and `next` pointer. Add/remove at either end is O(1). More allocation overhead but no compaction.

For algo-problem purposes: the index trick is fine. For library-grade code: there are npm packages.

### When you reach for a queue

- **Breadth-first search.** Visit nearest-first. Queue is the natural fit.
- **Job / task queues.** Things to process in arrival order.
- **Buffering.** Reader is faster than writer (or vice versa).
- **"Sliding window" problems.** Often deque (double-ended queue) territory.

---

## Bonus exposure: Deque (double-ended queue)

A queue where you can add or remove from **either end**. Useful for sliding-window maxima, palindrome checks on streams, etc. Not in the JS standard library — you build it on top of a doubly-linked list or two stacks. Mentioned here so the name doesn't surprise you.

---

## Bonus exposure: Priority queue / heap

A queue where dequeue gives you the **highest priority** item, not the oldest. Implemented as a **heap** — a tree where every parent ≥ (or ≤) its children. Operations are O(log n). Used in Dijkstra, A*, top-K problems. Also not in the JS standard library.

---

## Side-by-side comparison

```
STACK (LIFO)                    QUEUE (FIFO)

push ─┐    ┌─ pop                enqueue ─┐
      ▼    ▲                              ▼
    ┌─────────┐                  ┌────────────────────┐
    │   E     │ ← newest         │  A → B → C → D → E │
    ├─────────┤                  └────────────────────┘
    │   D     │                    ▲              ▲
    ├─────────┤                    │              │
    │   C     │                  dequeue       newest (enqueued last)
    ├─────────┤                  (oldest)
    │   B     │
    ├─────────┤
    │   A     │ ← oldest
    └─────────┘
```

Pop from a stack: get **E** (newest).
Dequeue from a queue: get **A** (oldest).

That's the whole difference.

---

## FAQ

**Q: Is a stack just an array?**
Operationally yes. Conceptually a stack is "an array where I've restricted myself to LIFO access." The restriction is the value — it makes the code's intent obvious and prevents accidental misuse.

**Q: Why is `shift` slow but `pop` fast?**
`pop` removes from the end — no other elements need to move. `shift` removes from the front, which means every remaining element's index decreases by 1, which means the runtime walks through and physically moves them. O(n) work.

**Q: Can I use a linked list as a queue?**
Yes — that's actually the textbook implementation. Doubly-linked list = O(1) enqueue at tail, O(1) dequeue at head. Tradeoff is more memory per element (two pointers) and worse cache locality.

**Q: Stack overflow — what does that actually mean?**
The **call stack** is a real stack (managed by the runtime, one per thread). Every function call pushes a frame containing local variables and the return address. Infinite recursion → unbounded pushes → out of stack space → "Maximum call stack size exceeded." Convert to an explicit while loop with an explicit stack array to bypass.

**Q: What's a "deque" pronounced like?**
"Deck." It's short for double-ended queue, not the Spanish "*de qué*." People say both.

**Q: Are stacks and queues only useful with arrays of numbers?**
No — they hold whatever you want. BFS queue holds nodes. Undo stack holds command objects. Browser history stack holds URLs. The structure is about *access pattern*, not element type.

---

## Discussion prompts

- "How would you implement a queue using two stacks?" (Classic interview question. Enqueue → push to stack A. Dequeue → if B is empty, pop everything from A and push to B, then pop from B. Each element moves at most twice → amortized O(1).)
- "Browser back/forward — stacks or queues?" (Two stacks. Back stack + forward stack. Visiting a new page clears the forward stack.)
- "Why is BFS a queue and DFS a stack?" (BFS visits nodes level-by-level — oldest unvisited first → FIFO. DFS goes deep before wide — newest unvisited first → LIFO.)

---

## Next up

→ [04-trees-bst.md](04-trees-bst.md) — recursion shows up next, and trees are where stacks become really natural.
