# DS / Algo Week — Group Session Plan

A light week. One member out, so we're swapping the usual product build for a data-structures and algorithms session.

The plan is **exposure-driven**: name-drop a wide range of concepts, but only test on the basics. Nobody is solving "invert a binary tree" cold or implementing a red-black tree. Familiarity > mastery this week.

---

## Agenda (60 min, give or take)

| Block | Time | What |
|------:|-----:|------|
| 1 | 5 min  | Kickoff + Big-O refresher (one slide of intuition, no formal proofs) |
| 2 | 15 min | **Concept tour** — quick exposure to 6-ish data structures, ~3 min each |
| 3 | 10 min | **BST quiz** — verbal only, no coding (see `04-trees-bst.md`) |
| 4 | 15 min | **Palindrome** — code it together, two-pointer (see `PROBLEMS.md`) |
| 5 | 15 min | **Two Sum** — code it together, hashmap (see `PROBLEMS.md`) |

If we hit time, the bonus problems in `PROBLEMS.md` are graph/traversal flavored and good for a second session.

---

## Concept tour — what we'll cover

Each file follows the same shape: diagram, mental model, JS code, Big-O, FAQ.

| File | Topic | Quizzed? |
|------|-------|----------|
| [01-arrays.md](01-arrays.md) | Arrays, two-pointer | ✅ palindrome |
| [02-hashmaps-objects.md](02-hashmaps-objects.md) | Objects, `Map`, hash collisions | ✅ two sum |
| [03-stacks-queues.md](03-stacks-queues.md) | LIFO/FIFO, push/pop/shift | exposure |
| [04-trees-bst.md](04-trees-bst.md) | Binary search trees, invariant | ✅ verbal quiz |
| [05-tree-traversal.md](05-tree-traversal.md) | Pre/in/post-order, level-order | exposure |
| [06-tries.md](06-tries.md) | Prefix trees, autocomplete | exposure |
| [07-graphs-bfs-dfs.md](07-graphs-bfs-dfs.md) | Adjacency list/matrix, BFS, DFS | exposure |
| [PROBLEMS.md](PROBLEMS.md) | Palindrome, two sum, BST quiz, bonuses | yes |

---

## What "exposure" means here

Exposure = "you've heard the name once, you can recognize the shape." That's it.

When someone later runs into a trie in a real codebase, the goal is for them to think *"oh, I've heard of those, it's the prefix-tree thing"* instead of *"???"*. That's a real win even if they can't implement it from scratch.

Mastery = "you can implement it on a whiteboard." We're only chasing that for arrays + hashmaps this week.

---

## Why these picks

- **Arrays + hashmaps** — 90% of real-world LeetCode-style problems are one of these two. Worth nailing.
- **Stacks + queues** — required to talk about BFS/DFS without hand-waving.
- **Trees + BST** — the gateway drug to recursion. BST invariant is also a clean intuition test.
- **Traversal orders** — three words people hear constantly, never define. 5-min explainer permanently demystifies them.
- **Tries** — the "wait there are MORE trees?" moment. Pairs naturally with the hashmap discussion.
- **Graphs + BFS/DFS** — adjacency list is one of those representations everyone has seen but few have actually written out. Worth doing on the whiteboard.

---

## What we're **not** doing

- Red-black trees / AVL trees (we'll mention they exist, that's it)
- Dynamic programming (separate session)
- Sorting algorithms beyond mentioning "JS sort is Timsort, it's fine"
- Anything requiring more than ~15 lines of code to solve

---

## Ground rules

- **No AI assist during the problems.** Pair / talk it out / use the whiteboard.
- **Wrong is fine.** The point is the conversation, not the green checkmark.
- **Senior folks: hang back.** Let the less-experienced people drive the keyboard.
- **All code examples in JS** — matches the day-job stack.

---

## After the session

If anything from the concept tour sparked interest, the FAQs in each file have follow-up rabbit holes worth chasing. Tries → autocomplete UIs. BFS → shortest-path / web crawlers. BST → how databases index things.
