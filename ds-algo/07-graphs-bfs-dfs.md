# 07 — Graphs, BFS, DFS

A graph is the most general "things connected to things" data structure. Trees and linked lists are both special cases.

This file is mostly exposure: we'll define graphs, show two ways to represent them, and walk through BFS and DFS once each. We're not solving "rotting oranges" today. (We will mention it.)

---

## Mental model

A graph is **a set of nodes ("vertices") and a set of connections between them ("edges").**

```
     A ───── B
     │       │
     │       │
     C ───── D ───── E
              \     /
               \   /
                \ /
                 F
```

Vertices: `{A, B, C, D, E, F}`
Edges: `{(A,B), (A,C), (B,D), (C,D), (D,E), (D,F), (E,F)}`

**Kinds of graphs** you'll hear about:

| Term | Means |
|------|-------|
| **Undirected** | Edges go both ways. (Friendship.) |
| **Directed** | Edges have a direction. (Twitter follows.) |
| **Weighted** | Edges have a number attached. (Road distances.) |
| **Cyclic** | At least one cycle (path back to where you started). |
| **Acyclic** | No cycles. A DAG (Directed Acyclic Graph) is a directed graph with no cycles — task dependencies, build systems, git history. |
| **Connected** | You can get from any node to any other. |
| **Sparse** | Few edges relative to nodes. |
| **Dense** | Many edges (close to N²). |

A **tree** is just a connected acyclic undirected graph with one node designated as root.

---

## Representing a graph

You see two representations in the wild. Pick based on whether your graph is sparse or dense.

### Adjacency list — the usual choice

For each node, store a list of its neighbors.

```
A: [B, C]
B: [A, D]
C: [A, D]
D: [B, C, E, F]
E: [D, F]
F: [D, E]
```

In JS:

```js
const graph = {
  A: ["B", "C"],
  B: ["A", "D"],
  C: ["A", "D"],
  D: ["B", "C", "E", "F"],
  E: ["D", "F"],
  F: ["D", "E"],
};

// Or with Map for the same reasons we prefer Map over Object in general:
const graph = new Map([
  ["A", ["B", "C"]],
  ["B", ["A", "D"]],
  // ...
]);
```

**Memory:** O(V + E) — proportional to vertices + edges. Good for sparse graphs.
**"Are A and B connected?"** O(degree of A) — has to scan A's list.
**"What are A's neighbors?"** O(degree of A) — already a list.

This is the right default 90% of the time.

### Adjacency matrix — when the graph is dense or fixed-size

A V × V grid where `matrix[i][j]` is 1 (or the edge weight) if there's an edge from `i` to `j`.

```
       A  B  C  D  E  F
    A [ 0  1  1  0  0  0 ]
    B [ 1  0  0  1  0  0 ]
    C [ 1  0  0  1  0  0 ]
    D [ 0  1  1  0  1  1 ]
    E [ 0  0  0  1  0  1 ]
    F [ 0  0  0  1  1  0 ]
```

**Memory:** O(V²). Bad for sparse graphs (mostly zeros), fine for dense graphs.
**"Are A and B connected?"** O(1) — single array lookup.
**"What are A's neighbors?"** O(V) — scan a whole row.

Use it when V is bounded and small (board games, finite state machines, grid problems where the matrix is the grid itself).

### A third pattern: grid problems

A lot of LeetCode "graph" problems aren't given as adjacency lists at all — they're 2D grids where neighbors are the 4 (or 8) adjacent cells.

```
[ [0, 1, 0, 0],
  [0, 1, 0, 1],
  [0, 0, 0, 1],
  [1, 1, 0, 0] ]
```

Same algorithms apply. "Neighbors of cell (r, c)" = `[(r-1,c), (r+1,c), (r,c-1), (r,c+1)]` filtered to in-bounds.

---

## BFS — Breadth First Search

**Visit nodes by distance from the start.** Closest first.

The shape: dequeue a node, process it, enqueue its unvisited neighbors. Use a **queue**.

Use cases:
- **Shortest path in an unweighted graph.** (For weighted, you need Dijkstra.)
- "How many hops between two users?"
- "Rotting oranges" — at each minute, the rot spreads to neighboring fresh oranges. That's BFS layers.
- Crawling a website breadth-first.

```js
const bfs = (graph, start) => {
  const visited = new Set([start]);
  const queue = [start];
  let head = 0;
  const order = [];
  while (head < queue.length) {
    const node = queue[head++];
    order.push(node);
    for (const neighbor of graph[node]) {
      if (!visited.has(neighbor)) {
        visited.add(neighbor);
        queue.push(neighbor);
      }
    }
  }
  return order;
};

bfs(graph, "A");
// → ["A", "B", "C", "D", "E", "F"]
// (A at distance 0; B, C at distance 1; D at 2; E, F at 3)
```

The `visited` set is **mandatory**. Without it, a cycle in the graph makes the loop run forever.

### Tracking distance / shortest path

Slight tweak — store the depth in the queue, or use parallel arrays:

```js
const shortestPath = (graph, start, end) => {
  const visited = new Set([start]);
  const queue = [[start, 0]];
  let head = 0;
  while (head < queue.length) {
    const [node, dist] = queue[head++];
    if (node === end) return dist;
    for (const neighbor of graph[node]) {
      if (!visited.has(neighbor)) {
        visited.add(neighbor);
        queue.push([neighbor, dist + 1]);
      }
    }
  }
  return -1;
};
```

---

## DFS — Depth First Search

**Go deep first.** Pick a neighbor, follow it as far as you can, then backtrack.

Use a **stack** — either explicit, or implicit via recursion.

Use cases:
- "Is there a path from A to B?" (When you don't care about shortest.)
- Detecting cycles.
- Topological sort.
- Maze-solving (one path at a time).
- Connected-components / "count islands" / flood fill.

```js
// Recursive
const dfs = (graph, node, visited = new Set(), order = []) => {
  visited.add(node);
  order.push(node);
  for (const neighbor of graph[node]) {
    if (!visited.has(neighbor)) {
      dfs(graph, neighbor, visited, order);
    }
  }
  return order;
};

// Iterative (with explicit stack)
const dfsIter = (graph, start) => {
  const visited = new Set();
  const stack = [start];
  const order = [];
  while (stack.length) {
    const node = stack.pop();
    if (visited.has(node)) continue;
    visited.add(node);
    order.push(node);
    for (const neighbor of graph[node]) {
      if (!visited.has(neighbor)) stack.push(neighbor);
    }
  }
  return order;
};
```

### BFS vs DFS — the one-line difference

```
BFS:  pull from FRONT of collection (queue, FIFO) → wide before deep
DFS:  pull from BACK  of collection (stack, LIFO) → deep before wide
```

That's it. The traversal logic is otherwise identical: pop a node, mark visited, push neighbors. Swap one line → swap algorithms.

---

## Visualizing the difference

For the graph at the top of the file, starting from A:

**BFS order** (queue):
```
visit A          queue: [B, C]
visit B          queue: [C, D]
visit C          queue: [D]      (A already visited, not re-added)
visit D          queue: [E, F]
visit E          queue: [F]
visit F          queue: []
```
Order: `A, B, C, D, E, F`

**DFS order** (stack, picking the last-pushed neighbor):
```
visit A          stack: [B, C]
visit C          stack: [B, D]   (popped C)
visit D          stack: [B, B, E, F]
visit F          stack: [B, B, E, E]
visit E          stack: [B, B]   (E was reached via F)
visit B          stack: []
```
Order: `A, C, D, F, E, B`

Different orders, same nodes. Different use cases.

---

## Bonus exposure: weighted-graph algorithms

Briefly named so they don't sound foreign next time:

| Name | Solves | One-line summary |
|------|--------|------------------|
| Dijkstra | Shortest path, non-negative weights | BFS but with a min-heap instead of a queue |
| Bellman-Ford | Shortest path, allows negative weights | Slower than Dijkstra, can detect negative cycles |
| A* | Shortest path with a heuristic | Dijkstra + an estimate function. Game pathfinding. |
| Kruskal / Prim | Minimum spanning tree | "Cheapest way to connect every node" |
| Floyd-Warshall | All-pairs shortest path | O(V³). Brute and elegant. |
| Topological sort | Linear ordering of a DAG | Used by build systems, schedulers |

You will not implement these this week. But hearing the names once means they don't bounce off you later.

---

## FAQ

**Q: Adjacency list or matrix — how do I choose?**
Default to list. Switch to matrix if (a) the graph is dense (close to V² edges), (b) you need fast "is edge X-Y present?" queries, or (c) V is small and bounded.

**Q: Why does BFS need a `visited` set but tree traversal doesn't?**
Trees have no cycles by definition. Graphs can — without `visited`, you'd loop forever between A→B→A→B→.... Tree traversal can omit it because the only way back up is via `parent`, which the traversal doesn't follow.

**Q: Can DFS find the shortest path?**
On an unweighted graph: not naturally. DFS goes deep first, so the first time it reaches the target it might have taken a long detour. BFS visits by distance, so the first time it reaches the target, that's the shortest. For DFS you'd have to explore all paths and take the min — exponential blowup.

**Q: What's "topological sort"?**
For a DAG (directed acyclic graph), an ordering of nodes such that every edge goes from earlier to later in the ordering. Use cases: build order (compile dependencies before dependents), course prerequisites, deployment order. Algorithm: repeated DFS or repeated "remove nodes with no incoming edges."

**Q: Why does the queue version use `head++` instead of `shift()`?**
Same reason as in `03-stacks-queues.md` — `arr.shift()` is O(n) because every other element has to slide left by one. Index trick (track head, advance instead of removing) keeps everything O(1). For tiny graphs you won't notice. For graph problems with 100k+ nodes, you absolutely will.

**Q: BFS and the "shortest path in chess knight moves" problem — same?**
Yes. Treat each square as a node, "the 8 knight moves" as its edges. BFS from start square. First time you visit the target square, that's the minimum number of moves. Classic interview problem.

**Q: How do you DFS without recursion?**
Use an explicit stack array. Pop a node → process → push unvisited neighbors. The pattern is in the code above. Useful when graphs are huge enough that recursion would blow the call stack.

---

## Discussion prompts

- "If your graph has 1 million nodes and you want to find a shortest path, what would you use?" (BFS for unweighted. Dijkstra for weighted-with-non-negative. A* if you have a useful heuristic like geographic distance.)
- "Why is BFS the right choice for 'rotting oranges'?" (Each minute, rot spreads to direct neighbors — that's exactly one BFS layer. The answer is "how many BFS layers until everything's rotten.")
- "Is a doubly-linked list a graph?" (Yes — V nodes, 2(V-1) edges. Just a very thin one.)

---

## Done with the concept tour

→ [PROBLEMS.md](PROBLEMS.md) — palindrome, two sum, BST quiz, and a couple of bonus questions if we hit time.
