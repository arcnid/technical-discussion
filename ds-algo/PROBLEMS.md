# Problems

The problems we'll actually solve this session. Two main, one verbal quiz, two bonus.

Each problem has:
- The prompt
- Hints (in collapsible blocks — open only when stuck)
- The solution (in a collapsible block — open at the end together)
- A short discussion of "why this solution" so we're learning the *pattern*, not just the answer

**Rules of engagement:** Pair up, no AI assist, talk it out. Wrong attempts are good attempts — that's where the conversation lives.

---

## Problem 1: Valid Palindrome

**Prompt:** Given a string `s`, return `true` if it reads the same forwards and backwards. For this version, only consider alphanumeric characters and ignore case.

Examples:
- `"A man, a plan, a canal: Panama"` → `true`
- `"race a car"` → `false`
- `""` → `true`

Constraints: `s.length` up to 200k. So O(n²) might pass but O(n) is the goal.

### Hint 1
<details>
<summary>Click for hint</summary>

You don't actually have to *build* the cleaned string. Think about whether you can walk the original string and skip non-alphanumeric characters as you go.

</details>

### Hint 2
<details>
<summary>Click for hint</summary>

Two pointers, one at each end. Move both inward, comparing as you go. Skip over non-alphanumeric characters on either side.

</details>

### Hint 3
<details>
<summary>Click for hint</summary>

Helper function: `isAlphanumeric(ch)` — use a regex `/[a-z0-9]/i` or check char codes. Compare lowercased characters at each step.

</details>

### Solution
<details>
<summary>Click for solution</summary>

```js
const isPalindrome = (s) => {
  const isAlphanumeric = (ch) => /[a-z0-9]/i.test(ch);

  let l = 0;
  let r = s.length - 1;

  while (l < r) {
    while (l < r && !isAlphanumeric(s[l])) l++;
    while (l < r && !isAlphanumeric(s[r])) r--;
    if (s[l].toLowerCase() !== s[r].toLowerCase()) return false;
    l++;
    r--;
  }

  return true;
};
```

**Walkthrough on `"A man, a plan, a canal: Panama"`:**
- Pointers start at `A` and `a` (last char). Equal (case-insensitive). Move in.
- Next: at space and `m`. Space isn't alphanumeric → skip left. Now `m` and `m`. Equal.
- Continue inward, skipping commas, spaces, colons.
- Eventually pointers cross. Return `true`.

**Complexity:**
- Time: O(n). Each character visited at most once by each pointer.
- Space: O(1). No second string built.

**Why this and not "reverse and compare":**
```js
const naive = (s) => {
  const cleaned = s.toLowerCase().replace(/[^a-z0-9]/g, "");
  return cleaned === cleaned.split("").reverse().join("");
};
```
This also works and is honestly fine for an interview answer. The two-pointer version is the pattern though — same shape solves "remove duplicates from sorted array," "container with most water," "3sum." Worth knowing.

</details>

---

## Problem 2: Two Sum

**Prompt:** Given an array of integers `nums` and a target integer, return the **indices** of the two numbers that add up to `target`. Assume exactly one solution. You may not use the same element twice.

Examples:
- `nums = [2, 7, 11, 15], target = 9` → `[0, 1]` (because `2 + 7 = 9`)
- `nums = [3, 2, 4], target = 6` → `[1, 2]`
- `nums = [3, 3], target = 6` → `[0, 1]`

### Hint 1
<details>
<summary>Click for hint</summary>

The obvious nested-loop solution is O(n²). Can you do it in one pass?

</details>

### Hint 2
<details>
<summary>Click for hint</summary>

For each element `x` you encounter, what's the *partner* number you'd want to find? If you've already seen it, you're done.

</details>

### Hint 3
<details>
<summary>Click for hint</summary>

Use a `Map` from value → index. As you iterate, check if `target - nums[i]` is already in the map.

</details>

### Solution
<details>
<summary>Click for solution</summary>

```js
const twoSum = (nums, target) => {
  const seen = new Map();
  for (let i = 0; i < nums.length; i++) {
    const need = target - nums[i];
    if (seen.has(need)) return [seen.get(need), i];
    seen.set(nums[i], i);
  }
};
```

**Walkthrough on `[2, 7, 11, 15], target=9`:**
- i=0, nums[i]=2. need=7. seen={}. Not there. seen ← {2: 0}.
- i=1, nums[i]=7. need=2. seen={2:0}. Found! Return [0, 1].

**Complexity:**
- Time: O(n). One pass.
- Space: O(n) for the map.

**Why this pattern matters:**
The structure is "I'm about to look something up that depends on what I've already seen — let me cache the things I've seen in a hashmap as I go." It collapses a nested loop into a single pass.

Same shape solves:
- "Contains duplicate" (key = value, just check `has`)
- "First non-repeating character" (key = char, value = count)
- "Group anagrams" (key = sorted letters, value = list of strings)
- "Longest substring without repeating characters" (sliding window + hashmap)

</details>

---

## Quiz: Binary Search Trees (verbal, no coding)

No keyboards. Group discussion.

1. **State the BST invariant.**
   <details><summary>Answer</summary>
   For every node: all values in the left subtree are less than the node; all values in the right subtree are greater.
   </details>

2. **Why is BST search O(log n)?**
   <details><summary>Answer</summary>
   Each comparison eliminates half the remaining tree (you go left or right, never both). Same logic as binary search on a sorted array.
   </details>

3. **When does that O(log n) guarantee break?**
   <details><summary>Answer</summary>
   When the tree becomes unbalanced. Worst case: insert sorted values into a plain BST → it degrades to a linked list, and search is O(n). Self-balancing variants (red-black, AVL) prevent this by rotating during inserts.
   </details>

4. **Given this BST, where would you insert `5`?**

   ```
            8
           / \
          3  10
         / \   \
        1   6  14
           / \
          4   7
   ```
   <details><summary>Answer</summary>
   5 < 8 → go left. 5 > 3 → go right. 5 < 6 → go left. 5 > 4 → go right. 4 has no right child → insert 5 as right child of 4.
   </details>

5. **What's the result of an *inorder* traversal of this tree?**
   <details><summary>Answer</summary>
   1, 3, 4, 6, 7, 8, 10, 14. Inorder on a BST always yields sorted order. That's the killer feature of inorder.
   </details>

6. **Is a heap a BST?**
   <details><summary>Answer</summary>
   No. A heap is also a binary tree, but the invariant is "parent ≤ children" (min-heap) or "parent ≥ children" (max-heap). Siblings have no defined ordering, so you can't search a heap in O(log n) — only the root is special.
   </details>

7. **Name two places BSTs (or BST variants) are used in production.**
   <details><summary>Answer</summary>
   Many right answers: database indexes (B-tree / B+ tree variants), language standard-library sorted maps (Java `TreeMap`, C++ `std::map` — both red-black), in-memory ordered sets, range queries, scheduler priority structures.
   </details>

---

## Bonus 1: Maximum Depth of Binary Tree

Use this if we hit time after the main problems. Pure recursion warm-up.

**Prompt:** Given the root of a binary tree, return its maximum depth (number of nodes along the longest path from root to a leaf).

```
    3
   / \
  9  20
     / \
    15  7
```
→ depth = 3.

### Hint
<details>
<summary>Hint</summary>

Each subtree's depth is `1 + max(depth(left), depth(right))`. Empty tree → 0.

</details>

### Solution
<details>
<summary>Solution</summary>

```js
const maxDepth = (root) => {
  if (root === null) return 0;
  return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
};
```

Three lines. The whole shape of "tree recursion": base case for null, combine results from children.

**Complexity:** O(n) time (every node visited once), O(h) space (recursion stack proportional to height — log n for balanced, n for worst case).

</details>

---

## Bonus 2: Number of Islands

Classic graph / grid problem. Only attempt if everyone wants more.

**Prompt:** Given a 2D grid of `'1'` (land) and `'0'` (water), return the number of islands. An island is a group of land cells connected horizontally or vertically.

```
[ [1, 1, 0, 0, 0],
  [1, 1, 0, 0, 0],
  [0, 0, 1, 0, 0],
  [0, 0, 0, 1, 1] ]
```
→ 3 islands.

### Hint 1
<details>
<summary>Hint</summary>

Walk every cell. When you hit unvisited land, increment your counter, then BFS or DFS to mark every land cell in that island as visited.

</details>

### Hint 2
<details>
<summary>Hint</summary>

You can "mark visited" by mutating the grid (turn `'1'` into `'0'`). Saves a separate visited set.

</details>

### Solution
<details>
<summary>Solution</summary>

```js
const numIslands = (grid) => {
  if (grid.length === 0) return 0;
  const rows = grid.length;
  const cols = grid[0].length;
  let count = 0;

  const dfs = (r, c) => {
    if (r < 0 || c < 0 || r >= rows || c >= cols) return;
    if (grid[r][c] !== "1") return;
    grid[r][c] = "0";    // mark visited by sinking the land
    dfs(r + 1, c);
    dfs(r - 1, c);
    dfs(r, c + 1);
    dfs(r, c - 1);
  };

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (grid[r][c] === "1") {
        count++;
        dfs(r, c);
      }
    }
  }
  return count;
};
```

**Walkthrough:** Outer loop finds the first cell of each island (any `'1'` we haven't sunk yet). Inner DFS sinks the rest of that island so we don't recount it.

**Complexity:**
- Time: O(rows × cols). Each cell visited at most twice (once by outer loop, once by DFS).
- Space: O(rows × cols) worst-case recursion stack (if the whole grid is one snake-shaped island).

**Pattern note:** "Walk every cell, BFS/DFS from unvisited land" generalizes to flood-fill, paint-bucket tools, connected-components in any graph, and a dozen other LeetCode problems. The naming is grid-specific; the shape is graph-general.

</details>

---

## Wrap-up discussion

After we finish, 5 min of "what stuck":
- Which pattern was new for you?
- Which one of these had you not heard of at all?
- What's a problem at work where one of these might actually apply?

We're not testing recall — we're testing whether the names stop being scary. Goal achieved if "BFS" and "trie" feel like vocab you've handled rather than mystery acronyms.
