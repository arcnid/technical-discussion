# 06 — Tries (Prefix Trees)

A tree where each **edge represents a character** and the path from the root to a node spells out a string. The name is from "re**trie**val" and is pronounced "try" (sometimes "tree," which gets confusing fast).

This is the data structure behind autocomplete, spell-check, IP routing tables, and a bunch of bioinformatics. Worth a 5-minute "oh THAT'S how it works" intro.

---

## Mental model

A hashmap finds a key in O(1) on average. So why do we need anything else?

Because hashmaps **can't do prefix queries.** If I want to know "all strings in this set that start with `ca`", a hashmap is useless — I'd have to iterate over every key and string-match.

A trie organizes strings *by their prefixes*, so prefix queries become a tree walk.

```
Words inserted: cat, car, card, cart, can, dog

              ┌─────┐
              │ root│
              └──┬──┘
          ┌─────┴─────┐
          ▼           ▼
        ┌───┐       ┌───┐
        │ c │       │ d │
        └─┬─┘       └─┬─┘
        ┌─┴─┐         ▼
        ▼   ▼       ┌───┐
      ┌───┐ ┌───┐   │ o │
      │ a │ │ a │   └─┬─┘
      └─┬─┘ └─┬─┘     ▼
        │     │     ┌───┐
        ▼     ▼     │ g │ ✓ ("dog" ends)
      ┌───┐ ┌───┐   └───┘
      │ n │ │ r │ ✓ ("car" ends, has children for card/cart)
      └───┘ └─┬─┘
       ✓     ┌┴─┬┐
      "can" ┌▼┐ ┌▼┐
            │d│ │t│
            └─┘ └─┘
             ✓   ✓
           "card" "cart"
```

Actual structure of the same tree with the "is this an end-of-word" flag explicit:

```
root
├── c
│   ├── a
│   │   ├── n   (end: "can")
│   │   ├── r   (end: "car")
│   │   │   ├── d   (end: "card")
│   │   │   └── t   (end: "cart")
│   │   └── t   (end: "cat")
│   └── ...
└── d
    └── o
        └── g   (end: "dog")
```

**Each node holds:**
- A map from "next character" → child node.
- A boolean "is this the end of a word."

That's it. Two fields.

---

## Code: a basic trie

```js
class TrieNode {
  constructor() {
    this.children = new Map();   // char → TrieNode
    this.isEnd = false;
  }
}

class Trie {
  constructor() { this.root = new TrieNode(); }

  insert(word) {
    let node = this.root;
    for (const ch of word) {
      if (!node.children.has(ch)) {
        node.children.set(ch, new TrieNode());
      }
      node = node.children.get(ch);
    }
    node.isEnd = true;
  }

  // Is this exact word in the trie?
  contains(word) {
    const node = this._walk(word);
    return node !== null && node.isEnd;
  }

  // Is *any* word with this prefix in the trie?
  hasPrefix(prefix) {
    return this._walk(prefix) !== null;
  }

  // Walk to the node at the end of `s`, or return null if it falls off.
  _walk(s) {
    let node = this.root;
    for (const ch of s) {
      if (!node.children.has(ch)) return null;
      node = node.children.get(ch);
    }
    return node;
  }
}

const t = new Trie();
for (const w of ["cat", "car", "card", "cart", "can", "dog"]) t.insert(w);

t.contains("car");      // true
t.contains("ca");       // false  (not a complete word)
t.hasPrefix("ca");      // true
t.hasPrefix("cab");     // false
```

---

## Autocomplete in ~15 lines

The killer use case. Given a prefix, return all words in the trie that start with it.

```js
const autocomplete = (trie, prefix) => {
  const start = trie._walk(prefix);
  if (start === null) return [];
  const results = [];
  const walk = (node, path) => {
    if (node.isEnd) results.push(path);
    for (const [ch, child] of node.children) {
      walk(child, path + ch);
    }
  };
  walk(start, prefix);
  return results;
};

autocomplete(t, "ca");
// → ["can", "car", "card", "cart", "cat"]
```

This is essentially what a search box does when you type `ca` and it shows you suggestions. (Real ones add ranking by frequency, but the data structure is a trie.)

---

## Big-O

Let *L* = length of the word being inserted/searched, *N* = number of words in the trie.

| Operation | Big-O |
|-----------|-------|
| Insert a word | O(L) |
| Search exact word | O(L) |
| Prefix exists? | O(L) |
| Autocomplete (return all matches) | O(L + total characters in results) |

Notably: lookup time is independent of N. A trie with 10 words and a trie with 10 million words both answer "does 'ca' exist as a prefix?" in roughly the same time. That's the magic.

**Tradeoff:** memory. Tries duplicate the prefix structure across all words. For natural-language dictionaries you can compact with techniques like **radix trees** (squash chains of single-child nodes) or **DAWGs** (share suffixes too). For most app-level use cases the basic trie is fine.

---

## When to actually reach for a trie

- **Autocomplete / typeahead.** The canonical use.
- **Spell check.** Trie of dictionary, search with bounded edit distance.
- **IP routing.** Longest-prefix matching on IP addresses — that's a trie keyed by bits.
- **Bioinformatics.** Genome/protein sequence prefixes.
- **URL routing.** Some frameworks route requests by walking a trie keyed by path segments.

When **not** to: when you only need exact-match lookups and don't care about prefixes. A hashmap is faster (constant time, not O(L)) and uses less memory.

---

## FAQ

**Q: Why is `node.children` a `Map` instead of an object or array?**
For lessons, any will do. Real-world tradeoffs:
- **`Map`** — works for any character (Unicode, emoji), no prototype pollution. Recommended.
- **Object** — fine for ASCII keys, slightly faster lookup in some engines, but has the `Object.hasOwn` footgun for keys like `constructor`.
- **Array of size 26** — only works for lowercase English. Fastest lookup (direct index). Common in LeetCode solutions.

**Q: How does the trie know "car" is a complete word but "ca" isn't?**
The `isEnd` flag. Without it, you couldn't tell — every prefix walk reaches a node. The flag distinguishes "I'm a complete word that happens to also be a prefix of longer words" from "I'm just a prefix on the way to other words."

**Q: How is this different from a BST?**
BST organizes by value comparison: parent vs child by `<`/`>`. Trie organizes by prefix: each edge is a character. Searching a BST for "cat" does O(log n) value comparisons. Searching a trie for "cat" does exactly 3 single-character steps.

**Q: How much memory does a trie use?**
Worst case, O(total characters across all words). In practice often less because words share prefixes (cat, car, card, cart share `ca`). For really memory-sensitive applications, compressed variants (radix tree, DAWG) reduce this further.

**Q: I've heard of "radix tree" — same thing?**
Yes, with compression. A plain trie has a chain of single-child nodes for unique suffixes ("ca**rds**" has each letter as its own node). A radix tree squashes those chains into single edges labeled with strings. Linux kernel uses radix trees for some things. Same idea, just denser.

**Q: Suffix tree, suffix array — what?**
Different cousin. A suffix tree contains every suffix of a single string — used for substring queries on huge texts (bioinformatics, plagiarism detection). Construction is famously fiddly (Ukkonen's algorithm). Mentioned for name recognition; nobody implements one casually.

**Q: Could I do autocomplete with a sorted array + binary search?**
Yes, surprisingly well, if your data is static. Binary-search for the first matching prefix, then scan forward until the prefix stops matching. Lookup is O(L + log N + K) where K is the number of matches. Tries win when you're inserting/deleting frequently or when L >> log N.

---

## Discussion prompts

- "If your contact list autocompletes phone numbers as you type, what data structure?" (A trie keyed by digits 0-9. Each node has 10 possible children. Routers do something similar with IP bits.)
- "Why would Google not use a single giant trie for search-as-you-type?" (Scale, ranking, personalization, freshness. The trie idea is in there somewhere, but production search is dozens of systems stacked.)
- "How would you store *case-insensitive* words?" (Lowercase on insert and on lookup. Or use a `Map` keyed by lowercased char. Don't try to encode case in the trie itself — too much branching.)

---

## Next up

→ [07-graphs-bfs-dfs.md](07-graphs-bfs-dfs.md) — generalize the tree idea: what if nodes can have *any* connections, not just parent-child?
