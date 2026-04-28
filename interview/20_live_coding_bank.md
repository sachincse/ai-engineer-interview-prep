# Chapter 20 — Live Coding Bank

> **Why this chapter exists:** Avrioc is confirmed to use both a third-party timed coding test (HackerRank-style, ~60 minutes, DS&A) and a human-led ML/Python coding round in their technical pipeline. One previous candidate explicitly mentioned **rotate-an-array** as a problem they got. This chapter is a structured drill bank covering both surfaces. Every problem follows the same shape so you build a habit: state the problem, think aloud, give the brute force, give the optimal, code it, walk through a concrete input, talk complexity, anticipate follow-ups.

> **How to use this chapter:** Read each problem. Set a thirty-second timer and think through the approach in your head **before** looking at the solution. Then code it from scratch on paper or a whiteboard. Don't peek until you've stalled for at least two minutes. Do at least three problems from each major section. The goal is not memorization — it's training your hands to type the patterns and your mouth to narrate while you do.

---

## Part A — DS&A patterns (timed test, ~30–45 min, HackerRank/CodeSignal style)

These are the patterns that show up over and over on coding tests at AI infrastructure companies. They share a property: each one is a **reusable template**, not a one-off trick. Once you've internalized the sliding-window template, for example, you can apply it to a dozen variants.

---

### A.1 Rotate an array by k positions  (CONFIRMED asked at Avrioc)

**Problem.** Given an array `arr = [1, 2, 3, 4, 5, 6, 7]` and an integer `k = 3`, rotate the array to the right by `k` positions in place. Expected output: `[5, 6, 7, 1, 2, 3, 4]`.

**Think-aloud strategy.** "Right rotation by k means the last k elements come to the front. The naive approach is to rotate one position at a time, k times, but that's O(n·k) and too slow for large k. The slick approach is the three-reversal trick: reverse the whole array, then reverse the first k elements, then reverse the rest. It works because reversing twice in nested ranges effectively swaps the two halves while preserving internal order. I'll also normalize k with `k %= n` so that k larger than n still works correctly."

**Brute force (O(n·k) time, O(1) space).** Pop from the back, push to the front, repeat k times. Easy to understand, too slow for large inputs. Mention it, then move on.

**Optimal — three reversals (O(n) time, O(1) space).**

```python
def rotate(arr: list[int], k: int) -> None:
    """Rotate `arr` right by k positions in place."""
    n = len(arr)
    if n == 0:
        return
    k %= n  # handle k > n
    arr.reverse()                     # full reverse
    arr[:k] = reversed(arr[:k])       # reverse first k
    arr[k:] = reversed(arr[k:])       # reverse the rest
```

**Walkthrough on the concrete input `[1, 2, 3, 4, 5, 6, 7]`, k = 3.**

```
Step 0 (input):           [1, 2, 3, 4, 5, 6, 7]
Step 1 (reverse all):     [7, 6, 5, 4, 3, 2, 1]
Step 2 (reverse [0:3]):   [5, 6, 7, 4, 3, 2, 1]
Step 3 (reverse [3:]):    [5, 6, 7, 1, 2, 3, 4]   ← answer
```

**Complexity narrative.** Each reversal is O(length-of-slice). The three slices sum to n + k + (n-k) = 2n elements touched, so this is O(n) time. We mutate the array in place with no auxiliary array, so it's O(1) space. The naive cyclic-shift approach would have been O(n·k), which for k = 10^6 and n = 10^7 would be 10^13 operations — completely infeasible.

**Common whiteboard mistakes.**
1. Forgetting `k %= n`. If k = 10 and n = 7, you rotate too far and get wrong output.
2. Using `arr[:k] = arr[:k][::-1]` with sliced copy — works but allocates an extra list. The interviewer will dock you for "in place."
3. Using `arr[-k:] + arr[:-k]` (a one-liner) — produces correct output but allocates a fresh list of size n. Great if they ask for "any solution," wrong if they ask for in-place.

**Follow-up: "What if I asked you to rotate left by k?"** Either negate k (`k = -k % n`) and use the same right-rotate code, or rearrange the reversals: reverse first k, reverse the rest, reverse the whole thing.

**Follow-up: "Can you do it without modifying the input?"** Then return `arr[-k:] + arr[:-k]` — O(n) time and O(n) extra space.

---

### A.2 Two-sum

**Problem.** Given `nums = [2, 7, 11, 15]` and `target = 9`, return the indices of the two numbers that add up to the target. Output: `[0, 1]` because `nums[0] + nums[1] = 2 + 7 = 9`.

**Think-aloud strategy.** "Brute force is two nested loops, O(n²). The classic optimization is a hash map: for each element x, check if `target - x` is already in the map; if yes, we have our pair, otherwise store x with its index. One pass, O(n) time."

**Brute force (O(n²)).** Two nested loops, return the first pair that sums.

**Optimal — hash map (O(n) time, O(n) space).**

```python
def two_sum(nums: list[int], target: int) -> list[int]:
    seen: dict[int, int] = {}  # value -> index
    for i, x in enumerate(nums):
        need = target - x
        if need in seen:
            return [seen[need], i]
        seen[x] = i
    return []
```

**Walkthrough on `nums = [2, 7, 11, 15]`, target = 9.**

```
i=0, x=2:  need=7, seen={},          7 not in seen,  seen={2:0}
i=1, x=7:  need=2, seen={2:0},       2 IS in seen!   return [0, 1]
```

**Complexity.** O(n) time because we do one pass. O(n) space for the hash map in the worst case (the answer pair is the last two elements).

**Common mistakes.** Storing the index before checking — that allows `nums[i] + nums[i] = target` to falsely match a single element with itself. The check-then-store order matters.

**Follow-up: "What if the array is sorted?"** Two pointers from both ends. Move left pointer right when sum is too small, right pointer left when too big. O(1) space, no hash map.

**Follow-up: "What if there are multiple pairs and I want all of them?"** Sort, then two-pointer with a while-loop that skips duplicates after each found pair.

---

### A.3 Longest substring without repeating characters

**Problem.** Given `s = "abcabcbb"`, return the length of the longest substring with all unique characters. Output: `3` (which is "abc").

**Think-aloud strategy.** "This is the canonical sliding-window problem. I'll keep a window `[left, right]` and a hash map of last-seen index per character. As I slide right, if the character already appeared inside the current window, I jump `left` to one past its previous index. Track the max window size as I go."

**Brute force.** For every starting index, expand a substring while maintaining a set; O(n²).

**Optimal — sliding window (O(n) time, O(min(n, charset)) space).**

```python
def longest_unique_substring(s: str) -> int:
    last: dict[str, int] = {}
    left = 0
    best = 0
    for right, ch in enumerate(s):
        if ch in last and last[ch] >= left:
            left = last[ch] + 1
        last[ch] = right
        best = max(best, right - left + 1)
    return best
```

**Walkthrough on `s = "abcabcbb"`.**

```
right=0  ch='a'  last={}              left=0  last={a:0}              best=1
right=1  ch='b'  last={a:0}           left=0  last={a:0, b:1}         best=2
right=2  ch='c'  last={a:0, b:1}      left=0  last={a:0, b:1, c:2}    best=3
right=3  ch='a'  last[a]=0 >= 0       left=1  last={a:3, b:1, c:2}    best=3
right=4  ch='b'  last[b]=1 >= 1       left=2  last={a:3, b:4, c:2}    best=3
right=5  ch='c'  last[c]=2 >= 2       left=3  last={a:3, b:4, c:5}    best=3
right=6  ch='b'  last[b]=4 >= 3       left=5  last={a:3, b:6, c:5}    best=3
right=7  ch='b'  last[b]=6 >= 5       left=7  last={a:3, b:7, c:5}    best=3
```

Final answer: `3`.

**Complexity.** O(n) time — every character is visited once by `right` and at most once by `left`. O(min(n, |alphabet|)) space.

**Common mistakes.** Forgetting the `last[ch] >= left` guard — without it, a character that appeared *before* the window's left boundary would incorrectly contract the window. The mental model is: the map remembers ALL positions seen, not just positions inside the window.

**Follow-up: "Now with at most K distinct characters."** Same template; replace the "duplicate" condition with "distinct count > K," and contract from the left by decrementing a counter map.

**Follow-up: "Minimum window covering target string T."** Same template, expand right until window covers T, then contract left while still covering, track minimum.

---

### A.4 Merge intervals

**Problem.** Given `intervals = [[1,3], [2,6], [8,10], [15,18]]`, merge overlapping intervals. Output: `[[1,6], [8,10], [15,18]]`.

**Think-aloud strategy.** "Sort by start time. Then walk through and either extend the last accepted interval (if the current one overlaps) or push the current as a new interval."

**Optimal — sort + linear pass (O(n log n) time, O(1) extra space).**

```python
def merge(intervals: list[list[int]]) -> list[list[int]]:
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    out = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out
```

**Walkthrough on `[[1,3], [2,6], [8,10], [15,18]]`.**

```
After sort:        [[1,3], [2,6], [8,10], [15,18]]
out = [[1,3]]
start=2 end=6:     2 <= 3, merge → out = [[1,6]]
start=8 end=10:    8 > 6, append → out = [[1,6], [8,10]]
start=15 end=18:   15 > 10, append → out = [[1,6], [8,10], [15,18]]
```

**Complexity.** O(n log n) time dominated by the sort, O(n) space for the output (or O(1) extra if you mutate in place).

**Common mistakes.** Using `start < out[-1][1]` instead of `<=`; that incorrectly fails to merge `[1,3]` and `[3,5]`. The convention "intervals that touch at a single point merge" is the typical interpretation, but always confirm with the interviewer.

**Follow-up: "What if intervals come as a stream — you can't sort?"** Use a balanced BST or a sorted container (Python's `sortedcontainers.SortedList`); insertion is O(log n), and on each insert you check immediate neighbors for overlap.

---

### A.5 Top-K frequent elements (heap pattern)

**Problem.** Given `nums = [1, 1, 1, 2, 2, 3]` and `k = 2`, return the k most frequent elements. Output: `[1, 2]`.

**Think-aloud strategy.** "Count frequencies with `Counter`. Then a min-heap of size k: for each element, push, and if size exceeds k, pop the smallest. At the end the heap holds the top k. Or use the built-in `heapq.nlargest` which does it in one shot."

**Optimal — Counter + nlargest (O(n log k) time).**

```python
import heapq
from collections import Counter

def top_k_frequent(nums: list[int], k: int) -> list[int]:
    counts = Counter(nums)
    return heapq.nlargest(k, counts, key=counts.get)
```

**Walkthrough on `nums = [1,1,1,2,2,3]`, k = 2.**

```
counts = {1: 3, 2: 2, 3: 1}
heapq.nlargest(2, counts, key=counts.get) → [1, 2]
```

**Complexity narrative.** Building the counter is O(n). `heapq.nlargest(k, ...)` is O(n log k) — it maintains a heap of size k, doing log-k work per element. Faster than sorting at O(n log n) when k is small relative to n.

**Common mistakes.** Using `sorted(counts, key=counts.get, reverse=True)[:k]` — works, but it's O(n log n), and the interviewer wants O(n log k).

**Follow-up: "Can you do better than O(n log k)?"** Yes — bucket sort by frequency, since frequencies are bounded by n. Build buckets `freq_buckets[i] = [elements with frequency i]`, walk from high frequency down. O(n) time but O(n) space and only beats heap-based when k is large.

---

### A.6 Binary search — first position where condition flips

**Problem.** Given a monotonic predicate `ok(x)` (False for low x, True for high x), find the smallest x in `[lo, hi]` where `ok(x)` is True. This single template solves: first-bad-version, smallest-divisor, capacity-to-ship-within-D-days, square root, and many others.

**Think-aloud strategy.** "Binary search bugs come from off-by-one errors and infinite loops. Use this canonical loop with `lo < hi` (not `<=`), `mid = (lo + hi) // 2`, and update `lo = mid + 1` when ok(mid) is False. The invariant is that `lo` is always a candidate answer or out-of-bounds, and `hi` is always a known True. Memorize the shape once, reuse forever."

**Optimal template (O(log range) time).**

```python
def first_true(lo: int, hi: int, ok) -> int:
    """Return the smallest x in [lo, hi] where ok(x) is True. Assumes monotonic."""
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid           # mid might be the answer; keep it in range
        else:
            lo = mid + 1       # mid is not the answer; exclude it
    return lo
```

**Worked application — `sqrt(target)` floor.** Find the largest x such that `x*x <= target`. Equivalently, find the smallest x such that `(x+1)*(x+1) > target`, then return `x`. Or rephrase: `ok(x) = x*x > target`; first_true gives the smallest x where x² exceeds target, so answer is `first_true(...) - 1`.

**Common mistakes.** Using `lo <= hi` with the same updates causes infinite loops on the boundary. Using `mid + 1` for both branches is wrong (you might skip the answer). Always trace through `[lo=0, hi=1]` mentally to verify your version terminates.

**Follow-up: "What if the search space is real-valued (e.g., capacity in floats)?"** Loop a fixed number of times (like 100 iterations) or until `hi - lo < epsilon`. Don't compare `lo < hi` directly with floats.

---

### A.7 Binary tree level-order traversal (BFS)

**Problem.** Given a binary tree, return its level order traversal — `[[3], [9, 20], [15, 7]]` for the classic [3, [9], [20, [15], [7]]] tree.

**Think-aloud strategy.** "BFS with a queue. The trick to grouping by level is to capture the queue size at the start of each iteration — that size is exactly the number of nodes on the current level. Process that many, then move on."

**Optimal — BFS with level batching (O(n) time, O(n) space).**

```python
from collections import deque

def level_order(root):
    if not root:
        return []
    out, q = [], deque([root])
    while q:
        level = []
        for _ in range(len(q)):       # snapshot size now — it grows as we append children
            node = q.popleft()
            level.append(node.val)
            if node.left:  q.append(node.left)
            if node.right: q.append(node.right)
        out.append(level)
    return out
```

**Walkthrough on tree `3 → (9, 20 → (15, 7))`.**

```
q=[3]                   level=[3],  q after children=[9, 20]   out=[[3]]
q=[9, 20], size=2       level=[9, 20], q=[15, 7]                out=[[3], [9, 20]]
q=[15, 7], size=2       level=[15, 7], q=[]                     out=[[3], [9, 20], [15, 7]]
done
```

**Complexity.** O(n) time, O(n) space (queue can hold up to one full level, which can be n/2 nodes for a balanced tree).

**Common mistakes.** Using `for _ in range(len(q))` is critical — if you write `while q` inside, you'll process children as part of the current level and lose the level boundaries.

**Follow-up: "Zigzag order — alternate left-to-right and right-to-left per level."** Same code, but on odd-indexed levels reverse the `level` list before appending to `out`.

---

### A.8 LRU Cache  (commonly asked at AI/infra companies)

**Problem.** Implement an LRU cache with `get(key)` and `put(key, value)`, both O(1).

**Think-aloud strategy.** "Two operations need to be O(1): finding a key and reordering for recency. The textbook answer is doubly-linked-list plus hash map, but in Python `OrderedDict` already does this — the hash map plus internal linked list. `move_to_end` reorders, `popitem(last=False)` evicts the LRU."

**Optimal — OrderedDict (O(1) get/put).**

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.d: OrderedDict = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.d:
            return -1
        self.d.move_to_end(key)        # mark as most recently used
        return self.d[key]

    def put(self, key: int, value: int) -> None:
        if key in self.d:
            self.d.move_to_end(key)
        self.d[key] = value
        if len(self.d) > self.cap:
            self.d.popitem(last=False)  # evict least recently used
```

**Walkthrough.** Capacity = 2.

```
put(1, A):   {1:A}
put(2, B):   {1:A, 2:B}
get(1):      → A, order becomes {2:B, 1:A}
put(3, C):   {2:B, 1:A, 3:C} → over capacity → evict 2 → {1:A, 3:C}
get(2):      → -1 (evicted)
get(3):      → C
```

**Why this is interview-relevant.** KV-cache eviction in LLM serving is a real-world LRU problem; vLLM's PagedAttention scheduler evicts least-recently-used blocks under memory pressure. Mention this connection unprompted — it shows you've thought about the production analog.

**Common mistakes.** Forgetting to `move_to_end` on `get` (so reads don't refresh recency). Forgetting to `move_to_end` when overwriting an existing key on `put`.

**Follow-up: "Make it thread-safe."** Wrap each method in a `threading.Lock`. Be careful that the lock covers the entire compound operation (read + reorder), not just the read.

**Follow-up: "Implement it without OrderedDict, just a hash map and a doubly-linked list."** That's the classic from-scratch version — be ready to write `Node` with prev/next pointers, a `head` and `tail` sentinel, and `_remove(node)` / `_add_to_front(node)` helpers. Twenty lines, but it's a different muscle.

---

### A.9 Producer-consumer with bounded queue

**Problem.** Implement a thread-safe bounded queue with multiple producers writing and multiple consumers reading. Producers should block when the queue is full; consumers should block when it's empty. Provide a clean shutdown mechanism.

**Think-aloud strategy.** "This is bread-and-butter for batched-inference servers. The cleanest way in Python is `queue.Queue` (which is thread-safe) plus a `threading.Event` for graceful shutdown. The queue handles all the locking internally; we just orchestrate."

**Optimal pattern.**

```python
import threading, queue, time

q: queue.Queue[int] = queue.Queue(maxsize=10)
stop = threading.Event()

def producer():
    i = 0
    while not stop.is_set():
        q.put(i, timeout=0.5)         # blocks if full, raises queue.Full on timeout
        i += 1

def consumer():
    while not stop.is_set() or not q.empty():
        try:
            item = q.get(timeout=0.1)
        except queue.Empty:
            continue
        # process item
        q.task_done()
```

**Why this matters in ML serving.** Most batched-inference servers (Triton, vLLM, custom batchers) follow this pattern internally: HTTP handlers are producers pushing requests onto a queue; a batch-builder thread is the consumer pulling up to N requests and running a forward pass. Be ready to explain the choice between `queue.Queue` (threads) and `asyncio.Queue` (coroutines) — they look similar but the underlying concurrency model is different.

**Common mistakes.** Forgetting `task_done()` if you ever call `q.join()`. Setting an unbounded queue (no `maxsize`), which lets producers run away and exhaust memory.

**Follow-up: "What about asyncio?"** Use `asyncio.Queue` and `await q.put(...)` / `await q.get()`. The control flow is identical but uses the event loop instead of OS threads.

---

### A.10 Valid balanced parentheses

**Problem.** Given a string of `()[]{}`, return True if balanced and properly nested.

**Think-aloud strategy.** "Stack. Push opens; on close, peek the stack — must match the corresponding open. Empty at the end means balanced."

**Optimal (O(n) time, O(n) space).**

```python
def is_valid(s: str) -> bool:
    pairs = {')': '(', ']': '[', '}': '{'}
    stack: list[str] = []
    for ch in s:
        if ch in '([{':
            stack.append(ch)
        elif ch in ')]}':
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
        # ignore other characters if mixed input
    return not stack
```

**Walkthrough on `s = "([{}])"`.**

```
ch='(': push → stack=['(']
ch='[': push → stack=['(', '[']
ch='{': push → stack=['(', '[', '{']
ch='}': matches '{' → pop → stack=['(', '[']
ch=']': matches '[' → pop → stack=['(']
ch=')': matches '(' → pop → stack=[]
end: stack empty → True
```

**Common mistakes.** Forgetting to check `if not stack` before peeking — `stack[-1]` on empty list throws IndexError, which is wrong behavior (should return False).

---

### A.11 Reverse a linked list

**Problem.** Given the head of a singly-linked list, reverse it and return the new head.

**Think-aloud strategy.** "Three pointers: previous (initially None), current (starts at head), and next (lookahead). At each step: save next, point current's next to prev, advance prev and current."

**Optimal (O(n) time, O(1) space).**

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_list(head: ListNode | None) -> ListNode | None:
    prev, curr = None, head
    while curr:
        nxt = curr.next        # save lookahead
        curr.next = prev       # reverse pointer
        prev = curr            # advance prev
        curr = nxt             # advance curr
    return prev
```

**Walkthrough on `1 → 2 → 3 → None`.**

```
curr=1, prev=None: nxt=2, 1.next=None, prev=1, curr=2
curr=2, prev=1:    nxt=3, 2.next=1,    prev=2, curr=3
curr=3, prev=2:    nxt=None, 3.next=2, prev=3, curr=None
loop ends, return prev=3 → 3 → 2 → 1 → None
```

**Common mistakes.** Forgetting to save `nxt` before mutating `curr.next` — you lose the rest of the list. Returning `head` instead of `prev` — head is now the tail.

**Follow-up: "Recursive version?"** Yes — `def rev(node): if not node or not node.next: return node; new_head = rev(node.next); node.next.next = node; node.next = None; return new_head`. O(n) stack space.

---

### A.12 Maximum subarray sum (Kadane's algorithm)

**Problem.** Given `nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]`, find the contiguous subarray with the largest sum. Output: `6` (from `[4, -1, 2, 1]`).

**Think-aloud strategy.** "Classic DP in disguise. At each position, decide: extend the previous subarray, or start fresh from here. Take the max. Track the global best as we go."

**Optimal — Kadane (O(n) time, O(1) space).**

```python
def max_subarray(nums: list[int]) -> int:
    cur = best = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)     # extend or restart
        best = max(best, cur)
    return best
```

**Walkthrough on `[-2, 1, -3, 4, -1, 2, 1, -5, 4]`.**

```
x=-2  cur=-2  best=-2
x= 1  cur=max(1, -2+1)=1   best=1
x=-3  cur=max(-3, 1-3)=-2  best=1
x= 4  cur=max(4, -2+4)=4   best=4
x=-1  cur=max(-1, 4-1)=3   best=4
x= 2  cur=max(2, 3+2)=5    best=5
x= 1  cur=max(1, 5+1)=6    best=6
x=-5  cur=max(-5, 6-5)=1   best=6
x= 4  cur=max(4, 1+4)=5    best=6
return 6
```

**Common mistakes.** Initializing `cur = best = 0` — wrong if the array is all negative. Always initialize with `nums[0]`.

**Follow-up: "Return the indices of the subarray."** Track `start` and `end` alongside `cur`; reset `start` whenever `cur` resets to `x` instead of extending.

---

### A.13 Number of islands (DFS on a grid)

**Problem.** Given a 2D grid of '1's (land) and '0's (water), count the number of islands (connected groups of land).

**Think-aloud strategy.** "Walk every cell. When I find a '1', it's a new island — increment the count and DFS-flood-fill all connected '1's, marking them visited (turn them to '0' or use a separate visited set)."

**Optimal — DFS (O(rows × cols) time and space).**

```python
def num_islands(grid: list[list[str]]) -> int:
    if not grid or not grid[0]:
        return 0
    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r: int, c: int) -> None:
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != '1':
            return
        grid[r][c] = '0'              # mark visited
        dfs(r+1, c); dfs(r-1, c)
        dfs(r, c+1); dfs(r, c-1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)
    return count
```

**Common mistakes.** Recursion depth — a giant island can blow Python's default recursion limit (1000). For large grids, switch to an iterative BFS with a deque.

**Follow-up: "What if I want the largest island, not the count?"** DFS returns the size of the connected component; track the max.

---

## Part B — ML / Python coding (human-led round)

These are problems a senior engineer would actually ask in a 60-minute round. Each tests both your ML knowledge and your Python production sense.

---

### B.1 Scaled dot-product attention from scratch  (CLASSIC, expect this)

**Problem.** Implement scaled dot-product attention. Input shapes: Q is `(B, H, T_q, d_k)`, K and V have similar shapes. Optional mask. Return `(B, H, T_q, d_v)`.

**Think-aloud strategy.** "Three core steps: scaled dot product to get scores, softmax along the key dimension to get attention weights, then weight V. The scaling factor is `1/sqrt(d_k)`, which keeps the variance of dot products bounded so softmax doesn't saturate. Mask is applied before softmax, replacing masked positions with `-inf` so they get zero probability."

**Implementation.**

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(
    Q: torch.Tensor,                          # (B, H, T_q, d_k)
    K: torch.Tensor,                          # (B, H, T_k, d_k)
    V: torch.Tensor,                          # (B, H, T_k, d_v)
    mask: torch.Tensor | None = None,         # (B, 1, T_q, T_k); 1 = keep, 0 = mask out
    dropout_p: float = 0.0,
) -> torch.Tensor:
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)    # (B, H, T_q, T_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    if dropout_p > 0:
        attn = F.dropout(attn, p=dropout_p)
    return attn @ V                                     # (B, H, T_q, d_v)
```

**Walkthrough on a tiny example.** Let B=1, H=1, T_q=T_k=2, d_k=d_v=2. Q = K = V = `[[1, 0], [0, 1]]`.

```
scores = Q @ K.T = [[1, 0], [0, 1]] / sqrt(2) ≈ [[0.71, 0], [0, 0.71]]
softmax(row 0) = softmax([0.71, 0]) ≈ [0.67, 0.33]
softmax(row 1) = softmax([0, 0.71]) ≈ [0.33, 0.67]
attn @ V = [[0.67, 0.33], [0.33, 0.67]]
```

**Anticipated follow-ups while you write the code.**

- **"Why divide by sqrt(d_k)?"** Without scaling, dot products grow as `O(d_k)`, softmax saturates with one huge value, and gradients vanish. Scaling keeps the variance approximately one. Intuitively: if Q and K have unit-variance components, `Q·K = sum of d_k products`, so variance scales as d_k, std as sqrt(d_k).
- **"Where does causal masking fit?"** Build a triangular mask `torch.tril(torch.ones(T, T))` and broadcast it through the `mask` argument. Anything above the diagonal is zeroed out, so token i can only attend to tokens 0..i.
- **"What is FlashAttention doing differently?"** Same math, but tiles Q/K/V into SRAM-resident blocks, computes softmax block-wise via the online softmax trick, and recomputes the attention matrix during backward instead of storing it. Memory drops from O(N²) to O(N), and end-to-end speed goes up 2-4x because the bottleneck on H100s is HBM bandwidth, not FLOPs.

**Common mistakes.** Using `view` instead of `reshape` after a transpose (transposed tensors are non-contiguous; `view` errors). Forgetting the scaling factor entirely. Applying dropout *before* softmax instead of after.

---

### B.2 Multi-head attention

**Problem.** Implement multi-head attention as a `nn.Module`. Constructor takes `d_model` and `n_heads`; forward takes `x` of shape `(B, T, d_model)` plus optional mask.

**Think-aloud strategy.** "Multi-head splits the d_model dimension into n_heads × d_k. The clean way to do this is one big linear `qkv = Linear(d_model, 3*d_model)`, then reshape to separate Q/K/V and head dimensions. Run scaled dot-product attention. Then merge heads back and project through an output linear."

**Implementation.**

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_k = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.d_k).permute(2, 0, 3, 1, 4)
        # qkv shape: (3, B, H, T, d_k)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = scaled_dot_product_attention(q, k, v, mask, self.dropout)   # (B, H, T, d_k)
        out = out.transpose(1, 2).reshape(B, T, -1)                       # (B, T, d_model)
        return self.out(out)
```

**Walkthrough on shapes.** B=2, T=10, d_model=64, H=8 → d_k=8.

```
x:     (2, 10, 64)
qkv after linear:                             (2, 10, 192)
qkv.reshape(B, T, 3, H, d_k):                 (2, 10, 3, 8, 8)
qkv.permute(2, 0, 3, 1, 4):                   (3, 2, 8, 10, 8)
q, k, v each:                                 (2, 8, 10, 8)
attention output:                             (2, 8, 10, 8)
out.transpose(1, 2):                          (2, 10, 8, 8)
out.reshape(B, T, -1):                        (2, 10, 64)
```

**Common pitfalls.** Forgetting to call the output linear (`self.out`). Mishandling the head reshape — off-by-one in the permute is the classic mistake. Using `view` where you must `reshape` after transpose.

**Follow-up: "How would you make this support KV caching for inference?"** Cache `k` and `v` across forward calls; on each new token, only compute the new k/v row and concatenate with the cache. Mask is updated to extend by one position.

---

### B.3 FastAPI streaming endpoint with cancellation  (HIGHLY likely — JD names FastAPI + vLLM)

**Problem.** Build a `/chat` endpoint that streams tokens from a vLLM upstream server using SSE. Must handle client disconnection gracefully.

**Think-aloud strategy.** "Three things to get right: return `StreamingResponse` (not JSONResponse), use `httpx.AsyncClient.stream` to consume the upstream stream non-blocking, and check `request.is_disconnected()` in the generator loop so the upstream call is aborted when the client goes away."

**Implementation.**

```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 256

VLLM_URL = "http://vllm-service:8000/v1/completions"

@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    async def token_stream():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST", VLLM_URL,
                json={"prompt": req.prompt, "max_tokens": req.max_tokens, "stream": True},
            ) as upstream:
                async for chunk in upstream.aiter_lines():
                    if await request.is_disconnected():
                        # client gone — break, exiting the context manager cancels upstream
                        break
                    if chunk.startswith("data: "):
                        yield chunk + "\n\n"
    return StreamingResponse(token_stream(), media_type="text/event-stream")
```

**The three things they're checking.**
1. **You return `StreamingResponse`**, not a `JSONResponse` with a generator. The latter buffers the entire response and ships it once, losing the streaming behavior entirely.
2. **You handle disconnects** with `request.is_disconnected()`. Without it, your server keeps consuming GPU resources for a client that left ten seconds ago.
3. **You use `httpx.AsyncClient` with `client.stream`.** `requests` would block the event loop because it's synchronous; one stuck request stalls every other request on the same worker.

**Common mistakes.** Forgetting `media_type="text/event-stream"` — without it, browsers and proxies may buffer the response. Forgetting `\n\n` between SSE messages — clients won't parse them. Returning `httpx` errors directly to the client without translating to a 5xx response.

**Follow-up: "How would you add token-by-token usage tracking?"** Wrap the generator in a counter that increments on each `data:` line and emits a final usage chunk before closing the stream. Persist to your usage table after `yield` completes.

---

### B.4 Ray Serve deployment with batching

**Problem.** Build a Ray Serve deployment that loads an LLM and serves text generation, automatically batching incoming requests.

**Think-aloud strategy.** "Ray Serve has a built-in `@serve.batch` decorator that coalesces individual calls into a list. I declare `num_replicas` and `num_gpus` to tell Ray how to allocate, then write the `__call__` method that takes a list of prompts and returns a list of generations. Ray Serve handles the queuing and batching window."

**Implementation.**

```python
from ray import serve
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

@serve.deployment(num_replicas=2, ray_actor_options={"num_gpus": 1})
class LLMDeployment:
    def __init__(self, model_id: str = "meta-llama/Llama-3.2-1B"):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cuda"
        )

    @serve.batch(max_batch_size=16, batch_wait_timeout_s=0.01)
    async def __call__(self, prompts: list[str]) -> list[str]:
        inputs = self.tok(prompts, return_tensors="pt", padding=True).to("cuda")
        out = self.model.generate(**inputs, max_new_tokens=128)
        return self.tok.batch_decode(out, skip_special_tokens=True)

deploy = LLMDeployment.bind()
```

**Walkthrough on a load scenario.** Suppose ten clients call the endpoint within a five-millisecond window. Ray Serve's batcher waits up to ten milliseconds (`batch_wait_timeout_s`) or until sixteen prompts accumulate (`max_batch_size`), whichever comes first. With ten requests in five milliseconds, after another five milliseconds it dispatches a batch of ten — one forward pass instead of ten.

**Anticipated follow-up: "Why `serve.batch` vs vLLM's continuous batching?"** `serve.batch` does request-level batching with a static window — you wait, you batch, you run forward, everyone in the batch waits for the slowest sequence to finish. That's fine for embeddings, classification, or any model where outputs are similarly-sized. vLLM does iteration-level continuous batching where finished sequences free their slots immediately and new ones can join mid-batch. For LLMs with highly variable output lengths, vLLM is the right tool. `serve.batch` is great for upstream stages of a Ray Serve graph (re-rankers, classifiers).

**Common mistakes.** Forgetting `padding=True` — variable-length prompts will throw. Forgetting `device_map="cuda"` — the model loads on CPU and inference is glacially slow. Setting `max_batch_size` too high relative to GPU memory — OOMs under load.

---

### B.5 TTL + LRU cache (production pattern)

**Problem.** Implement a cache with both a maximum size (LRU eviction) and a per-entry TTL (time-to-live). Used for caching LLM responses, embeddings, feature lookups.

**Think-aloud strategy.** "Combine LRU semantics with TTL filtering. On `get`, check the timestamp; if expired, remove and return None. On `put`, store with current timestamp; evict oldest if over capacity. Use OrderedDict for O(1) LRU operations."

**Implementation.**

```python
import time
from collections import OrderedDict
from typing import Any

class TTLCache:
    def __init__(self, max_size: int, ttl_seconds: float):
        self.max = max_size
        self.ttl = ttl_seconds
        self.d: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        if key not in self.d:
            return None
        ts, val = self.d[key]
        if time.time() - ts > self.ttl:
            del self.d[key]
            return None
        self.d.move_to_end(key)
        return val

    def put(self, key: str, value: Any) -> None:
        self.d[key] = (time.time(), value)
        self.d.move_to_end(key)
        while len(self.d) > self.max:
            self.d.popitem(last=False)
```

**Walkthrough.** max_size=2, ttl=5.

```
t=0:  put("a", 1)        d = {"a":(0,1)}
t=1:  put("b", 2)        d = {"a":(0,1), "b":(1,2)}
t=2:  put("c", 3)        d = {"b":(1,2), "c":(2,3)}    ("a" evicted by LRU)
t=8:  get("b")           ts=1, age=7 > 5, expired → del, return None
                          d = {"c":(2,3)}
t=8:  get("c")           ts=2, age=6 > 5, expired → del, return None
                          d = {}
```

**Why production-relevant.** Caching LLM responses for repeated prompts is one of the highest-leverage cost optimizations in any LLM-powered product. Embedding caches similarly — embedding the same text twice is wasted compute. The kind of code reviewers nod at.

**Follow-up: "Make this thread-safe."** Wrap each method in a `threading.Lock`. For a hot path under heavy concurrency, consider a sharded approach (multiple sub-caches keyed by hash) to reduce lock contention.

**Follow-up: "How would you make this distributed?"** Replace the OrderedDict with Redis. `SETEX key value ttl` for put, `GET key` for get, and Redis handles eviction via `maxmemory-policy allkeys-lru`. The local code becomes a thin client.

---

### B.6 Cosine similarity matrix (numpy fluency test)

**Problem.** Given two matrices A `(n, d)` and B `(m, d)`, compute the `(n, m)` cosine similarity matrix.

**Think-aloud strategy.** "Cosine similarity is `dot(a, b) / (||a|| * ||b||)`. Vectorize with numpy: normalize each row to unit length, then matrix-multiply. Add a small epsilon to the denominator to avoid divide-by-zero on the rare zero-row."

**Implementation.**

```python
import numpy as np

def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A: (n, d), B: (m, d) → (n, m)"""
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A_n @ B_n.T
```

**Walkthrough on tiny example.** A = `[[1, 0], [0, 1]]`, B = `[[1, 1]]`.

```
||A row 0|| = 1     ||A row 1|| = 1     ||B row 0|| = sqrt(2)
A_n = [[1, 0], [0, 1]]
B_n = [[1/sqrt(2), 1/sqrt(2)]]
A_n @ B_n.T = [[1/sqrt(2)], [1/sqrt(2)]] ≈ [[0.707], [0.707]]
```

**Common mistakes.** Forgetting `keepdims=True` — without it, the norm collapses to shape `(n,)` and the broadcast division is wrong. Forgetting the epsilon — zero-vector rows divide by zero and produce NaN.

**Follow-up: "What if A and B are too large to fit in memory?"** Chunk: split A into batches of, say, 1000 rows; process each batch against the full B; concatenate. For huge B as well, use FAISS or a vector database — that's the production answer.

---

### B.7 In-memory RAG retriever

**Problem.** Build a simple semantic search retriever: add documents with `add(id, text)`, search top-k with `search(query, k)`. Use a passed-in `embed_fn`.

**Think-aloud strategy.** "Store documents with their embeddings. On search, embed the query, normalize, dot against the stored matrix, take top-k. Use `argpartition` for top-k since it's O(n) versus argsort's O(n log n)."

**Implementation.**

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class Doc:
    id: str
    text: str
    embedding: np.ndarray

class InMemoryRetriever:
    def __init__(self, embed_fn):
        self.embed = embed_fn
        self.docs: list[Doc] = []
        self.matrix: np.ndarray | None = None

    def add(self, doc_id: str, text: str) -> None:
        emb = self.embed(text)
        self.docs.append(Doc(doc_id, text, emb))
        self.matrix = np.stack([d.embedding for d in self.docs])

    def search(self, query: str, k: int = 5) -> list[Doc]:
        q = self.embed(query)
        q = q / (np.linalg.norm(q) + 1e-12)
        M = self.matrix / (np.linalg.norm(self.matrix, axis=1, keepdims=True) + 1e-12)
        scores = M @ q
        # argpartition: O(n) to find top-k unsorted; then sort just those k
        idx = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [self.docs[i] for i in idx]
```

**Anticipated follow-up: "Why argpartition then argsort, not just argsort?"** Argpartition is O(n) — it finds the k smallest (or largest) without sorting them. Argsort on the full array is O(n log n). For n=100,000 and k=5, that's a 14x speedup. We then argsort just the k partitioned values — that's O(k log k), trivially small.

**Anticipated follow-up: "How would you scale this to 10 million docs?"** Switch from numpy to FAISS (HNSW index) or a vector database (Pinecone, Weaviate, pgvector). Approximate nearest neighbor search is O(log n) per query with very high recall (>95% with proper tuning).

**Common mistakes.** Re-stacking the entire matrix on every `add` — fine for prototyping, terrible for production (use incremental updates). Forgetting to normalize — your "cosine similarity" becomes "dot product," which weights longer vectors more.

---

### B.8 Pydantic v2 model with validators

**Problem.** Build a request model for an LLM generation endpoint with validation: prompt non-empty, temperature in [0, 2], top_p in (0, 1], max_tokens in [1, 4096]. Strip whitespace and reject all-whitespace prompts.

**Think-aloud strategy.** "Pydantic v2 provides `Field` for simple bounds and `field_validator` for custom logic. The combination handles 95% of input hardening for an LLM API."

**Implementation.**

```python
from pydantic import BaseModel, Field, field_validator

class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8192)
    temperature: float = Field(ge=0.0, le=2.0, default=0.7)
    top_p: float = Field(gt=0.0, le=1.0, default=0.95)
    max_tokens: int = Field(ge=1, le=4096, default=512)
    user_id: str | None = None

    @field_validator("prompt")
    @classmethod
    def strip_prompt(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("prompt cannot be only whitespace")
        return v
```

**Why this is the kind of code reviewers nod at.** Most candidates skip input validation in interviews. Including it signals production maturity. The combination of `Field` constraints and `field_validator` covers both structural validation (length, range) and semantic validation (post-strip non-empty).

**Follow-up: "How do you propagate validation errors as nice JSON?"** FastAPI does this for free — Pydantic ValidationError gets serialized to a 422 with `{"detail": [{"loc": ..., "msg": ..., "type": ...}]}`. You can override the default handler with `@app.exception_handler(RequestValidationError)` to reformat.

---

### B.9 Causal-masked attention (variant of B.1)

**Problem.** Modify scaled dot-product attention to apply a causal mask automatically — token i can only attend to tokens 0..i.

**Think-aloud strategy.** "Build a triangular mask of shape (T, T) once at construction or on each forward; broadcast through the standard attention. `torch.tril(torch.ones(T, T))` gives a lower-triangular matrix of 1s; positions where the mask is 0 (above the diagonal) get masked out."

**Implementation.**

```python
import torch
import torch.nn.functional as F

def causal_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    # Q, K, V: (B, H, T, d_k)
    B, H, T, d_k = Q.shape
    mask = torch.tril(torch.ones(T, T, device=Q.device, dtype=torch.bool))   # (T, T)
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)                          # (B, H, T, T)
    scores = scores.masked_fill(~mask, float("-inf"))
    return F.softmax(scores, dim=-1) @ V
```

**Walkthrough on T=3.**

```
mask =
  [1 0 0]
  [1 1 0]
  [1 1 1]

token 0 attends to: position 0 only
token 1 attends to: positions 0, 1
token 2 attends to: positions 0, 1, 2
```

This is exactly the autoregressive behavior — at training time, every position predicts the next one based only on what came before, which matches the inference-time generation pattern.

**Common mistake.** Building the mask on the wrong device — if Q is on CUDA but the mask is on CPU, the masked_fill will silently move it (or error). Always device-pin: `device=Q.device`.

---

### B.10 Training loop with gradient accumulation

**Problem.** Write a PyTorch training loop that accumulates gradients over `accum_steps` mini-batches before stepping the optimizer, to simulate a larger effective batch size on a memory-constrained GPU.

**Think-aloud strategy.** "Gradient accumulation: divide the loss by accum_steps so the effective gradient is the average over N mini-batches. Call backward() each step but only zero_grad and step every Nth iteration. Crucial subtle detail: scale the loss BEFORE backward, not after."

**Implementation.**

```python
import torch
from torch.utils.data import DataLoader

def train(model, loader: DataLoader, optimizer, loss_fn, accum_steps: int = 4, epochs: int = 1):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        for step, (x, y) in enumerate(loader):
            x, y = x.cuda(), y.cuda()
            logits = model(x)
            loss = loss_fn(logits, y) / accum_steps     # scale before backward
            loss.backward()
            if (step + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
```

**Why this matters.** A 70B-parameter LoRA fine-tune with batch size 1 may need an effective batch of 32 to be stable. With 32 GPUs you can do data-parallel; with 1 GPU you do gradient accumulation over 32 steps. Same effective batch, slower wall time.

**Common mistakes.** Forgetting to divide the loss by `accum_steps` — gradients are then 4x too big and the learning rate is effectively 4x. Calling `optimizer.step()` every iteration — defeats the purpose. Calling `zero_grad()` every iteration — also defeats the purpose.

**Follow-up: "Combined with mixed precision (BF16)?"** Wrap the forward in `torch.autocast(dtype=torch.bfloat16)`. With BF16, no GradScaler is needed (unlike FP16). Step is the same.

---

## Part C — Quick warmups (60 seconds each)

These are mental-flexibility drills. Solve each in your head, talk out loud as you do, and write a three-line approach if pen is available.

1. **Reverse a linked list.** Three pointers — prev, curr, nxt. Iterate, reverse, advance.
2. **Detect a cycle in a linked list.** Floyd's tortoise and hare — fast pointer moves 2x speed of slow; if they meet, cycle exists.
3. **Validate balanced parentheses.** Stack — push opens, on close peek and pop, empty at end means valid.
4. **Find the kth largest element.** `heapq.nlargest(k, nums)[-1]` — O(n log k).
5. **Word frequency from a string.** `from collections import Counter; Counter(s.split())`.
6. **Flatten a nested list.** Recursion or stack-based iterative; check `isinstance(x, list)`.
7. **Are two strings anagrams?** `Counter(a) == Counter(b)`, or `sorted(a) == sorted(b)` if Counter feels heavy.
8. **Maximum subarray sum.** Kadane: `cur = max(x, cur + x); best = max(best, cur)`.
9. **Number of islands (DFS on grid).** Walk every cell, on '1' increment and flood-fill.
10. **Implement defaultdict(list) from scratch.** Subclass dict, override `__missing__` to insert and return `[]`.

---

## Part D — Live-coding meta strategy

Beyond knowing the algorithms, the interviewer is grading your *process*. Even when stuck, doing these things keeps you in the game:

1. **Read the problem aloud and restate it.** "So the input is X, the output is Y, and the constraints are Z. Edge cases I'm thinking about: empty input, single element, very large input." This both confirms understanding and buys you thinking time.

2. **Talk through the brute force first**, then say "but we can do better." This shows you can reason about the problem; it also gives you a fallback if the optimal solution doesn't materialize.

3. **Pick a data structure first**, then write the function signature with types, then the body. Walking that order forces you to commit to a plan before scribbling.

4. **Test with one example by hand** before declaring done. Walk the input through the loop iteration by iteration — catches off-by-one errors live, and signals to the interviewer that you're rigorous.

5. **Mention complexity unprompted.** "This is O(n) time and O(k) space because..." If you don't say it, they will write it down and ask later — better to volunteer.

6. **If you stall, narrate the blocker.** "Let me think out loud. The blocker is that I need to find both the value and its index, which is why a plain set won't work — I need a hash map." Interviewers grade reasoning at least as much as the final answer.

7. **When asked "is your code correct?", actually check it.** Re-read your code from top to bottom. Don't just say "yes." Senior engineers spot bugs by re-reading.

8. **If you don't know something, say so cleanly.** "I don't remember the exact API for that — I'd look it up. Here's how I'd structure the call." Way better than bluffing and getting caught.

---

## Part E — The four things that will make you stand out

1. **Type-hint everything.** `def f(x: list[int]) -> int:` — instant signal of production maturity.
2. **Name variables semantically.** `seen_indices` beats `d`; `left, right` beats `i, j`. The interviewer reads your code; readability matters.
3. **Talk about complexity proactively.** They will write it down whether you say it or not. Be the one who says it.
4. **Test the edge cases yourself.** Empty input, one element, duplicates, negative numbers, very large input. Walk through at least one before you say "done."

---

## How to say this in an interview

- "Let me restate the problem to make sure I have it right..."
- "The brute force is O(n²); I think we can do O(n) with a hash map."
- "I want to use a sliding window here — the right edge expands, the left contracts when we see a duplicate."
- "Let me trace through with the input you gave me. After the first iteration, left is at zero, right is at one, the map looks like..."
- "Time complexity is O(n) since each element is touched at most twice. Space is O(min(n, charset size))."
- "One edge case I want to check — what about an empty input?"

These cadences are what the interviewer is listening for. Practice saying them out loud, even on easy problems.

---

Continue to **[Chapter 21 — Slurm, DGX, HPC stack](21_slurm_dgx_hpc.md)**.
